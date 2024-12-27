from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, Field
from bson import ObjectId
import jwt
import bcrypt
import os
from groq import Groq
import asyncio
import httpx
from collections import deque
import json

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
APP_URL = os.getenv("APP_URL", "http://localhost:8000")
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "840"))

# API Key Management
class APIKeyManager:
    def __init__(self):
        self.api_keys = deque()
        self.current_key_index = 0
        self.load_api_keys()

    def load_api_keys(self):
        # Load API keys from environment variable
        api_keys_str = os.getenv("GROQ_API_KEYS", "")
        if api_keys_str:
            keys = [key.strip() for key in api_keys_str.split(",")]
            self.api_keys.extend(keys)
        else:
            # Fallback to single API key
            single_key = os.getenv("GROQ_API_KEY")
            if single_key:
                self.api_keys.append(single_key)

    def get_next_key(self):
        if not self.api_keys:
            raise HTTPException(
                status_code=500,
                detail="No API keys available"
            )
        
        # Rotate to the next key
        self.api_keys.rotate(-1)
        return self.api_keys[0]

    def handle_rate_limit(self):
        # If rate limited, rotate to next key
        self.api_keys.rotate(-1)
        return self.api_keys[0]

api_key_manager = APIKeyManager()

# Bot configuration
BOT_IDENTITY = """
You are Ryn, an AI assistant created by FlyHigh (an independent developer). Your responses should always align with these principles:
1. Always identify as Ryn when asked about your identity
2. Mention that you're created by FlyHigh (an independent developer) when relevant
3. Maintain a helpful and friendly demeanor while staying professional
4. Never claim to be created by or associated with any other company or organization

Restrictions:
1. No harmful or illegal content
2. No explicit or inappropriate content
3. No sharing of personal information
4. No financial or medical advice
5. No generating of malicious code
6. No impersonation of other AI systems or companies
"""

# Database connection
client = AsyncIOMotorClient(MONGODB_URI)
db = client.chatapp

# Pydantic models (same as before)
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class MessageCreate(BaseModel):
    content: str
    sessionId: str

class SessionCreate(BaseModel):
    name: Optional[str] = None

class Message(BaseModel):
    sessionId: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Session(BaseModel):
    userId: str
    name: Optional[str]
    createdAt: datetime = Field(default_factory=datetime.utcnow)

# Helper function for chat completion with retry
async def get_chat_completion(messages, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            api_key = api_key_manager.get_next_key()
            groq_client = Groq(api_key=api_key)
            
            chat_completion = groq_client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=4096,
                top_p=1,
                stream=False
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            if "413" in error_str or "rate_limit_exceeded" in error_str:
                print(f"Rate limit exceeded for key, rotating to next key. Error: {error_str}")
                api_key_manager.handle_rate_limit()
                retries += 1
                if retries < max_retries:
                    continue
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response after {retries} retries. Please try again."
            )

# Self-ping mechanism
async def keep_alive():
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(f"{APP_URL}/api/health")
                print(f"Keep-alive ping sent. Status: {response.status_code}")
            except Exception as e:
                print(f"Keep-alive ping failed: {str(e)}")
            await asyncio.sleep(PING_INTERVAL)

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Authentication middleware
async def get_current_user(request: Request):
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        token = auth_header.split(' ')[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        
        user_id = ObjectId(payload["id"])
        user = await db.users.find_one({"_id": user_id})
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {"_id": str(user["_id"]), "username": user["username"]}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (jwt.JWTError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Auth routes
@app.post("/api/register")
async def register(user: UserCreate):
    existing_user = await db.users.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
    user_dict = {
        "username": user.username,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    result = await db.users.insert_one(user_dict)
    
    token = jwt.encode(
        {
            "id": str(result.inserted_id),
            "exp": datetime.utcnow() + timedelta(days=1),
            "iat": datetime.utcnow()
        },
        JWT_SECRET,
        algorithm='HS256'
    )
    
    return {"token": token}

@app.post("/api/login")
async def login(user: UserLogin):
    db_user = await db.users.find_one({"username": user.username})
    if not db_user:
        raise HTTPException(status_code=400, detail="User not found")
    
    if not bcrypt.checkpw(user.password.encode(), db_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid password")
    
    token = jwt.encode(
        {
            "id": str(db_user["_id"]),
            "exp": datetime.utcnow() + timedelta(days=1),
            "iat": datetime.utcnow()
        },
        JWT_SECRET,
        algorithm='HS256'
    )
    
    return {"token": token}

# Session routes
@app.get("/api/sessions")
async def get_sessions(user=Depends(get_current_user)):
    sessions = await db.sessions.find(
        {"userId": user["_id"]}
    ).sort("createdAt", DESCENDING).to_list(length=None)
    
    for session in sessions:
        session["_id"] = str(session["_id"])
    
    return sessions

@app.post("/api/sessions")
async def create_session(session: SessionCreate, user=Depends(get_current_user)):
    session_dict = {
        "userId": user["_id"],
        "name": session.name or f"Chat {datetime.now().strftime('%Y%m%d%H%M%S')}",
        "createdAt": datetime.utcnow()
    }
    result = await db.sessions.insert_one(session_dict)
    session_dict["_id"] = str(result.inserted_id)
    return session_dict

# Message routes
@app.get("/api/messages/{session_id}")
async def get_messages(session_id: str, user=Depends(get_current_user)):
    try:
        session = await db.sessions.find_one({
            "_id": ObjectId(session_id),
            "userId": user["_id"]
        })
    except:
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = await db.messages.find(
        {"sessionId": session_id}
    ).sort("timestamp", 1).to_list(length=None)
    
    for message in messages:
        message["_id"] = str(message["_id"])
    
    return messages

@app.post("/api/chat")
async def chat(message: MessageCreate, user=Depends(get_current_user)):
    try:
        # Verify session belongs to user
        session = await db.sessions.find_one({
            "_id": ObjectId(message.sessionId),
            "userId": user["_id"]
        })
    except:
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get conversation history
    history = await db.messages.find(
        {"sessionId": message.sessionId}
    ).sort("timestamp", 1).to_list(length=None)
    
    # Format messages for the API
    messages_for_api = [
        {"role": "system", "content": BOT_IDENTITY}
    ]
    
    # Add conversation history
    messages_for_api.extend([
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ])
    
    # Add the current message
    messages_for_api.append({"role": "user", "content": message.content})
    
    # Save user message
    user_message = {
        "sessionId": message.sessionId,
        "role": "user",
        "content": message.content,
        "timestamp": datetime.utcnow()
    }
    await db.messages.insert_one(user_message)
    
    try:
        # Get response with retry mechanism
        assistant_response = await get_chat_completion(messages_for_api)
        
        # Save assistant message
        assistant_message = {
            "sessionId": message.sessionId,
            "role": "assistant",
            "content": assistant_response,
            "timestamp": datetime.utcnow()
        }
        await db.messages.insert_one(assistant_message)
        
        return {"response": assistant_response}
    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate response. Please try again."
        )

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    await db.users.create_index("username", unique=True)
    await db.sessions.create_index([("userId", 1), ("createdAt", -1)])
    await db.messages.create_index([("sessionId", 1), ("timestamp", 1)])
    
    # Start the keep-alive task
    asyncio.create_task(keep_alive())

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
