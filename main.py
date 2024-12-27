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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
APP_URL = os.getenv("APP_URL", "http://localhost:8000")  # Add this line
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "840"))  # 14 minutes in seconds

# Database connection
client = AsyncIOMotorClient(MONGODB_URI)
db = client.chatapp

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Pydantic models
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
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]
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
        # Get response from Groq with conversation history
        chat_completion = groq_client.chat.completions.create(
            messages=messages_for_api,
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=4096,
            top_p=1,
            stream=False
        )
        
        assistant_response = chat_completion.choices[0].message.content
        
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
