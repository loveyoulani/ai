from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
import bcrypt
import os
from pydantic import BaseModel
import httpx
from groq import Groq

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

# Database connection
client = AsyncIOMotorClient(MONGODB_URI)
db = client.chatapp

# Initialize Groq client without additional configuration
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

# Authentication middleware
async def get_current_user(request: Request):
    token = request.headers.get('Authorization')
    if not token or not token.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    token = token.split(' ')[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        user = await db.users.find_one({"_id": payload["id"]})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Auth routes
@app.post("/api/register")
async def register(user: UserCreate):
    existing_user = await db.users.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
    user_dict = {
        "username": user.username,
        "password": hashed_password
    }
    
    result = await db.users.insert_one(user_dict)
    token = jwt.encode(
        {"id": str(result.inserted_id), "exp": datetime.utcnow() + timedelta(days=1)},
        JWT_SECRET
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
        {"id": str(db_user["_id"]), "exp": datetime.utcnow() + timedelta(days=1)},
        JWT_SECRET
    )
    
    return {"token": token}

# Session routes
@app.get("/api/sessions")
async def get_sessions(user=Depends(get_current_user)):
    sessions = await db.sessions.find(
        {"userId": str(user["_id"])}
    ).sort("createdAt", DESCENDING).to_list(length=None)
    return sessions

@app.post("/api/sessions")
async def create_session(session: SessionCreate, user=Depends(get_current_user)):
    session_dict = {
        "userId": str(user["_id"]),
        "name": session.name or f"Chat {datetime.now().strftime('%Y%m%d%H%M%S')}",
        "createdAt": datetime.utcnow()
    }
    result = await db.sessions.insert_one(session_dict)
    session_dict["_id"] = str(result.inserted_id)
    return session_dict

# Message routes
@app.get("/api/messages/{session_id}")
async def get_messages(session_id: str, user=Depends(get_current_user)):
    session = await db.sessions.find_one({
        "_id": session_id,
        "userId": str(user["_id"])
    })
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = await db.messages.find(
        {"sessionId": session_id}
    ).sort("timestamp", 1).to_list(length=None)
    return messages

@app.post("/api/chat")
async def chat(message: MessageCreate, user=Depends(get_current_user)):
    # Verify session belongs to user
    session = await db.sessions.find_one({
        "_id": message.sessionId,
        "userId": str(user["_id"])
    })
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save user message
    user_message = {
        "sessionId": message.sessionId,
        "role": "user",
        "content": message.content,
        "timestamp": datetime.utcnow()
    }
    await db.messages.insert_one(user_message)
    
    try:
        # Get response from Groq using synchronous client
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": message.content}],
            model="mixtral-8x7b-32768",
            temperature=0.7
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
