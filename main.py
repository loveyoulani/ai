from contextlib import asynccontextmanager
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
import uvicorn
import os
from groq import Groq
import asyncio
import httpx
from collections import deque
import json
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables with defaults
class Config:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
    APP_URL = os.getenv("APP_URL", "http://localhost:8000")
    PING_INTERVAL = int(os.getenv("PING_INTERVAL", "840"))
    PORT = int(os.getenv("PORT", "8000"))

# Enhanced API Key Management
class APIKeyManager:
    def __init__(self):
        self.api_keys = deque()
        self.current_key_index = 0
        self.load_api_keys()
        logger.info(f"Initialized API Key Manager with {len(self.api_keys)} keys")

    def load_api_keys(self):
        api_keys_str = os.getenv("GROQ_API_KEYS", "")
        if api_keys_str:
            keys = [key.strip() for key in api_keys_str.split(",")]
            self.api_keys.extend(keys)
        else:
            single_key = os.getenv("GROQ_API_KEY")
            if single_key:
                self.api_keys.append(single_key)
            else:
                logger.error("No API keys found in environment variables!")

    def get_next_key(self):
        if not self.api_keys:
            logger.error("No API keys available")
            raise HTTPException(status_code=500, detail="No API keys configured")
        self.api_keys.rotate(-1)
        return self.api_keys[0]

    def handle_rate_limit(self):
        logger.warning("Rate limit hit, rotating to next API key")
        self.api_keys.rotate(-1)
        return self.api_keys[0]

api_key_manager = APIKeyManager()

# Pydantic models with enhanced validation
class UserCreate(BaseModel):
    username: str
    password: str

    class Config:
        min_anystr_length = 3
        max_anystr_length = 50

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

# Enhanced chat completion with better error handling
async def get_chat_completion(messages, max_retries=3):
    retries = 0
    last_error = None

    while retries < max_retries:
        try:
            api_key = api_key_manager.get_next_key()
            groq_client = Groq(api_key=api_key)
            
            logger.info(f"Attempting chat completion (attempt {retries + 1}/{max_retries})")
            
            chat_completion = groq_client.chat.completions.create(
                messages=messages,
                model="deepseek-r1-distill-llama-70b",
                temperature=0.6,
                max_tokens=4096,
                top_p=1,
                stream=False
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            logger.error(f"Chat completion error: {error_str}")
            
            if "413" in error_str or "rate_limit_exceeded" in error_str:
                logger.warning("Rate limit exceeded, rotating API key")
                api_key_manager.handle_rate_limit()
                retries += 1
                await asyncio.sleep(1)  # Add delay between retries
                continue
            
            raise HTTPException(
                status_code=500,
                detail=f"Chat completion error: {error_str}"
            )
    
    logger.error(f"Failed after {max_retries} attempts. Last error: {last_error}")
    raise HTTPException(
        status_code=500,
        detail=f"Failed to generate response after {max_retries} retries"
    )

# Improved keep-alive mechanism
async def keep_alive():
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(f"{Config.APP_URL}/api/health")
                logger.debug(f"Keep-alive ping sent. Status: {response.status_code}")
            except Exception as e:
                logger.error(f"Keep-alive ping failed: {str(e)}")
            await asyncio.sleep(Config.PING_INTERVAL)

# Enhanced lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    client = None
    try:
        logger.info("Starting application...")
        client = AsyncIOMotorClient(Config.MONGODB_URI)
        app.state.db = client.chatapp
        
        # Create indexes
        await app.state.db.users.create_index("username", unique=True)
        await app.state.db.sessions.create_index([("userId", 1), ("createdAt", -1)])
        await app.state.db.messages.create_index([("sessionId", 1), ("timestamp", 1)])
        
        # Start keep-alive
        keep_alive_task = asyncio.create_task(keep_alive())
        
        yield
        
        # Shutdown
        logger.info("Shutting down application...")
        keep_alive_task.cancel()
        try:
            await keep_alive_task
        except asyncio.CancelledError:
            pass
    except Exception as e:
        logger.error(f"Lifespan error: {str(e)}")
        raise
    finally:
        if client:
            client.close()

# Initialize FastAPI with enhanced error handling
app = FastAPI(lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced authentication middleware
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
        payload = jwt.decode(token, Config.JWT_SECRET, algorithms=['HS256'])
        
        user_id = ObjectId(payload["id"])
        user = await app.state.db.users.find_one({"_id": user_id})
        
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
    except (jwt.JWTError, ValueError) as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Enhanced routes with better error handling
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow()}

@app.post("/api/register")
async def register(user: UserCreate):
    try:
        existing_user = await app.state.db.users.find_one({"username": user.username})
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        hashed_password = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
        user_dict = {
            "username": user.username,
            "password": hashed_password,
            "created_at": datetime.utcnow()
        }
        
        result = await app.state.db.users.insert_one(user_dict)
        
        token = jwt.encode(
            {
                "id": str(result.inserted_id),
                "exp": datetime.utcnow() + timedelta(days=1),
                "iat": datetime.utcnow()
            },
            Config.JWT_SECRET,
            algorithm='HS256'
        )
        
        return {"token": token}
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
async def login(user: UserLogin):
    try:
        db_user = await app.state.db.users.find_one({"username": user.username})
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
            Config.JWT_SECRET,
            algorithm='HS256'
        )
        
        return {"token": token}
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def get_sessions(user=Depends(get_current_user)):
    try:
        sessions = await app.state.db.sessions.find(
            {"userId": user["_id"]}
        ).sort("createdAt", DESCENDING).to_list(length=None)
        
        for session in sessions:
            session["_id"] = str(session["_id"])
        
        return sessions
    except Exception as e:
        logger.error(f"Error fetching sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions")
async def create_session(session: SessionCreate, user=Depends(get_current_user)):
    try:
        session_dict = {
            "userId": user["_id"],
            "name": session.name or f"Chat {datetime.now().strftime('%Y%m%d%H%M%S')}",
            "createdAt": datetime.utcnow()
        }
        result = await app.state.db.sessions.insert_one(session_dict)
        session_dict["_id"] = str(result.inserted_id)
        return session_dict
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/messages/{session_id}")
async def get_messages(session_id: str, user=Depends(get_current_user)):
    try:
        session = await app.state.db.sessions.find_one({
            "_id": ObjectId(session_id),
            "userId": user["_id"]
        })
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = await app.state.db.messages.find(
            {"sessionId": session_id}
        ).sort("timestamp", 1).to_list(length=None)
        
        for message in messages:
            message["_id"] = str(message["_id"])
        
        return messages
    except Exception as e:
        logger.error(f"Error fetching messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(message: MessageCreate, user=Depends(get_current_user)):
    try:
        logger.info(f"Processing chat request for session: {message.sessionId}")
        
        # Validate session
        try:
            session = await app.state.db.sessions.find_one({
                "_id": ObjectId(message.sessionId),
                "userId": user["_id"]
            })
        except Exception as e:
            logger.error(f"Session validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid session ID: {str(e)}")
            
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get message history
        history = await app.state.db.messages.find(
            {"sessionId": message.sessionId}
        ).sort("timestamp", 1).to_list(length=None)
        
        messages_for_api = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ]
        messages_for_api.append({"role": "user", "content": message.content})
        
        # Save user message
        user_message = {
            "sessionId": message.sessionId,
            "role": "user",
            "content": message.content,
            "timestamp": datetime.utcnow()
        }
        await app.state.db.messages.insert_one(user_message)
        logger.info("Saved user message")
        
        # Get and save assistant response
        try:
            logger.info("Requesting chat completion")
            assistant_response = await get_chat_completion(messages_for_api)
            
            assistant_message = {
                "sessionId": message.sessionId,
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.utcnow()
            }
            await app.state.db.messages.insert_one(assistant_message)
            logger.info("Saved assistant response")
            
            return {"response": assistant_response}
            
        except Exception as e:
            logger.error(f"Chat completion error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
            logger.error(f"Unexpected error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

# Helper function to validate MongoDB connection
async def check_db_connection():
    try:
        await app.state.db.command('ping')
        return True
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return False

# Enhanced health check endpoint
@app.get("/api/health")
async def health_check():
    db_status = "connected" if await check_db_connection() else "disconnected"
    api_keys_status = "available" if len(api_key_manager.api_keys) > 0 else "unavailable"
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow(),
        "database": db_status,
        "api_keys": api_keys_status,
        "version": "1.0.0"
    }

# Main entry point
if __name__ == "__main__":
    # Verify required environment variables
    required_vars = ["MONGODB_URI", "JWT_SECRET"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    # Verify API keys
    if not api_key_manager.api_keys:
        logger.error("No GROQ API keys configured. Please set GROQ_API_KEY or GROQ_API_KEYS environment variable.")
        exit(1)
    
    logger.info(f"Starting server on port {Config.PORT}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.PORT,
        reload=False,
        log_level="info"
    )
