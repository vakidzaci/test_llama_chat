"""
FastAPI backend for the RAG assistant.
Provides authentication, rate limiting, and RAG query endpoints.
"""
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from backend.user_db import db
from backend.rate_limiter import rate_limiter
from backend.rag_service import rag_service


app = FastAPI(
    title="RAG Codebase Assistant",
    description="Multi-user codebase assistant with RAG",
    version="1.0.0"
)

# CORS middleware - permissive for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    api_key: str
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=20)


class SourceInfo(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]


class HealthResponse(BaseModel):
    status: str
    collection: str
    persist_dir: str
    chunk_count: int
    ollama_connected: bool
    chroma_connected: bool


# Helper function to verify API key
def verify_api_key_header(x_api_key: str = Header(...)) -> dict:
    """Verify API key from header and return user info."""
    user_info = db.verify_api_key(x_api_key)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return user_info


# Helper function to check rate limit
def check_rate_limit(user_info: dict):
    """Check if user has exceeded rate limit."""
    user_id = user_info["user_id"]
    if not rate_limiter.is_allowed(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {config.RATE_LIMIT_RPM} requests per minute."
        )


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    health_info = rag_service.health_check()
    
    return HealthResponse(
        status="ok" if health_info.get("chroma_connected") else "degraded",
        collection=health_info["collection_name"],
        persist_dir=health_info["persist_dir"],
        chunk_count=health_info["chunk_count"],
        ollama_connected=health_info.get("ollama_connected", False),
        chroma_connected=health_info.get("chroma_connected", False)
    )


@app.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """Register a new user and return an API key."""
    api_key = db.register_user(request.username, request.password)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    return AuthResponse(
        api_key=api_key,
        message=f"User '{request.username}' registered successfully"
    )


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Login and return an API key."""
    api_key = db.login_user(request.username, request.password)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    return AuthResponse(
        api_key=api_key,
        message=f"Login successful"
    )


@app.post("/rag/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Query the codebase using RAG.

    Headers:
        X-API-Key: Your API key from registration/login

    Body:
        question: Your question about the codebase
        top_k: Optional number of chunks to retrieve (default: 5)
    """
    # Verify API key
    user_info = verify_api_key_header(x_api_key)

    # Check rate limit
    check_rate_limit(user_info)

    # Perform RAG query
    try:
        result = rag_service.query(request.question, request.top_k)

        sources = [
            SourceInfo(
                file_path=s["file_path"],
                start_line=s["start_line"],
                end_line=s["end_line"],
                distance=s["distance"]
            )
            for s in result["sources"]
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Codebase Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "register": "/auth/register",
            "login": "/auth/login",
            "query": "/rag/query"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        log_level="info"
    )