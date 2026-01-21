"""
Configuration module for the RAG assistant.
Loads environment variables with proper fallbacks.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Find and load .env file from project root
current_file = Path(__file__).resolve()
project_root = current_file.parent
env_path = project_root / ".env"

# Try parent directory if .env not found
if not env_path.exists():
    env_path = project_root.parent / ".env"

# Load from .env.example if .env doesn't exist
if not env_path.exists():
    env_path = project_root / ".env.example"
    if not env_path.exists():
        env_path = project_root.parent / ".env.example"

if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")
else:
    print("Warning: No .env file found, using defaults")

# Chroma Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "codebase")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

# RAG Configuration
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "3"))  # Reduced from 5 to avoid context overflow
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))  # Reduced from 8000 for safer context limits
CHUNK_SIZE_LINES = int(os.getenv("CHUNK_SIZE_LINES", "150"))
OVERLAP_LINES = int(os.getenv("OVERLAP_LINES", "30"))

# Rate Limiting
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "20"))

# User Database
USERS_DB_PATH = os.getenv("USERS_DB_PATH", "users.db")

# Backend
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# Timeouts
OLLAMA_TIMEOUT = 120  # seconds
