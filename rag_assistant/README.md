# RAG Codebase Assistant

A multi-user codebase assistant built with RAG (Retrieval-Augmented Generation) for querying Python repositories.

## 🏗️ Architecture

- **Backend**: FastAPI with authentication, rate limiting, and RAG endpoints
- **Frontend**: Streamlit UI for user interaction
- **Vector DB**: Chroma (single shared collection for all users)
- **LLM Integration**: Ollama via LangChain Community (for embeddings and chat)
- **User DB**: SQLite with hashed passwords and API keys

## ✨ Features

- ✅ Single shared Chroma collection for all users
- ✅ User authentication with API keys
- ✅ Per-user rate limiting
- ✅ Retrieval-only LLM access (no full repo context)
- ✅ Line-based chunking with overlap
- ✅ Source citation with file paths and line ranges
- ✅ Configurable via environment variables
- ✅ **LangChain Community integration** for robust LLM/vector store operations

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama (https://ollama.ai)
   # Then pull required models:
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd rag_assistant

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env if needed (defaults should work for local development)
```

Default configuration:
- Ollama: `http://localhost:11434`
- Embed model: `nomic-embed-text`
- Chat model: `llama3.2`
- Rate limit: 20 requests/minute
- Top-K chunks: 5
- Max context: 8000 characters

### 3. Index a Codebase

```bash
# Index a Python repository
python scripts/index_codebase.py /path/to/your/python/repo

# To reset and reindex
python scripts/index_codebase.py /path/to/your/python/repo --reset
```

Example output:
```
Indexing repository: /path/to/repo
Found 45 Python files.
Created 312 chunks.
Generating embeddings and adding to collection...
✅ Indexing complete! Total chunks in collection: 312
```

### 4. Start the Backend

```bash
# From project root
python backend/main.py

# Or using uvicorn directly
uvicorn backend.main:app --reload
```

Backend will be available at: `http://localhost:8000`

API docs: `http://localhost:8000/docs`

### 5. Start the UI

```bash
# In a new terminal
streamlit run ui/app.py
```

UI will open at: `http://localhost:8501`

## 📁 Project Structure

```
rag_assistant/
├── backend/
│   ├── main.py           # FastAPI application
│   ├── user_db.py        # User authentication & API keys
│   ├── rate_limiter.py   # Rate limiting logic
│   └── rag_service.py    # RAG pipeline (embed, retrieve, generate)
├── scripts/
│   └── index_codebase.py # Codebase indexing script
├── ui/
│   └── app.py            # Streamlit user interface
├── config.py             # Shared configuration
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── README.md            # This file
```

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

Returns system status and collection info.

### Register User
```bash
POST /auth/register
Content-Type: application/json

{
  "username": "john_doe",
  "password": "secure_password"
}

Response:
{
  "api_key": "your-api-key-here",
  "message": "User 'john_doe' registered successfully"
}
```

### Login
```bash
POST /auth/login
Content-Type: application/json

{
  "username": "john_doe",
  "password": "secure_password"
}

Response:
{
  "api_key": "your-api-key-here",
  "message": "Login successful"
}
```

### Query RAG
```bash
POST /rag/query
X-API-Key: your-api-key-here
Content-Type: application/json

{
  "question": "How does authentication work?",
  "top_k": 5
}

Response:
{
  "answer": "The authentication system uses...",
  "sources": [
    {
      "file_path": "backend/user_db.py",
      "start_line": 45,
      "end_line": 95,
      "distance": 0.234
    }
  ]
}
```

## 🔒 Security Features

- **Password Hashing**: Bcrypt with salt
- **API Key Hashing**: SHA-256 hashed storage
- **Rate Limiting**: In-memory sliding window (20 req/min default)
- **Authentication**: Required for all RAG queries

## ⚙️ Configuration Options

Edit `.env` to customize:

```bash
# Chroma
CHROMA_PERSIST_DIR=chroma_db
CHROMA_COLLECTION=codebase

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=llama3.2

# RAG
TOP_K_DEFAULT=5
MAX_CONTEXT_CHARS=8000
CHUNK_SIZE_LINES=150
OVERLAP_LINES=30

# Rate Limiting
RATE_LIMIT_RPM=20

# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

## 📊 Chunking Strategy

- **Method**: Line-based with overlap
- **Default Size**: 150 lines per chunk
- **Overlap**: 30 lines between chunks
- **Chunk ID**: Stable hash of `file_path::start_line::end_line::content_hash`

## 🎯 Design Constraints

### Hard Constraints (Implemented)
✅ Single Chroma collection (`codebase`)  
✅ Single persistent directory (`chroma_db`)  
✅ No per-user collections or namespaces  
✅ LLM sees only retrieved chunks (no full repo)  
✅ Ollama for embeddings and chat  
✅ FastAPI backend + Streamlit UI  
✅ Multi-user auth + rate limiting  
✅ SQLite for user storage  
✅ Hashed passwords and API keys  

### Known Limitations

1. **Rate limiting**: In-memory only (single instance)
2. **Old chunks**: Manual `--reset` required if not using upsert
3. **Model changes**: Reindex required if embedding model changes
4. **Context limits**: Enforced at `MAX_CONTEXT_CHARS`

## 🧪 Testing the System

### 1. Test Health Check
```bash
curl http://localhost:8000/health
```

### 2. Register a User
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass123"}'
```

### 3. Query the System
```bash
# Save your API key from registration
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{"question": "What does this codebase do?", "top_k": 5}'
```

## 🐛 Troubleshooting

### Ollama Not Running
```
Error: Cannot connect to Ollama
```
**Solution**: Start Ollama and ensure models are pulled:
```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Collection Not Found
```
Error: Collection not initialized
```
**Solution**: Run the indexing script first:
```bash
python scripts/index_codebase.py /path/to/repo
```

### Backend Connection Failed (UI)
```
Cannot connect to backend
```
**Solution**: Ensure FastAPI backend is running:
```bash
python backend/main.py
```

### Rate Limit Exceeded
```
429: Rate limit exceeded
```
**Solution**: Wait 60 seconds or adjust `RATE_LIMIT_RPM` in `.env`

## 📚 Example Usage

```python
# Example query via Python
import requests

API_KEY = "your-api-key"
BACKEND_URL = "http://localhost:8000"

response = requests.post(
    f"{BACKEND_URL}/rag/query",
    headers={"X-API-Key": API_KEY},
    json={
        "question": "How is user authentication implemented?",
        "top_k": 5
    }
)

result = response.json()
print(result["answer"])
for source in result["sources"]:
    print(f"  - {source['file_path']} (lines {source['start_line']}-{source['end_line']})")
```

## 🔧 Development

### LangChain Integration

This system uses **LangChain Community** for:
- **OllamaEmbeddings**: Generates embeddings for queries and documents
- **ChatOllama**: Handles chat completions with context-aware responses
- **Chroma Vectorstore**: Manages vector storage and similarity search

Benefits:
- Cleaner abstractions over raw API calls
- Better error handling and retries
- Standardized interfaces across LLM providers
- Easy to swap providers (e.g., OpenAI, Anthropic) by changing components

### Running with Auto-Reload
```bash
# Backend with auto-reload
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Streamlit with auto-reload (default behavior)
streamlit run ui/app.py
```

### Accessing API Docs
FastAPI provides interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📝 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

This is a reference implementation based on specific constraints. Adapt as needed for your use case!

## ✅ Acceptance Checklist

- [x] `python scripts/index_codebase.py /path/to/repo` creates/updates `chroma_db/` and collection `codebase`
- [x] FastAPI runs and `/health` returns correct persist dir + collection
- [x] User can register/login and receive an API key
- [x] Streamlit can authenticate and call `/rag/query`
- [x] `/rag/query` returns an answer + sources
- [x] Answer is generated only from retrieved context
- [x] No extra collections, no per-user DB separation, no repo-wide context leakage
- [x] Password and API key hashing implemented
- [x] Rate limiting per user
- [x] Configuration via .env
- [x] Proper error handling for Ollama/Chroma failures
