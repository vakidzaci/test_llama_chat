# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** running with models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Index Your Codebase
```bash
# Replace with your Python repo path
python scripts/index_codebase.py /path/to/your/python/repo
```

### 3. Start Backend (Terminal 1)
```bash
./run_backend.sh
# Or: python backend/main.py
```

### 4. Start UI (Terminal 2)
```bash
./run_ui.sh
# Or: streamlit run ui/app.py
```

## First Query

1. Open UI at `http://localhost:8501`
2. Register a new account
3. Ask a question: "What does this codebase do?"
4. See results with source citations!

## Test with cURL

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "test123"}'

# Save the returned api_key, then:
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"question": "How does authentication work?"}'
```

## Default Configuration

- **Backend**: http://localhost:8000
- **UI**: http://localhost:8501
- **Rate Limit**: 20 requests/minute
- **Top-K**: 5 chunks
- **Chunk Size**: 150 lines
- **Overlap**: 30 lines

## Troubleshooting

### "Cannot connect to Ollama"
→ Start Ollama: `ollama serve`

### "Collection not initialized"
→ Run indexing: `python scripts/index_codebase.py /path/to/repo`

### "Cannot connect to backend" (in UI)
→ Start backend: `./run_backend.sh`

## Next Steps

- See `README.md` for full documentation
- Check `http://localhost:8000/docs` for API docs
- Edit `.env` for custom configuration
- Try `--reset` flag to reindex: `python scripts/index_codebase.py /path/to/repo --reset`

Happy querying! 🎉
