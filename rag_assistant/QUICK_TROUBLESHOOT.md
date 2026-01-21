# Quick Troubleshooting Commands

## 🚨 Getting "distance" error? Run these:

### 1. Diagnose the issue
```bash
cd rag_assistant
python diagnose_collection.py
```

### 2. If it says "Collection empty" or "Not found"
```bash
# Index your codebase
python scripts/index_codebase.py C:\path\to\your\python\repo

# Windows example:
python scripts/index_codebase.py C:\Users\YourName\Documents\my-project
```

### 3. If collection exists but still errors
```bash
# Re-index with reset
python scripts/index_codebase.py C:\path\to\your\python\repo --reset
```

### 4. Restart backend
```bash
# Stop with Ctrl+C, then:
python backend/main.py
```

### 5. Test query
```bash
curl -X POST http://localhost:8000/rag/query -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" -d "{\"question\": \"test\"}"
```

---

## 📋 Other Common Issues

### "Ollama not running"
```bash
# Start Ollama
ollama serve

# In another terminal, verify models:
ollama list
```

### "Collection not initialized"
```bash
# Check what's in chroma_db
dir chroma_db

# Should see files. If empty, run indexing
python scripts/index_codebase.py /path/to/repo
```

### "Context length exceeded"
```bash
# Edit .env
TOP_K_DEFAULT=1
MAX_CONTEXT_CHARS=2000

# Restart backend
python backend/main.py
```

---

## ✅ Verification Checklist

Run these to verify everything works:

```bash
# 1. Check Ollama
curl http://localhost:11434/api/tags

# 2. Check backend
curl http://localhost:8000/health

# 3. Diagnose collection
python diagnose_collection.py

# 4. Test query (after registration)
curl -X POST http://localhost:8000/rag/query -H "X-API-Key: KEY" -H "Content-Type: application/json" -d "{\"question\": \"test\"}"
```

All should return success! ✅

---

## 🆘 Still stuck?

1. Check backend console for error messages
2. Look in `FIX_DISTANCE_ERROR.md` for detailed solutions
3. Make sure you downloaded the latest version
4. Verify you're using Python 3.8+
5. Check all dependencies installed: `pip install -r requirements.txt`
