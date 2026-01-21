# Fixing "Input Length Exceeds Context Length" Error

## Error Message
```
The input length exceeds the context length
```
or
```
Context too large for the model
```

## What This Means

Your query + retrieved code chunks are too large for the Ollama model's context window. Most models have limits:
- `llama3.2` (default): ~2048 tokens
- `llama3.2:8k`: ~8192 tokens  
- `llama3.1:8b`: ~8192 tokens

**1 token ≈ 4 characters**, so 2048 tokens ≈ 8000 characters total (including prompt, context, and question)

---

## ✅ Immediate Solutions (Pick One)

### Solution 1: Reduce top_k (Easiest)

**Query with fewer chunks:**

```bash
# Instead of default (3 chunks)
curl -X POST http://localhost:8000/rag/query \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "your question", "top_k": 1}'
```

**In Streamlit UI**: Slide the "top_k" slider to 1 or 2

---

### Solution 2: Use a Larger Model (Recommended)

**Switch to a model with bigger context:**

```bash
# Stop current Ollama
# Pull a model with larger context
ollama pull llama3.2:8k

# or
ollama pull llama3.1:8b
```

**Update `.env`:**
```bash
OLLAMA_CHAT_MODEL=llama3.2:8k
```

**Restart backend:**
```bash
python backend/main.py
```

---

### Solution 3: Reduce MAX_CONTEXT_CHARS

**Edit `.env`:**
```bash
MAX_CONTEXT_CHARS=2000  # Reduced from 4000
TOP_K_DEFAULT=2          # Reduced from 3
```

**Restart backend**

---

## 🔧 Understanding the Fix (Already Applied)

The latest version has **smarter context management**:

### What Changed:

**Before:**
- Would try to fit 5 chunks regardless of size
- Simple character truncation (could cut mid-sentence)
- No awareness of model limits

**After (New Version):**
- Default reduced to 3 chunks (from 5)
- Context limit reduced to 4000 chars (from 8000)
- Smart truncation that preserves whole chunks
- Stops adding chunks when limit reached
- Clear error messages with solutions

### New Context Building Logic:

```python
# Builds context chunk by chunk
# Stops when MAX_CONTEXT_CHARS reached
# Preserves complete chunks
# Adds notice if chunks were skipped

if chunks_used < len(chunks):
    context += f"\n[Showing {chunks_used} of {len(chunks)} chunks]"
```

---

## 📊 Current Configuration (Latest Version)

**Default Settings:**
```
TOP_K_DEFAULT = 3          # Retrieves 3 chunks
MAX_CONTEXT_CHARS = 4000   # Limits to 4000 characters
Model: llama3.2            # ~2048 token limit
```

**Safe for most queries!**

---

## 🎯 Recommended Model Configurations

### For Small Context Models (llama3.2)
```env
TOP_K_DEFAULT=2
MAX_CONTEXT_CHARS=3000
```

### For Medium Context Models (llama3.2:8k, llama3.1:8b)
```env
TOP_K_DEFAULT=5
MAX_CONTEXT_CHARS=8000
OLLAMA_CHAT_MODEL=llama3.2:8k
```

### For Large Context Models (claude, gpt-4)
```env
TOP_K_DEFAULT=10
MAX_CONTEXT_CHARS=20000
# (If using OpenAI/Anthropic via LangChain)
```

---

## 🔍 Debugging Steps

### 1. Check Your Current Model
```bash
curl http://localhost:11434/api/tags
```

Look for your chat model and its context window size.

### 2. Test with Minimal Query
```bash
# Try with just 1 chunk
curl -X POST http://localhost:8000/rag/query \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "top_k": 1}'
```

If this works, the issue is context size.

### 3. Check Your Configuration
```bash
cat .env | grep -E "(TOP_K|MAX_CONTEXT|CHAT_MODEL)"
```

Should show:
```
TOP_K_DEFAULT=3
MAX_CONTEXT_CHARS=4000
OLLAMA_CHAT_MODEL=llama3.2
```

---

## 💡 Advanced Solutions

### Option A: Smaller Chunks (Reindex Required)

**Edit `.env` before indexing:**
```env
CHUNK_SIZE_LINES=80   # Reduced from 150
OVERLAP_LINES=15      # Reduced from 30
```

**Reindex:**
```bash
python scripts/index_codebase.py /path/to/repo --reset
```

**Benefit**: Each chunk is smaller, so more fit in context.

---

### Option B: Dynamic top_k Based on Question

**Modify query logic** (advanced):
```python
# Short question = more chunks allowed
# Long question = fewer chunks allowed

question_tokens = len(question) // 4
remaining_budget = (model_context_limit - question_tokens - 500)
max_chunks = remaining_budget // avg_chunk_tokens
```

---

### Option C: Use Prompt Compression

**Add compression step** (advanced):
```python
# Summarize chunks before sending to LLM
# Or use a reranker to pick only most relevant
```

---

## 🚀 Quick Reference

| Issue | Solution | When to Use |
|-------|----------|-------------|
| Error on any query | Reduce top_k to 1-2 | Immediate fix |
| Want more context | Use llama3.2:8k model | Best balance |
| Still hitting limit | Lower MAX_CONTEXT_CHARS | Small models |
| Need lots of context | Use GPT-4/Claude API | Production |

---

## ✅ Verification

After applying fix:

**1. Query should work:**
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this code do?"}'
```

**2. Check response includes:**
```json
{
  "answer": "...",
  "sources": [...]
}
```

**3. No error about context length**

---

## 🎓 Understanding Token Limits

**Quick Math:**

```
System Prompt:      ~200 tokens
Retrieved Context:  ~800 tokens (3 chunks × 150 lines × ~2 tokens/line)
User Question:      ~50 tokens
Response Buffer:    ~500 tokens
-----------------------------------
Total Needed:       ~1550 tokens ✅ Fits in 2048

But if top_k=5:
Retrieved Context:  ~1300 tokens
Total Needed:       ~2050 tokens ❌ Exceeds 2048!
```

**That's why we reduced defaults to 3 chunks!**

---

## 📥 Get the Fixed Version

The latest `rag_assistant_langchain.tar.gz` includes:
- ✅ Reduced defaults (top_k=3, max_context=4000)
- ✅ Smart chunk truncation
- ✅ Better error messages
- ✅ Context overflow prevention

---

## Still Having Issues?

**Enable debug logging:**

```python
# In backend/main.py, add at top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check context size being sent:**

```python
# In rag_service.py _generate_answer, add:
print(f"Context length: {len(context)} chars")
print(f"Question length: {len(question)} chars")
print(f"Total: {len(context) + len(question) + 500} chars")
```

This will show you exactly how much is being sent to the model.

---

## Summary

**The fix is simple:**
1. ✅ Use latest version (safer defaults)
2. ✅ Reduce top_k if needed (1-2 for small models)
3. ✅ Or use a bigger model (llama3.2:8k)
4. ✅ Adjust MAX_CONTEXT_CHARS in .env

**Most users won't see this error with the new defaults!** 🎯
