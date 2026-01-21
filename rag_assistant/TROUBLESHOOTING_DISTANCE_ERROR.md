# Troubleshooting: "RAG Query failed distance" Error

## Error Message
```
Query failed: RAG Query failed distance
```

## Root Cause
This error occurs when the similarity search results don't have the expected structure, usually due to:
1. Empty or improperly indexed Chroma collection
2. Missing metadata in indexed chunks
3. Type mismatch in distance/score values

## ✅ Solution (Already Fixed in Latest Version)

The latest `rag_assistant_langchain.tar.gz` has improved error handling that prevents this issue.

### If You're Seeing This Error:

**Step 1: Check if Codebase is Indexed**
```bash
# Check health endpoint
curl http://localhost:8000/health
```

Look for `"chunk_count"` - should be > 0. If it's 0:
```json
{
  "chunk_count": 0,  // ❌ Problem!
  "chroma_connected": true
}
```

**Solution**: Run the indexing script
```bash
python scripts/index_codebase.py /path/to/your/python/repo
```

---

**Step 2: Verify Collection Exists**
```bash
# From project root
python -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_db')
try:
    collection = client.get_collection('codebase')
    print(f'✅ Collection exists with {collection.count()} chunks')
except:
    print('❌ Collection not found - run indexing!')
"
```

---

**Step 3: Re-index if Needed**
```bash
# Reset and reindex
python scripts/index_codebase.py /path/to/repo --reset
```

---

**Step 4: Update to Latest Code**

If you downloaded earlier, update `backend/rag_service.py` with the improved error handling:

**Find this in `_retrieve_chunks` method (around line 54):**

**Replace:**
```python
def _retrieve_chunks(self, query: str, top_k: int) -> List[Dict]:
    if not self.vectorstore:
        raise ValueError("Vectorstore not initialized. Please run indexing first.")
    
    results_with_scores = self.vectorstore.similarity_search_with_score(
        query=query,
        k=top_k
    )
    
    chunks = []
    for doc, score in results_with_scores:
        chunks.append({
            "document": doc.page_content,
            "metadata": doc.metadata,
            "distance": float(score)
        })
    
    return chunks
```

**With:**
```python
def _retrieve_chunks(self, query: str, top_k: int) -> List[Dict]:
    if not self.vectorstore:
        raise ValueError("Vectorstore not initialized. Please run indexing first.")
    
    try:
        results_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        chunks = []
        for doc, score in results_with_scores:
            # Ensure metadata has required fields
            metadata = doc.metadata if doc.metadata else {}
            
            chunks.append({
                "document": doc.page_content,
                "metadata": {
                    "file_path": metadata.get("file_path", "unknown"),
                    "start_line": metadata.get("start_line", 0),
                    "end_line": metadata.get("end_line", 0)
                },
                "distance": float(score) if score is not None else 0.0
            })
        
        return chunks
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Retrieval failed: {str(e)}")
```

Then restart the backend:
```bash
# Press Ctrl+C to stop
python backend/main.py
```

---

## Quick Verification Steps

**1. Test Health:**
```bash
curl http://localhost:8000/health
```

**2. Check Collection:**
```bash
python -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_db')
collection = client.get_collection('codebase')
print(f'Chunks: {collection.count()}')
if collection.count() > 0:
    sample = collection.peek(1)
    print('Sample metadata:', sample['metadatas'][0])
"
```

**3. Test Query:**
```bash
# After registering and getting API key
curl -X POST http://localhost:8000/rag/query \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
```

---

## Common Issues

### Issue: "Collection not initialized"
**Cause**: No indexing has been done  
**Fix**: Run `python scripts/index_codebase.py /path/to/repo`

### Issue: "chunk_count: 0"
**Cause**: Indexing failed or no Python files found  
**Fix**: 
- Check repo path has .py files
- Check indexing script output for errors
- Try with `--reset` flag

### Issue: Still getting distance error after reindexing
**Cause**: Old backend code without error handling  
**Fix**: 
- Download latest `rag_assistant_langchain.tar.gz`
- Or manually update `backend/rag_service.py` as shown above
- Restart backend

---

## Prevention

To avoid this error:

1. **Always index before querying:**
   ```bash
   python scripts/index_codebase.py /path/to/repo
   ```

2. **Verify indexing succeeded:**
   ```bash
   curl http://localhost:8000/health | grep chunk_count
   ```

3. **Use latest code** which has defensive error handling

---

## Still Having Issues?

**Enable Debug Logging:**

In `backend/main.py`, add at the top:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Restart backend and check the console output when querying. This will show detailed error messages.

**Check Ollama:**
```bash
# Make sure Ollama is running
curl http://localhost:11434/api/tags

# Should return list of models
```

**Check Chroma DB Directory:**
```bash
ls -la chroma_db/
# Should see files if indexed
```

---

## The Latest Download Has All Fixes! ✅

Download `rag_assistant_langchain.tar.gz` for the version with improved error handling.
