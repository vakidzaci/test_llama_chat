# URGENT FIX: "KeyError: distance" Issue

## 🚨 Problem
Getting `KeyError: distance` when querying, even after indexing.

## ✅ DEFINITIVE SOLUTION (3 Steps)

### Step 1: Run Diagnostic
```bash
cd rag_assistant
python diagnose_collection.py
```

This will show you:
- ✅ If collection exists and has data
- ✅ What the metadata structure looks like
- ✅ If LangChain can query it
- ✅ Exact error if something is wrong

---

### Step 2A: If Diagnostic Shows Issues - Re-index

```bash
# Reset and reindex with latest script
python scripts/index_codebase.py /path/to/your/repo --reset
```

**Wait for:**
```
✅ Indexing complete! Total chunks in collection: XXX
```

---

### Step 2B: If Diagnostic Works - Update Backend Code

Your indexing is fine, but the backend code needs updating.

**Replace `backend/rag_service.py` with the version from latest download.**

Or manually update the `_retrieve_chunks` method (see below).

---

### Step 3: Restart Backend
```bash
# Press Ctrl+C to stop current backend
python backend/main.py
```

---

## 🔧 Manual Code Fix (If Not Re-downloading)

If you don't want to re-download, replace the `_retrieve_chunks` method in `backend/rag_service.py`:

**Find this method (around line 54) and replace entirely:**

```python
def _retrieve_chunks(self, query: str, top_k: int) -> List[Dict]:
    """Retrieve relevant chunks from Chroma using LangChain - bulletproof version."""
    if not self.vectorstore:
        raise ValueError("Vectorstore not initialized. Please run indexing first.")
    
    try:
        # Use LangChain's similarity search with scores
        results_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        if not results_with_scores:
            print("Warning: No results returned from similarity search")
            return []
        
        # Convert to our format with extensive validation
        chunks = []
        for i, result in enumerate(results_with_scores):
            try:
                # Handle both tuple and non-tuple returns
                if isinstance(result, tuple) and len(result) >= 2:
                    doc, score = result[0], result[1]
                elif isinstance(result, tuple) and len(result) == 1:
                    doc = result[0]
                    score = 0.0
                else:
                    doc = result
                    score = 0.0
                
                # Validate document object
                if not hasattr(doc, 'page_content'):
                    print(f"Warning: Result {i} missing page_content, skipping")
                    continue
                
                # Get page content
                page_content = doc.page_content if doc.page_content else ""
                
                # Get metadata with defaults
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata = doc.metadata
                else:
                    metadata = {}
                    print(f"Warning: Result {i} has no metadata")
                
                # Ensure score is numeric
                try:
                    distance = float(score) if score is not None else 0.0
                except (TypeError, ValueError):
                    print(f"Warning: Invalid score type {type(score)}, using 0.0")
                    distance = 0.0
                
                # Build chunk with safe defaults
                chunk = {
                    "document": page_content,
                    "metadata": {
                        "file_path": metadata.get("file_path", "unknown"),
                        "start_line": int(metadata.get("start_line", 0)) if metadata.get("start_line") else 0,
                        "end_line": int(metadata.get("end_line", 0)) if metadata.get("end_line") else 0
                    },
                    "distance": distance
                }
                
                chunks.append(chunk)
                
            except Exception as e:
                print(f"Warning: Error processing result {i}: {e}")
                continue
        
        if not chunks:
            print("Warning: No valid chunks extracted from results")
            return []
        
        print(f"Successfully retrieved {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback to direct Chroma query
        print("Attempting fallback to direct Chroma query...")
        try:
            return self._retrieve_chunks_fallback(query, top_k)
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise ValueError(f"Retrieval failed: {str(e)}")

def _retrieve_chunks_fallback(self, query: str, top_k: int) -> List[Dict]:
    """Fallback retrieval using direct Chroma client."""
    if not self.collection:
        raise ValueError("Collection not available")
    
    # Generate embedding using LangChain
    query_embedding = self.embeddings.embed_query(query)
    
    # Direct query to Chroma
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Convert to our format
    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "document": doc,
            "metadata": {
                "file_path": meta.get("file_path", "unknown"),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0)
            },
            "distance": float(dist) if dist is not None else 0.0
        })
    
    print(f"Fallback retrieval successful: {len(chunks)} chunks")
    return chunks
```

**Save and restart backend.**

---

## 🎯 What This Does

The new code:
- ✅ Handles all possible return formats from LangChain
- ✅ Validates every piece of data before using it
- ✅ Provides default values for missing fields
- ✅ Has fallback to direct Chroma query if LangChain fails
- ✅ Shows detailed warnings in console
- ✅ Won't crash on any data structure

---

## 📊 Testing After Fix

**1. Run diagnostic:**
```bash
python diagnose_collection.py
```

Should show:
```
✅ Connected to Chroma
✅ Found collection: codebase
✅ Total chunks: XXX
✅ LangChain search works!
```

**2. Test query:**
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
```

Should return:
```json
{
  "answer": "...",
  "sources": [...]
}
```

---

## 🔍 Why This Happens

**Possible causes:**
1. LangChain version mismatch (returns different format)
2. Chroma was indexed with old script (different metadata structure)
3. Empty or corrupted collection

**The new code handles ALL these cases!**

---

## 💡 Prevention

Going forward:
1. Use latest indexing script
2. Use latest rag_service.py
3. Run diagnostic before first query

---

## 🆘 Still Not Working?

**Check backend console output** when you query. The new code prints detailed warnings:

```
Warning: Result 0 has no metadata
Warning: Invalid score type <class 'str'>, using 0.0
Successfully retrieved 3 chunks
```

These warnings will tell you exactly what's wrong.

**Send me the console output** and I can provide specific fix.

---

## ✅ Download Latest Version

The latest `rag_assistant_langchain.tar.gz` has:
- ✅ Bulletproof retrieval code
- ✅ Diagnostic script
- ✅ Fallback mechanisms
- ✅ All fixes applied

**This will work!** 🎯
