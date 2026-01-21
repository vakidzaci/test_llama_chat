# Correct /rag/query Endpoint Implementation

## ✅ This is what backend/main.py should have (lines 150-194):

```python
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
```

## ❌ If you see THIS - it's wrong (DELETE IT):

```python
@app.post("/rag/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    user_info: dict = Header(..., alias="user_info", include_in_schema=False)
):
    """
    Query the codebase using RAG.
    Requires X-API-Key header for authentication.
    """
    # This endpoint uses a dependency injection pattern
    # We'll handle auth manually here for clarity
    pass  # <-- EMPTY! THIS IS BAD!
```

## How to Verify Your Installation

1. **Check if you have the correct version:**
   ```bash
   cd rag_assistant/backend
   grep -A 5 "def query_rag" main.py
   ```

2. **Should show:**
   ```python
   async def query_rag(
       request: QueryRequest,
       x_api_key: str = Header(..., alias="X-API-Key")
   ):
   ```

3. **Check there's NO `pass` in the function:**
   ```bash
   grep -A 30 "def query_rag" main.py | grep "pass"
   ```
   - Should return **nothing** (empty)

4. **Check there's only ONE query_rag function:**
   ```bash
   grep "async def query" main.py
   ```
   - Should show only **one line**

## If You Have the Old Version

**Re-download** the latest `rag_assistant_langchain.tar.gz` - it has the correct implementation.

Or manually replace the entire query endpoint with the correct code shown above.

## After Fixing

1. **Restart the backend:**
   ```bash
   # Press Ctrl+C to stop
   python backend/main.py
   ```

2. **Test with curl:**
   ```bash
   # Register
   curl -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" -d "{\"username\": \"test\", \"password\": \"test123\"}"
   
   # Query (use the API key you received)
   curl -X POST http://localhost:8000/rag/query -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" -d "{\"question\": \"test\"}"
   ```

## The latest download has this fix already applied!
