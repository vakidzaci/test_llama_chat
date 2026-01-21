# HOTFIX: Fix for "Field required" error in /rag/query endpoint

## Problem
Getting error: `Field required, loc: ['header', 'user_info']`

## Cause
Duplicate endpoint definition in `backend/main.py`

## Fix

Open `backend/main.py` and find this section (around line 104-120):

**REMOVE THIS:**
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
    pass


# Manual auth version of query endpoint
@app.post("/rag/query", response_model=QueryResponse, include_in_schema=True)
async def query_rag_manual(
```

**REPLACE WITH:**
```python
@app.post("/rag/query", response_model=QueryResponse)
async def query_rag(
```

Basically, delete the first broken endpoint and the comment, keeping only the function name `query_rag` instead of `query_rag_manual`.

## After Fixing

1. Save the file
2. Restart the backend:
   ```bash
   # Stop the current backend (Ctrl+C)
   python backend/main.py
   ```

3. Try the curl command again:
   ```cmd
   curl -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" -d "{\"username\": \"testuser\", \"password\": \"testpass123\"}"
   ```

## Or Download Fixed Version

Download the updated `rag_assistant_langchain.tar.gz` which has this fix already applied.
