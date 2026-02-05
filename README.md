# RAG Codebase Assistant

A multi-user codebase assistant built with RAG (Retrieval-Augmented Generation) for querying Python repositories.

## 🏗️ Architecture

- **Backend**: FastAPI with authentication, rate limiting, and RAG endpoints
- **Frontend**: Streamlit UI for user interaction
- **Vector DB**: Chroma (single shared collection for all users)
- **LLM**: Ollama (for embeddings and chat)
- **User DB**: SQLite with hashed passwords and API keys

## ✨ Features

- ✅ Single shared Chroma collection for all users
- ✅ User authentication with API keys
- ✅ Per-user rate limiting
- ✅ Retrieval-only LLM access (no full repo context)
- ✅ Line-based chunking with overlap
- ✅ Source citation with file paths and line ranges
- ✅ Configurable via environment variables

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



import os
import json
import requests
from flask import Flask, request, jsonify
import pdfplumber
from typing import Dict, Any, Tuple

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ollama configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')


def is_scanned_pdf(pdf_path: str, text_threshold: int = 50) -> bool:
    """
    Determine if a PDF is scanned or digital.
    
    Args:
        pdf_path: Path to the PDF file
        text_threshold: Minimum characters needed to consider it digital
    
    Returns:
        True if scanned (needs OCR), False if digital
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_text = ""
            # Check first 3 pages or all pages if fewer
            pages_to_check = min(3, len(pdf.pages))
            
            for i in range(pages_to_check):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                total_text += text
            
            # If very little text extracted, it's likely a scan
            return len(total_text.strip()) < text_threshold
    except Exception as e:
        print(f"Error checking PDF type: {e}")
        return True  # Assume scanned if can't determine


def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text and tables from a digital PDF.
    
    Returns:
        Dictionary with 'text' and 'tables' keys
    """
    content = {
        'text': '',
        'tables': []
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            all_tables = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                page_text = page.extract_text() or ""
                if page_text:
                    all_text.append(f"--- Page {page_num} ---\n{page_text}")
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        all_tables.append({
                            'page': page_num,
                            'table_index': table_idx + 1,
                            'data': table
                        })
            
            content['text'] = "\n\n".join(all_text)
            content['tables'] = all_tables
            
    except Exception as e:
        raise Exception(f"Failed to extract PDF content: {str(e)}")
    
    return content


def format_tables_for_llm(tables: list) -> str:
    """Format tables into a readable string for the LLM."""
    if not tables:
        return ""
    
    formatted = "\n\n=== TABLES FOUND ===\n"
    for table_info in tables:
        formatted += f"\nTable on Page {table_info['page']} (Table #{table_info['table_index']}):\n"
        for row in table_info['data']:
            formatted += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
    
    return formatted


def extract_fields_with_ollama(text_content: str, tables_content: str) -> Dict[str, Any]:
    """
    Send extracted content to Ollama and get structured JSON back.
    
    Returns:
        Structured dictionary with document information
    """
    prompt = f"""You are a document analysis expert. Extract ALL relevant information from the following document text and tables.

Document Text:
{text_content}

{tables_content}

Your task is to extract and return ONLY a valid JSON object (no markdown, no explanation) with these fields:

{{
    "document_type": "loan_contract | report | id_document | other",
    "extracted_fields": {{
        "full_names": ["list of all person names found"],
        "iin": ["list of IIN/personal identification numbers"],
        "contract_number": "contract or document number if found",
        "phone_numbers": ["list of phone numbers"],
        "bank_details": {{
            "account_numbers": ["list of account numbers"],
            "bank_names": ["list of bank names"],
            "iban": ["list of IBAN if found"]
        }},
        "loan_details": {{
            "amount": "loan amount with currency",
            "interest_rate": "interest rate",
            "term": "loan term/period",
            "collateral": "collateral description if any"
        }},
        "dates": {{
            "contract_date": "contract signing date",
            "start_date": "start/issue date",
            "end_date": "end/expiry date",
            "other_dates": ["any other relevant dates"]
        }},
        "addresses": ["list of addresses found"],
        "other_important_info": {{}}
    }},
    "confidence": "high | medium | low"
}}

Return ONLY the JSON object, nothing else."""

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        llm_response = result.get('response', '{}')
        
        # Parse the JSON response
        try:
            structured_data = json.loads(llm_response)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group())
            else:
                raise ValueError("Failed to parse JSON from Ollama response")
        
        return structured_data
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ollama API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to process Ollama response: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL
    })


@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    """
    Main endpoint to process PDF documents.
    
    Expects:
        - PDF file in 'file' field of multipart/form-data
    
    Returns:
        - If scanned: error message about OCR needed
        - If digital: structured JSON with extracted fields
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({
            "error": "No file provided",
            "message": "Please upload a PDF file with the key 'file'"
        }), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            "error": "No file selected",
            "message": "Please select a PDF file to upload"
        }), 400
    
    # Check if file is PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({
            "error": "Invalid file type",
            "message": "Only PDF files are accepted"
        }), 400
    
    # Save temporary file
    temp_path = f"/tmp/{os.urandom(16).hex()}.pdf"
    
    try:
        file.save(temp_path)
        
        # Step 1: Check if PDF is scanned or digital
        if is_scanned_pdf(temp_path):
            return jsonify({
                "status": "error",
                "message": "Can't read - OCR pipeline needed",
                "details": "This appears to be a scanned PDF. Text extraction requires OCR processing.",
                "document_type": "scanned_pdf"
            }), 422  # Unprocessable Entity
        
        # Step 2: Extract text and tables from digital PDF
        try:
            pdf_content = extract_pdf_content(temp_path)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": "Failed to extract PDF content",
                "details": str(e)
            }), 500
        
        # Check if we got any content
        if not pdf_content['text'].strip():
            return jsonify({
                "status": "error",
                "message": "Can't read - OCR pipeline needed",
                "details": "No text could be extracted. This might be a scanned or image-based PDF.",
                "document_type": "scanned_pdf"
            }), 422
        
        # Step 3: Format tables for LLM
        tables_formatted = format_tables_for_llm(pdf_content['tables'])
        
        # Step 4: Send to Ollama for structured extraction
        try:
            structured_data = extract_fields_with_ollama(
                pdf_content['text'],
                tables_formatted
            )
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": "Failed to process document with AI",
                "details": str(e)
            }), 500
        
        # Step 5: Return successful response
        return jsonify({
            "status": "success",
            "message": "Document processed successfully",
            "data": structured_data,
            "metadata": {
                "pages_processed": len(pdf_content['text'].split('--- Page')),
                "tables_found": len(pdf_content['tables']),
                "text_length": len(pdf_content['text'])
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Unexpected error occurred",
            "details": str(e)
        }), 500
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        "error": "File too large",
        "message": "Maximum file size is 16MB"
    }), 413


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
