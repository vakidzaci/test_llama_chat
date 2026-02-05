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
import io
import re
import json
import hashlib
import requests
from datetime import datetime
from flask import Flask, request, jsonify

import fitz  # PyMuPDF
import pdfplumber

app = Flask(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Chunking / context protection
MAX_CHARS_FOR_MODEL = 90_000     # per chunk prompt payload
PAGES_PER_CHUNK = 6              # tune: 4-10 pages typical
OLLAMA_TIMEOUT_SEC = 240

# -----------------------------
# Target schema (shape contract)
# -----------------------------
TARGET_SCHEMA = {
    "document_type": "loan_contract | technical_report | id_card | unknown",
    "language": "kk | ru | en | unknown",
    "entities": {
        "client_name": None,
        "co_borrower_name": None,
        "manager_name": None,
        "organization_name": None
    },
    "identifiers": {
        "iin": [],
        "document_number": [],
        "contract_number": [],
        "account_numbers": [],
        "iban": [],
        "phone_numbers": [],
        "emails": []
    },
    "loan_terms": {
        "total_loan_amount": None,
        "currency": None,
        "interest_rate": None,
        "effective_rate": None,
        "term": None,
        "payment_schedule": None,
        "collateral": None
    },
    "dates": {
        "issue_date": None,
        "start_date": None,
        "end_date": None,
        "signing_date": None,
        "other_dates": []
    },
    "banking": {
        "bank_name": None,
        "bic": [],
        "kbe": [],
        "knp": [],
        "requisites_raw": []
    },
    "addresses": {
        "client_address": None,
        "organization_address": None
    },
    "extracted_facts": [],
    "confidence_notes": [],
    "raw_evidence": {
        "snippets": [],
        "tables": []
    }
}


# -----------------------------
# Utilities
# -----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def safe_json_parse(text: str) -> dict:
    """Strict JSON parse or salvage first {...} block."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)
    raise ValueError("Model did not return valid JSON")


def normalize_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


# -----------------------------
# Scan detection (per-page + overall)
# -----------------------------
def page_scan_likelihood(page: fitz.Page, min_text_chars=15) -> dict:
    text = (page.get_text("text") or "").strip()
    text_chars = len(text)

    rect = page.rect
    page_area = float(rect.width * rect.height) or 1.0

    img_list = page.get_images(full=True)
    img_area = 0.0
    for img in img_list:
        xref = img[0]
        rects = page.get_image_rects(xref)
        for r in rects:
            img_area += float(r.width * r.height)

    img_ratio = min(img_area / page_area, 1.0)
    likely_scanned = (text_chars < min_text_chars) and (img_ratio > 0.30)

    return {
        "text_chars": text_chars,
        "image_area_ratio": round(img_ratio, 3),
        "likely_scanned": likely_scanned
    }


def scan_profile(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    per_page = []
    total_text = 0
    scanned_pages = 0

    for i in range(doc.page_count):
        page = doc.load_page(i)
        info = page_scan_likelihood(page)
        total_text += info["text_chars"]
        if info["likely_scanned"]:
            scanned_pages += 1
        per_page.append({"page": i + 1, **info})

    page_count = doc.page_count
    scanned_ratio = scanned_pages / max(page_count, 1)

    # Decide: reject if mostly scanned OR almost no text overall
    overall_scanned = (scanned_ratio >= 0.6) or (total_text < 50)

    return {
        "page_count": page_count,
        "total_text_chars": total_text,
        "scanned_pages": scanned_pages,
        "scanned_ratio": round(scanned_ratio, 3),
        "overall_scanned": overall_scanned,
        "per_page": per_page
    }


# -----------------------------
# Text + table extraction
# -----------------------------
def extract_pages_text(doc: fitz.Document) -> list:
    pages_text = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        t = (page.get_text("text") or "").strip()
        pages_text.append({"page": i + 1, "text": t})
    return pages_text


def extract_tables(pdf_bytes: bytes) -> list:
    tables = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                page_tables = page.extract_tables() or []
            except Exception:
                page_tables = []
            for tbl in page_tables:
                if tbl and any(any((cell or "").strip() for cell in row) for row in tbl):
                    tables.append({"page": i + 1, "table": tbl})
    return tables


def remove_repeated_headers_footers(pages_text: list, min_repeat_ratio=0.6) -> list:
    """
    Light cleanup:
      - For each page, look at first 2 lines and last 2 lines.
      - If a line repeats on many pages, remove it.
    """
    first_lines = []
    last_lines = []
    for p in pages_text:
        lines = [ln.strip() for ln in (p["text"] or "").splitlines() if ln.strip()]
        first_lines.append(lines[:2])
        last_lines.append(lines[-2:] if len(lines) >= 2 else lines)

    # Count line frequency
    freq = {}
    all_lines = []
    for group in first_lines + last_lines:
        for ln in group:
            freq[ln] = freq.get(ln, 0) + 1

    threshold = max(2, int(len(pages_text) * min_repeat_ratio))
    repeated = {ln for ln, c in freq.items() if c >= threshold and len(ln) >= 5}

    cleaned = []
    for p in pages_text:
        lines = [ln for ln in (p["text"] or "").splitlines()]
        out = []
        for ln in lines:
            lns = ln.strip()
            if lns and lns in repeated:
                continue
            out.append(ln)
        cleaned.append({"page": p["page"], "text": "\n".join(out).strip()})
    return cleaned


# -----------------------------
# Regex candidate pre-extraction
# -----------------------------
IIN_RE = re.compile(r"\b\d{12}\b")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# KZ phones often: +7 7xx xxx xx xx, 8 7xx xxx xx xx, etc.
PHONE_RE = re.compile(
    r"(?:(?:\+7|8)\s*)?(?:\(?7\d{2}\)?\s*[\d\s-]{5,}\d)"
)

# Simple IBAN-like (KZ** + digits/letters)
IBAN_RE = re.compile(r"\bKZ[0-9A-Z]{18,32}\b", re.IGNORECASE)

# Interest rate patterns: 12%, 12.5 %, 0.175
RATE_RE = re.compile(r"\b\d{1,2}(?:[.,]\d{1,2})?\s*%|\b0\.\d{2,4}\b")

# Money amounts: 1 200 000,00; 1200000; 1,200,000; with KZT/USD/EUR/₸/$/€
AMOUNT_RE = re.compile(
    r"(?:(?:₸|\$|€)\s*)?\b\d{1,3}(?:[ \u00A0.,]\d{3})*(?:[.,]\d{2})?\b(?:\s*(?:KZT|USD|EUR|₸|\$|€))?",
    re.IGNORECASE
)

# Dates: 01.02.2026, 2026-02-01, 1/2/2026
DATE_RE = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
)

# Contract/document number-ish:
# "№ 123/45", "No. 123", "Договор 123", "Contract 123-AB"
DOCNO_RE = re.compile(
    r"(?:№|No\.?|Nº|Договор|Контракт|Contract|Agreement|Doc(?:ument)?)[^\n]{0,30}\b([A-Z0-9][A-Z0-9/-]{2,})\b",
    re.IGNORECASE
)

# BIC in KZ often 8 or 11 chars (SWIFT/BIC). We keep conservative:
BIC_RE = re.compile(r"\b[A-Z]{4}KZ[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b", re.IGNORECASE)


def extract_candidates_from_text(all_text: str) -> dict:
    iins = sorted(set(IIN_RE.findall(all_text)))
    emails = sorted(set(EMAIL_RE.findall(all_text)))

    raw_phones = PHONE_RE.findall(all_text)
    phones = []
    for ph in raw_phones:
        d = normalize_digits(ph)
        # KZ mobile numbers often 11 digits with country, 10 digits local; keep reasonable
        if 10 <= len(d) <= 12:
            phones.append(ph.strip())
    phones = sorted(set(phones))

    ibans = sorted(set(m.upper() for m in IBAN_RE.findall(all_text)))
    bics = sorted(set(m.upper() for m in BIC_RE.findall(all_text)))

    docnos = sorted(set(m.group(1) for m in DOCNO_RE.finditer(all_text)))

    rates = sorted(set(m.group(0).replace(" ", "") for m in RATE_RE.finditer(all_text)))

    dates = sorted(set(m.group(0) for m in DATE_RE.finditer(all_text)))

    # amounts are noisy; keep distinct but capped
    amounts = []
    for m in AMOUNT_RE.finditer(all_text):
        s = (m.group(0) or "").strip()
        # Filter out too-short numbers (like "2026" from a date)
        if len(normalize_digits(s)) >= 5:
            amounts.append(s)
    amounts = list(dict.fromkeys(amounts))[:50]

    return {
        "iin_candidates": iins,
        "email_candidates": emails,
        "phone_candidates": phones,
        "iban_candidates": ibans,
        "bic_candidates": bics,
        "doc_number_candidates": docnos,
        "rate_candidates": rates,
        "date_candidates": dates,
        "amount_candidates": amounts
    }


# -----------------------------
# Ollama prompting: chunk extraction + final merge
# -----------------------------
def build_chunk_prompt(chunk_payload: dict) -> str:
    schema_hint = {
        "document_type": "loan_contract | technical_report | id_card | unknown",
        "language": "kk | ru | en | unknown",
        "entities": {
            "client_name": "string|null",
            "co_borrower_name": "string|null",
            "manager_name": "string|null",
            "organization_name": "string|null"
        },
        "identifiers": {
            "iin": "array[string]",
            "document_number": "array[string]",
            "contract_number": "array[string]",
            "account_numbers": "array[string]",
            "iban": "array[string]",
            "phone_numbers": "array[string]",
            "emails": "array[string]"
        },
        "loan_terms": {
            "total_loan_amount": "string|null",
            "currency": "string|null",
            "interest_rate": "string|null",
            "effective_rate": "string|null",
            "term": "string|null",
            "payment_schedule": "string|null",
            "collateral": "string|null"
        },
        "dates": {
            "issue_date": "string|null",
            "start_date": "string|null",
            "end_date": "string|null",
            "signing_date": "string|null",
            "other_dates": "array[string]"
        },
        "banking": {
            "bank_name": "string|null",
            "bic": "array[string]",
            "kbe": "array[string]",
            "knp": "array[string]",
            "requisites_raw": "array[string]"
        },
        "addresses": {
            "client_address": "string|null",
            "organization_address": "string|null"
        },
        "extracted_facts": "array[object]",
        "confidence_notes": "array[string]",
        "raw_evidence": {
            "snippets": "array[object]",
            "tables": "array[object]"
        }
    }

    content = json.dumps(chunk_payload, ensure_ascii=False)
    if len(content) > MAX_CHARS_FOR_MODEL:
        content = content[:MAX_CHARS_FOR_MODEL] + "\n...[TRUNCATED]..."

    return f"""
You are an information extraction system. Return ONLY valid JSON (no markdown, no commentary).
Rules:
- Use provided candidates as hints, but confirm from evidence.
- Do NOT invent values. If missing, use null or empty arrays.
- Kazakhstan IIN is 12 digits (string).
- Add evidence for each extracted item in raw_evidence.snippets (page + short quote + field).
- If you see a value in a table, include the relevant table cells/row in raw_evidence.tables.

Output JSON must match this schema shape:
{json.dumps(schema_hint, ensure_ascii=False)}

Here is a chunk of the PDF (pages_text, tables, and candidate hints):
{content}
""".strip()


def build_merge_prompt(partials: list, global_candidates: dict) -> str:
    """
    Merge multiple chunk JSONs into one final JSON matching TARGET_SCHEMA shape.
    Rules: prefer non-null, union arrays, de-duplicate, keep best evidence.
    """
    payload = {
        "partials": partials,
        "global_candidates": global_candidates,
        "target_schema_shape": TARGET_SCHEMA
    }
    content = json.dumps(payload, ensure_ascii=False)
    if len(content) > MAX_CHARS_FOR_MODEL:
        content = content[:MAX_CHARS_FOR_MODEL] + "\n...[TRUNCATED]..."

    return f"""
You will MERGE multiple partial JSON extraction results into ONE final JSON.
Return ONLY valid JSON (no markdown, no commentary).

Merge rules:
- Arrays: union + de-duplicate.
- Scalars: choose the most specific non-null value supported by evidence.
- If conflicting values exist, keep the one with stronger evidence and add a note in confidence_notes.
- Keep raw_evidence: include the best snippets/tables; remove duplicates.
- Do NOT invent values; if unsure, use null/empty.

Final output MUST match the target schema shape exactly (keys present as in target).
Here is the merge input:
{content}
""".strip()


def call_ollama(prompt: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192
        }
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT_SEC)
    r.raise_for_status()
    resp = r.json()
    return safe_json_parse(resp.get("response", ""))


def repair_json_with_ollama(bad_text: str) -> dict:
    """
    If model output doesn't parse or doesn't match schema, ask for repair.
    """
    prompt = f"""
Return ONLY valid JSON, no markdown.
Fix the following to be valid JSON and match this exact schema shape (do not add new top-level keys):
{json.dumps(TARGET_SCHEMA, ensure_ascii=False)}

Constraints:
- Do NOT change values except to fix types/structure.
- Missing fields must be added with null/[] as appropriate.

Bad output:
{bad_text}
""".strip()
    return call_ollama(prompt)


# -----------------------------
# Output validation / normalization
# -----------------------------
def ensure_schema_shape(obj: dict) -> dict:
    """
    Ensures all keys exist per TARGET_SCHEMA.
    Does not deeply validate types perfectly, but enforces presence and basic array/scalar expectations.
    """
    def fill(template, value):
        if isinstance(template, dict):
            out = {}
            value = value if isinstance(value, dict) else {}
            for k, v in template.items():
                out[k] = fill(v, value.get(k))
            return out
        elif isinstance(template, list):
            return value if isinstance(value, list) else []
        else:
            # template is scalar placeholder
            # allow None or scalar
            return value if value is not None else None

    shaped = fill(TARGET_SCHEMA, obj if isinstance(obj, dict) else {})

    # Ensure key arrays are arrays
    for k in ["iin", "document_number", "contract_number", "account_numbers", "iban", "phone_numbers", "emails"]:
        shaped["identifiers"][k] = shaped["identifiers"][k] or []
        if not isinstance(shaped["identifiers"][k], list):
            shaped["identifiers"][k] = [str(shaped["identifiers"][k])]

    # Normalize IIN strings to 12-digit only
    cleaned_iins = []
    for x in shaped["identifiers"]["iin"]:
        d = normalize_digits(str(x))
        if len(d) == 12:
            cleaned_iins.append(d)
    shaped["identifiers"]["iin"] = sorted(set(cleaned_iins))

    # Normalize BIC, IBAN
    shaped["banking"]["bic"] = [str(x).upper() for x in (shaped["banking"]["bic"] or []) if str(x).strip()]
    shaped["identifiers"]["iban"] = [str(x).upper() for x in (shaped["identifiers"]["iban"] or []) if str(x).strip()]

    return shaped


# -----------------------------
# Chunk builder
# -----------------------------
def chunk_pages(pages_text: list, tables: list, candidates: dict) -> list:
    """
    Build chunk payloads:
      - pages i..i+PAGES_PER_CHUNK
      - tables that belong to those pages
      - include global candidates + chunk-local candidates
    """
    chunks = []
    total_pages = len(pages_text)

    # Index tables by page
    tables_by_page = {}
    for t in tables:
        tables_by_page.setdefault(t["page"], []).append(t["table"])

    for start in range(0, total_pages, PAGES_PER_CHUNK):
        end = min(total_pages, start + PAGES_PER_CHUNK)
        page_slice = pages_text[start:end]

        # collect tables within this range
        tbls = []
        for p in range(start + 1, end + 1):
            for table in tables_by_page.get(p, []):
                tbls.append({"page": p, "table": table})

        # chunk-local candidate extraction (helps when globals are huge/noisy)
        chunk_text = "\n".join(p["text"] for p in page_slice if p["text"])
        local_candidates = extract_candidates_from_text(chunk_text)

        chunk_payload = {
            "chunk_pages": [start + 1, end],
            "pages_text": page_slice,
            "tables": tbls,
            "candidate_hints_global": candidates,
            "candidate_hints_local": local_candidates
        }

        chunks.append(chunk_payload)

    return chunks


# -----------------------------
# Main pipeline
# -----------------------------
def extract_pipeline(pdf_bytes: bytes) -> dict:
    profile = scan_profile(pdf_bytes)
    if profile["overall_scanned"]:
        return {
            "ok": False,
            "error": "scanned_pdf",
            "message": "PDF appears to be a scan (no reliable text layer). OCR pipeline required.",
            "scan_profile": profile
        }

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = extract_pages_text(doc)
    pages_text = remove_repeated_headers_footers(pages_text)
    tables = extract_tables(pdf_bytes)

    all_text = "\n".join(p["text"] for p in pages_text if p["text"])
    candidates = extract_candidates_from_text(all_text)

    chunks = chunk_pages(pages_text, tables, candidates)

    partials = []
    for ch in chunks:
        prompt = build_chunk_prompt(ch)
        partial = call_ollama(prompt)
        partials.append(ensure_schema_shape(partial))

    merge_prompt = build_merge_prompt(partials, candidates)
    merged = call_ollama(merge_prompt)
    merged = ensure_schema_shape(merged)

    # Attach scan profile (optional) to confidence notes rather than changing schema
    if profile["scanned_pages"] > 0:
        merged["confidence_notes"].append(
            f"Mixed PDF: {profile['scanned_pages']}/{profile['page_count']} pages look scanned; fields may be missing."
        )

    return {"ok": True, "data": merged, "meta": {"scan_profile": profile, "candidates": candidates}}


# -----------------------------
# Flask route
# -----------------------------
@app.route("/extract", methods=["POST"])
def extract():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "missing_file"}), 400

    f = request.files["file"]
    pdf_bytes = f.read()
    if not pdf_bytes:
        return jsonify({"ok": False, "error": "empty_file"}), 400

    try:
        result = extract_pipeline(pdf_bytes)
    except requests.HTTPError as e:
        return jsonify({"ok": False, "error": "ollama_http", "message": str(e)}), 502
    except ValueError as e:
        # JSON parse failure; try repair if we can
        return jsonify({"ok": False, "error": "json_parse_failed", "message": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": "internal_error", "message": str(e)}), 500

    status = 200 if result.get("ok") else 422
    return jsonify(result), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
