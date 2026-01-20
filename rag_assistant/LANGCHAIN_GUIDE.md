# LangChain Integration Guide

This document explains how LangChain Community is integrated into the RAG Codebase Assistant and provides guidance for customization.

## Overview

The system uses **LangChain Community** to provide a robust, maintainable abstraction layer over:
- Ollama embeddings API
- Ollama chat API  
- Chroma vector database

## Components

### 1. OllamaEmbeddings

**Location**: `backend/rag_service.py`, `scripts/index_codebase.py`

**Purpose**: Generate vector embeddings for text

**Configuration**:
```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model=config.OLLAMA_EMBED_MODEL,  # e.g., "nomic-embed-text"
    base_url=config.OLLAMA_BASE_URL    # e.g., "http://localhost:11434"
)
```

**Methods Used**:
- `embed_query(text: str) -> List[float]`: Embed a single query
- `embed_documents(texts: List[str]) -> List[List[float]]`: Batch embed multiple documents

**Advantages over raw API**:
- Automatic error handling and retries
- Consistent interface with other embedding providers
- Built-in caching support
- Type safety with return values

### 2. ChatOllama

**Location**: `backend/rag_service.py`

**Purpose**: Generate chat completions with context

**Configuration**:
```python
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

chat_model = ChatOllama(
    model=config.OLLAMA_CHAT_MODEL,    # e.g., "llama3.2"
    base_url=config.OLLAMA_BASE_URL,
    temperature=0                       # Deterministic for code Q&A
)
```

**Usage**:
```python
messages = [
    SystemMessage(content="You are a helpful assistant..."),
    HumanMessage(content="User question here")
]

response = chat_model.invoke(messages)
answer = response.content
```

**Advantages over raw API**:
- Structured message format
- Streaming support (if needed)
- Token counting utilities
- Consistent interface across LLM providers

### 3. Chroma Vectorstore

**Location**: `backend/rag_service.py`, `scripts/index_codebase.py`

**Purpose**: Vector storage and similarity search

**Initialization**:
```python
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# For indexing (create new)
vectorstore = Chroma.from_documents(
    documents=documents,                      # List[Document]
    embedding=embeddings,                     # OllamaEmbeddings instance
    collection_name=config.CHROMA_COLLECTION,
    persist_directory=config.CHROMA_PERSIST_DIR,
    ids=[doc.metadata["chunk_id"] for doc in documents]
)

# For querying (load existing)
vectorstore = Chroma(
    collection_name=config.CHROMA_COLLECTION,
    embedding_function=embeddings,
    persist_directory=config.CHROMA_PERSIST_DIR
)
```

**Methods Used**:
- `similarity_search_with_score(query, k)`: Returns documents with distance scores
- `add_documents(documents, ids)`: Add new documents to existing collection

**Advantages over raw API**:
- Automatic embedding generation for queries
- Document abstraction with metadata
- Consistent search interface
- Easy persistence management

## Document Format

LangChain uses a unified `Document` class:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="The actual text content...",
    metadata={
        "file_path": "backend/main.py",
        "start_line": 10,
        "end_line": 50,
        "chunk_id": "unique_identifier"
    }
)
```

**Benefits**:
- Standardized format across all vectorstores
- Type-safe metadata handling
- Easy serialization/deserialization

## How It Works

### Indexing Flow

1. **Read files** → Parse Python files into chunks
2. **Create Documents** → Convert chunks to LangChain Documents
3. **Generate embeddings** → OllamaEmbeddings creates vectors automatically
4. **Store in Chroma** → Chroma.from_documents() handles persistence

```python
# Simplified indexing code
documents = [
    Document(
        page_content=chunk["text"],
        metadata={"file_path": chunk["path"], ...}
    )
    for chunk in all_chunks
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,  # LangChain handles embedding generation
    collection_name="codebase",
    persist_directory="chroma_db"
)
```

### Query Flow

1. **User question** → Received by FastAPI endpoint
2. **Embed query** → OllamaEmbeddings converts to vector (automatic)
3. **Similarity search** → Chroma finds relevant chunks
4. **Build context** → Format retrieved chunks
5. **Generate answer** → ChatOllama creates response
6. **Return results** → Send answer + sources to user

```python
# Simplified query code
chunks = vectorstore.similarity_search_with_score(
    query=question,
    k=top_k
)  # LangChain handles query embedding internally

context = build_context(chunks)

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=f"{context}\n\nQuestion: {question}")
]

response = chat_model.invoke(messages)
answer = response.content
```

## Switching Providers

### To OpenAI

```python
# Install: pip install langchain-openai

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Replace embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="your-key"
)

# Replace chat model
chat_model = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0,
    openai_api_key="your-key"
)

# Everything else stays the same!
```

### To Anthropic Claude

```python
# Install: pip install langchain-anthropic

from langchain_anthropic import AnthropicEmbeddings, ChatAnthropic

# Anthropic doesn't provide embeddings, so keep Ollama or use OpenAI
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Replace chat model
chat_model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    anthropic_api_key="your-key"
)
```

## Customization Examples

### 1. Add Streaming Support

```python
# In rag_service.py
def _generate_answer_streaming(self, question: str, context: str):
    """Generate answer with streaming."""
    messages = [...]
    
    for chunk in self.chat_model.stream(messages):
        yield chunk.content
```

### 2. Add Caching

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Enable caching for repeated queries
set_llm_cache(InMemoryCache())
```

### 3. Add Response Validation

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class CodeAnswer(BaseModel):
    answer: str
    confidence: float
    citations: List[str]

parser = PydanticOutputParser(pydantic_object=CodeAnswer)

# Add format instructions to prompt
format_instructions = parser.get_format_instructions()
```

### 4. Hybrid Search (Vector + Keyword)

```python
# Add BM25 retriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(documents)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 30% keyword, 70% semantic
)
```

## Best Practices

### 1. Error Handling

```python
try:
    result = vectorstore.similarity_search(query, k=5)
except Exception as e:
    logger.error(f"Retrieval failed: {e}")
    # Fallback logic
```

### 2. Batching

```python
# Index in batches for better performance
BATCH_SIZE = 100
for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    vectorstore.add_documents(batch)
```

### 3. Metadata Filtering

```python
# Search with metadata filters
results = vectorstore.similarity_search(
    query="authentication",
    k=5,
    filter={"file_path": "backend/user_db.py"}
)
```

### 4. Connection Pooling

```python
# Reuse embeddings/chat instances
# Don't recreate on every request
class RAGService:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(...)  # Reused
        self.chat_model = ChatOllama(...)        # Reused
```

## Troubleshooting

### Issue: "Connection refused" to Ollama

**Solution**: Ensure Ollama is running
```bash
ollama serve
```

### Issue: "Model not found"

**Solution**: Pull the required model
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Issue: Slow embedding generation

**Solution**: Use batching and consider GPU acceleration
```python
# Enable batching in OllamaEmbeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
    # Ollama handles batching internally
)
```

### Issue: Out of memory during indexing

**Solution**: Reduce batch size
```python
BATCH_SIZE = 25  # Reduce from 50
```

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain Community Integrations](https://python.langchain.com/docs/integrations/providers/)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Chroma Documentation](https://docs.trychroma.com/)

## Summary

LangChain Community provides:
- ✅ Clean abstractions over raw APIs
- ✅ Consistent interfaces across providers
- ✅ Built-in error handling and retries
- ✅ Easy provider switching
- ✅ Rich ecosystem of tools and extensions

This makes the codebase more maintainable and extensible while following best practices for production RAG systems.
