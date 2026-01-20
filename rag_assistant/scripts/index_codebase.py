#!/usr/bin/env python3
"""
Index a Python codebase into Chroma vector database using LangChain.

Usage:
    python scripts/index_codebase.py /path/to/repo
    python scripts/index_codebase.py /path/to/repo --reset
"""
import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Dict

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "dist", "build", ".eggs", "venv", "env"}


def get_python_files(repo_path: Path) -> List[Path]:
    """Recursively find all .py files, skipping common junk directories."""
    python_files = []
    
    for py_file in repo_path.rglob("*.py"):
        # Check if any parent directory is in SKIP_DIRS
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        python_files.append(py_file)
    
    return python_files


def chunk_file(file_path: Path, repo_root: Path) -> List[Dict]:
    """
    Chunk a file into overlapping segments.
    
    Returns list of dicts with:
        - chunk_text: the actual text
        - file_path: relative path from repo root
        - start_line: starting line number (1-indexed)
        - end_line: ending line number (1-indexed)
        - chunk_id: stable identifier
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    if not lines:
        return []
    
    chunks = []
    rel_path = str(file_path.relative_to(repo_root))
    
    chunk_size = config.CHUNK_SIZE_LINES
    overlap = config.OVERLAP_LINES
    
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        chunk_lines = lines[i:end]
        chunk_text = ''.join(chunk_lines)
        
        # Create stable chunk ID
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        chunk_id = f"{rel_path}::{i+1}::{end}::{content_hash}"
        
        chunks.append({
            "chunk_text": chunk_text,
            "file_path": rel_path,
            "start_line": i + 1,  # 1-indexed
            "end_line": end,
            "chunk_id": chunk_id
        })
        
        # Move forward by (chunk_size - overlap)
        i += max(1, chunk_size - overlap)
    
    return chunks


def index_repository(repo_path: str, reset: bool = False):
    """Main indexing function using LangChain."""
    repo_path = Path(repo_path).resolve()
    
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    if not repo_path.is_dir():
        print(f"Error: Path is not a directory: {repo_path}")
        sys.exit(1)
    
    print(f"Indexing repository: {repo_path}")
    print(f"Chroma persist dir: {config.CHROMA_PERSIST_DIR}")
    print(f"Collection name: {config.CHROMA_COLLECTION}")
    
    # Initialize LangChain embeddings
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )
    
    # Handle reset if requested
    if reset:
        print("Reset flag detected. Deleting existing collection...")
        try:
            client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            client.delete_collection(name=config.CHROMA_COLLECTION)
            print("Collection deleted.")
        except Exception as e:
            print(f"Note: Could not delete collection (might not exist): {e}")
    
    # Find all Python files
    python_files = get_python_files(repo_path)
    print(f"Found {len(python_files)} Python files.")
    
    if not python_files:
        print("Warning: No Python files found!")
        return
    
    # Process files and chunks
    all_chunks = []
    for py_file in python_files:
        chunks = chunk_file(py_file, repo_path)
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks.")
    
    if not all_chunks:
        print("Warning: No chunks created!")
        return
    
    # Convert chunks to LangChain Documents
    print("Converting chunks to LangChain Documents...")
    documents = []
    for chunk in all_chunks:
        doc = Document(
            page_content=chunk["chunk_text"],
            metadata={
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "chunk_id": chunk["chunk_id"]
            }
        )
        documents.append(doc)
    
    # Create or update Chroma vectorstore using LangChain
    print("Creating embeddings and indexing into Chroma (this may take a while)...")
    
    # Process in batches for progress feedback
    batch_size = 50
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        if i == 0:
            # Create vectorstore with first batch
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=config.CHROMA_COLLECTION,
                persist_directory=config.CHROMA_PERSIST_DIR,
                ids=[doc.metadata["chunk_id"] for doc in batch]
            )
        else:
            # Add subsequent batches
            vectorstore.add_documents(
                documents=batch,
                ids=[doc.metadata["chunk_id"] for doc in batch]
            )
    
    # Verify
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=config.CHROMA_COLLECTION)
    count = collection.count()
    print(f"\n✅ Indexing complete! Total chunks in collection: {count}")
    
    # Show sample
    sample = collection.peek(limit=3)
    if sample and sample.get('metadatas'):
        print("\nSample chunks:")
        for meta in sample['metadatas'][:3]:
            print(f"  - {meta['file_path']} (lines {meta['start_line']}-{meta['end_line']})")


def main():
    parser = argparse.ArgumentParser(description="Index a Python codebase into Chroma")
    parser.add_argument("repo_path", help="Path to the Python repository")
    parser.add_argument("--reset", action="store_true", help="Reset collection before indexing")
    
    args = parser.parse_args()
    
    try:
        index_repository(args.repo_path, reset=args.reset)
    except KeyboardInterrupt:
        print("\n\nIndexing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
