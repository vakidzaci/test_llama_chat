"""
RAG service for retrieval and generation.
Handles embedding, retrieval from Chroma, and chat generation with Ollama.
"""
import requests
import chromadb
from typing import List, Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class RAGService:
    """Handles RAG pipeline: embed query -> retrieve -> generate answer."""
    
    def __init__(self):
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        
        try:
            self.collection = self.client.get_collection(name=config.CHROMA_COLLECTION)
            print(f"Connected to Chroma collection: {config.CHROMA_COLLECTION}")
        except Exception as e:
            print(f"Warning: Could not load collection '{config.CHROMA_COLLECTION}': {e}")
            self.collection = None
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama."""
        url = f"{config.OLLAMA_BASE_URL}/api/embed"
        payload = {
            "model": config.OLLAMA_EMBED_MODEL,
            "input": text
        }
        
        response = requests.post(url, json=payload, timeout=config.OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if "embeddings" in data and len(data["embeddings"]) > 0:
            return data["embeddings"][0]
        elif "embedding" in data:
            return data["embedding"]
        else:
            raise ValueError(f"Unexpected embedding response format")
    
    def _retrieve_chunks(self, query: str, top_k: int) -> Dict:
        """Retrieve relevant chunks from Chroma."""
        if not self.collection:
            raise ValueError("Collection not initialized. Please run indexing first.")
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def _build_context(self, results: Dict) -> str:
        """Build context string from retrieved chunks."""
        if not results or not results.get("documents") or not results["documents"][0]:
            return ""
        
        context_parts = []
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            file_path = meta.get("file_path", "unknown")
            start_line = meta.get("start_line", "?")
            end_line = meta.get("end_line", "?")
            
            header = f"--- {file_path} (lines {start_line}-{end_line}) ---"
            context_parts.append(f"{header}\n{doc}\n")
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > config.MAX_CONTEXT_CHARS:
            context = context[:config.MAX_CONTEXT_CHARS] + "\n\n[Context truncated due to length limit]"
        
        return context
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Ollama chat model."""
        system_prompt = """You are a helpful code assistant. You answer questions based ONLY on the provided code context.

Rules:
1. Use ONLY the information from the provided code chunks
2. If the context doesn't contain enough information to answer, say so clearly
3. Cite file paths and line ranges when referencing specific code
4. Be concise but thorough
5. If you're uncertain, express that uncertainty
6. Never make up information not present in the context"""
        
        user_message = f"""Context from codebase:

{context}

---

Question: {question}

Answer the question using ONLY the provided context. Cite file paths and line ranges when relevant."""
        
        url = f"{config.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": config.OLLAMA_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=config.OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        # Extract answer from response
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        else:
            raise ValueError(f"Unexpected chat response format: {data.keys()}")
    
    def query(self, question: str, top_k: int = None) -> Dict:
        """
        Main RAG query method.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (uses default if None)
        
        Returns:
            Dict with 'answer' and 'sources'
        """
        if top_k is None:
            top_k = config.TOP_K_DEFAULT
        
        # Retrieve relevant chunks
        results = self._retrieve_chunks(question, top_k)
        
        # Check if we got results
        if not results or not results.get("documents") or not results["documents"][0]:
            return {
                "answer": "I couldn't find any relevant code chunks for your question. The codebase might not be indexed yet, or your question might be outside the scope of the indexed code.",
                "sources": []
            }
        
        # Build context
        context = self._build_context(results)
        
        if not context:
            return {
                "answer": "I couldn't find any relevant code chunks for your question.",
                "sources": []
            }
        
        # Generate answer
        answer = self._generate_answer(question, context)
        
        # Build sources list
        sources = []
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for meta, distance in zip(metadatas, distances):
            sources.append({
                "file_path": meta.get("file_path", "unknown"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "distance": round(distance, 4)
            })
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def health_check(self) -> Dict:
        """Check service health."""
        health = {
            "chroma_connected": False,
            "collection_name": config.CHROMA_COLLECTION,
            "persist_dir": config.CHROMA_PERSIST_DIR,
            "chunk_count": 0,
            "ollama_base_url": config.OLLAMA_BASE_URL,
            "embed_model": config.OLLAMA_EMBED_MODEL,
            "chat_model": config.OLLAMA_CHAT_MODEL
        }
        
        if self.collection:
            try:
                health["chroma_connected"] = True
                health["chunk_count"] = self.collection.count()
            except Exception as e:
                health["error"] = str(e)
        else:
            health["error"] = "Collection not initialized"
        
        # Test Ollama connection
        try:
            test_response = requests.get(
                f"{config.OLLAMA_BASE_URL}/api/tags",
                timeout=5
            )
            health["ollama_connected"] = test_response.status_code == 200
        except Exception as e:
            health["ollama_connected"] = False
            health["ollama_error"] = str(e)
        
        return health


# Global RAG service instance
rag_service = RAGService()
