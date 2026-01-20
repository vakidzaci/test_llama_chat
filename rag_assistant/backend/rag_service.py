"""
RAG service for retrieval and generation.
Handles embedding, retrieval from Chroma, and chat generation with Ollama.
Uses LangChain Community for integration.
"""
from typing import List, Dict, Optional
import sys
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class RAGService:
    """Handles RAG pipeline: embed query -> retrieve -> generate answer using LangChain."""
    
    def __init__(self):
        # Initialize LangChain embeddings
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        
        # Initialize LangChain chat model
        self.chat_model = ChatOllama(
            model=config.OLLAMA_CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0
        )
        
        # Initialize Chroma client for health checks
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        
        # Initialize LangChain Chroma vectorstore
        try:
            self.vectorstore = Chroma(
                collection_name=config.CHROMA_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_PERSIST_DIR
            )
            print(f"Connected to Chroma collection: {config.CHROMA_COLLECTION}")
            self.collection = self.chroma_client.get_collection(name=config.CHROMA_COLLECTION)
        except Exception as e:
            print(f"Warning: Could not load collection '{config.CHROMA_COLLECTION}': {e}")
            self.vectorstore = None
            self.collection = None
    
    def _retrieve_chunks(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve relevant chunks from Chroma using LangChain."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Please run indexing first.")
        
        # Use LangChain's similarity search with scores
        results_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        # Convert to our format
        chunks = []
        for doc, score in results_with_scores:
            chunks.append({
                "document": doc.page_content,
                "metadata": doc.metadata,
                "distance": float(score)  # LangChain returns distance/score
            })
        
        return chunks
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return ""
        
        context_parts = []
        
        for chunk in chunks:
            doc = chunk["document"]
            meta = chunk["metadata"]
            
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
        """Generate answer using LangChain's ChatOllama."""
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
        
        # Use LangChain's message format
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        # Generate response
        response = self.chat_model.invoke(messages)
        
        return response.content
    
    def query(self, question: str, top_k: int = None) -> Dict:
        """
        Main RAG query method using LangChain.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (uses default if None)
        
        Returns:
            Dict with 'answer' and 'sources'
        """
        if top_k is None:
            top_k = config.TOP_K_DEFAULT
        
        # Retrieve relevant chunks
        chunks = self._retrieve_chunks(question, top_k)
        
        # Check if we got results
        if not chunks:
            return {
                "answer": "I couldn't find any relevant code chunks for your question. The codebase might not be indexed yet, or your question might be outside the scope of the indexed code.",
                "sources": []
            }
        
        # Build context
        context = self._build_context(chunks)
        
        if not context:
            return {
                "answer": "I couldn't find any relevant code chunks for your question.",
                "sources": []
            }
        
        # Generate answer
        answer = self._generate_answer(question, context)
        
        # Build sources list
        sources = []
        for chunk in chunks:
            meta = chunk["metadata"]
            sources.append({
                "file_path": meta.get("file_path", "unknown"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "distance": round(chunk["distance"], 4)
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
        
        # Test Ollama connection via LangChain
        try:
            # Test embeddings
            test_embed = self.embeddings.embed_query("test")
            health["ollama_embeddings_connected"] = len(test_embed) > 0
        except Exception as e:
            health["ollama_embeddings_connected"] = False
            health["ollama_embeddings_error"] = str(e)
        
        try:
            # Test chat model
            test_response = self.chat_model.invoke("test")
            health["ollama_chat_connected"] = len(test_response.content) > 0
        except Exception as e:
            health["ollama_chat_connected"] = False
            health["ollama_chat_error"] = str(e)
        
        health["ollama_connected"] = health.get("ollama_embeddings_connected", False) and health.get("ollama_chat_connected", False)
        
        return health


# Global RAG service instance
rag_service = RAGService()
