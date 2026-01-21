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
                    # Continue with next result
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

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks, respecting context limits."""
        if not chunks:
            return ""

        context_parts = []
        current_length = 0
        chunks_used = 0

        for chunk in chunks:
            try:
                doc = chunk.get("document", "")
                meta = chunk.get("metadata", {})

                file_path = meta.get("file_path", "unknown")
                start_line = meta.get("start_line", "?")
                end_line = meta.get("end_line", "?")

                header = f"--- {file_path} (lines {start_line}-{end_line}) ---"
                chunk_text = f"{header}\n{doc}\n"
                chunk_length = len(chunk_text)

                # Check if adding this chunk would exceed limit
                if current_length + chunk_length > config.MAX_CONTEXT_CHARS:
                    # Only add if we haven't added any chunks yet (prevents empty context)
                    if chunks_used == 0:
                        # Truncate this first chunk to fit
                        available = config.MAX_CONTEXT_CHARS - len(header) - 100  # Leave room for header and notice
                        truncated_doc = doc[:available]
                        chunk_text = f"{header}\n{truncated_doc}\n\n[Chunk truncated to fit context limit]"
                        context_parts.append(chunk_text)
                        chunks_used += 1
                    break  # Stop adding more chunks

                context_parts.append(chunk_text)
                current_length += chunk_length
                chunks_used += 1

            except Exception as e:
                print(f"Warning: Error processing chunk for context: {e}")
                continue

        context = "\n".join(context_parts)

        # Add notice if we didn't use all chunks
        if chunks_used < len(chunks):
            context += f"\n\n[Note: Showing {chunks_used} of {len(chunks)} retrieved chunks due to context limit]"

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

        try:
            # Generate response
            response = self.chat_model.invoke(messages)
            return response.content

        except Exception as e:
            error_msg = str(e).lower()

            # Check for context length errors
            if "context" in error_msg and "length" in error_msg:
                raise ValueError(
                    "Context too large for the model. Try:\n"
                    "1. Reducing top_k (use fewer chunks)\n"
                    "2. Using a model with larger context (e.g., llama3.2:8k)\n"
                    "3. Adjusting MAX_CONTEXT_CHARS in .env"
                )
            elif "exceeds" in error_msg or "maximum" in error_msg:
                raise ValueError(
                    f"Input exceeds model limits. Reduce top_k or use a larger model. Error: {e}"
                )
            else:
                # Re-raise other errors
                raise

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
            try:
                meta = chunk.get("metadata", {})
                distance = chunk.get("distance", 0.0)

                sources.append({
                    "file_path": meta.get("file_path", "unknown"),
                    "start_line": meta.get("start_line", 0),
                    "end_line": meta.get("end_line", 0),
                    "distance": round(float(distance), 4)
                })
            except Exception as e:
                print(f"Warning: Error processing chunk for sources: {e}")
                # Skip malformed chunks
                continue

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