"""
RAG service for retrieval and generation.
Clean implementation using LangChain Community.
"""
from typing import List, Dict
import sys
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class RAGService:
    """RAG pipeline using LangChain."""

    def __init__(self):
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )

        # Initialize LLM (using Ollama instead of ChatOllama)
        self.llm = Ollama(
            model=config.OLLAMA_CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0
        )

        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

        # Initialize vectorstore
        try:
            self.vectorstore = Chroma(
                collection_name=config.CHROMA_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_PERSIST_DIR
            )
            self.collection = self.chroma_client.get_collection(name=config.CHROMA_COLLECTION)
        except Exception as e:
            print(f"Warning: Could not load collection: {e}")
            self.vectorstore = None
            self.collection = None

        # Create prompt template
        template = """You are a helpful code assistant. Answer questions based ONLY on the provided code context.

Rules:
1. Use ONLY the information from the provided code chunks
2. If the context doesn't contain enough information, say so clearly
3. Cite file paths and line ranges when referencing specific code
4. Be concise but thorough
5. Never make up information not present in the context

Context from codebase:
{context}

Question: {question}

Answer:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _retrieve_chunks(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve relevant chunks from Chroma."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Please run indexing first.")

        # Use similarity search
        results = self.vectorstore.similarity_search_with_score(query=query, k=top_k)

        chunks = []
        for doc, score in results:
            chunks.append({
                "document": doc.page_content,
                "metadata": {
                    "file_path": doc.metadata.get("file_path", "unknown"),
                    "start_line": doc.metadata.get("start_line", 0),
                    "end_line": doc.metadata.get("end_line", 0)
                },
                "distance": float(score)
            })

        return chunks

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from chunks."""
        if not chunks:
            return ""

        context_parts = []
        current_length = 0

        for chunk in chunks:
            doc = chunk["document"]
            meta = chunk["metadata"]

            file_path = meta["file_path"]
            start_line = meta["start_line"]
            end_line = meta["end_line"]

            header = f"--- {file_path} (lines {start_line}-{end_line}) ---"
            chunk_text = f"{header}\n{doc}\n"

            # Check length limit
            if current_length + len(chunk_text) > config.MAX_CONTEXT_CHARS:
                if len(context_parts) == 0:
                    # At least include truncated first chunk
                    available = config.MAX_CONTEXT_CHARS - len(header) - 100
                    truncated = doc[:available]
                    context_parts.append(f"{header}\n{truncated}\n[Truncated]")
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n".join(context_parts)

    def query(self, question: str, top_k: int = None) -> Dict:
        """
        Main RAG query method.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Dict with 'answer' and 'sources'
        """
        if top_k is None:
            top_k = config.TOP_K_DEFAULT

        # Retrieve chunks
        chunks = self._retrieve_chunks(question, top_k)

        if not chunks:
            return {
                "answer": "I couldn't find any relevant code chunks for your question.",
                "sources": []
            }

        # Build context
        context = self._build_context(chunks)

        if not context:
            return {
                "answer": "I couldn't find any relevant code chunks for your question.",
                "sources": []
            }

        # Generate answer using chain
        try:
            answer = self.chain.run(context=context, question=question)
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = f"Error generating answer: {str(e)}"

        # Build sources
        sources = []
        for chunk in chunks:
            meta = chunk["metadata"]
            sources.append({
                "file_path": meta["file_path"],
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
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
            "chat_model": config.OLLAMA_CHAT_MODEL,
            "ollama_connected": False
        }

        # Check Chroma
        if self.collection:
            try:
                health["chroma_connected"] = True
                health["chunk_count"] = self.collection.count()
            except Exception as e:
                health["error"] = str(e)

        # Test Ollama
        try:
            test_result = self.llm.predict("test")
            health["ollama_connected"] = len(test_result) > 0
        except Exception as e:
            health["ollama_connected"] = False
            health["ollama_error"] = str(e)

        return health


# Global instance
rag_service = RAGService()