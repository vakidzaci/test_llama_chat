#!/usr/bin/env python3
"""
Diagnostic script to check Chroma collection structure.
Run this to see what's actually in your indexed data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

import chromadb

def diagnose_collection():
    """Check collection structure and data."""
    print("=" * 60)
    print("Chroma Collection Diagnostic")
    print("=" * 60)
    
    try:
        # Connect to Chroma
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        print(f"✅ Connected to Chroma at: {config.CHROMA_PERSIST_DIR}")
        
        # Get collection
        collection = client.get_collection(name=config.CHROMA_COLLECTION)
        print(f"✅ Found collection: {config.CHROMA_COLLECTION}")
        
        # Check count
        count = collection.count()
        print(f"✅ Total chunks: {count}")
        
        if count == 0:
            print("\n❌ Collection is empty! Run indexing first:")
            print("   python scripts/index_codebase.py /path/to/repo")
            return
        
        # Peek at data
        print("\n" + "=" * 60)
        print("Sample Data (first 3 items)")
        print("=" * 60)
        
        sample = collection.peek(limit=3)
        
        if sample:
            print("\n📄 Documents:")
            for i, doc in enumerate(sample.get('documents', [])[:3], 1):
                print(f"\n{i}. {doc[:100]}..." if len(doc) > 100 else f"\n{i}. {doc}")
            
            print("\n📋 Metadata:")
            for i, meta in enumerate(sample.get('metadatas', [])[:3], 1):
                print(f"\n{i}. {meta}")
            
            print("\n🔑 IDs:")
            for i, id_ in enumerate(sample.get('ids', [])[:3], 1):
                print(f"{i}. {id_}")
            
            print("\n📊 Embeddings:")
            embeddings = sample.get('embeddings', [])
            if embeddings:
                print(f"Shape: {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")
            else:
                print("No embeddings in peek (normal)")
        
        # Test query
        print("\n" + "=" * 60)
        print("Testing Similarity Search")
        print("=" * 60)
        
        try:
            # Get actual embedding from Ollama for better test
            from langchain_community.embeddings import OllamaEmbeddings
            
            embeddings = OllamaEmbeddings(
                model=config.OLLAMA_EMBED_MODEL,
                base_url=config.OLLAMA_BASE_URL
            )
            
            print("Generating test embedding with Ollama...")
            test_embedding = embeddings.embed_query("test query")
            print(f"✅ Generated embedding with {len(test_embedding)} dimensions")
            
            # Try direct query with real embedding
            results = collection.query(
                query_embeddings=[test_embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )
            
            print("✅ Direct query works!")
            if results.get('metadatas') and results['metadatas'][0]:
                print(f"   Result metadata: {results['metadatas'][0][0]}")
            if results.get('distances') and results['distances'][0]:
                print(f"   Has distances: {len(results['distances'][0])} results")
            
        except Exception as e:
            print(f"⚠️  Direct query failed (this is OK if LangChain works): {e}")
        
        # Test with LangChain
        print("\n" + "=" * 60)
        print("Testing LangChain Integration")
        print("=" * 60)
        
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            from langchain_community.vectorstores import Chroma
            
            embeddings = OllamaEmbeddings(
                model=config.OLLAMA_EMBED_MODEL,
                base_url=config.OLLAMA_BASE_URL
            )
            
            vectorstore = Chroma(
                collection_name=config.CHROMA_COLLECTION,
                embedding_function=embeddings,
                persist_directory=config.CHROMA_PERSIST_DIR
            )
            
            print("✅ LangChain vectorstore initialized")
            
            # Test search
            results = vectorstore.similarity_search_with_score(
                query="test",
                k=1
            )
            
            print(f"✅ LangChain search works! Got {len(results)} results")
            
            if results:
                doc, score = results[0]
                print(f"\n📄 Document preview: {doc.page_content[:100]}...")
                print(f"📋 Metadata: {doc.metadata}")
                print(f"📊 Score: {score}")
                print(f"   Score type: {type(score)}")
                
        except Exception as e:
            print(f"❌ LangChain test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Diagnosis Complete")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_collection()
