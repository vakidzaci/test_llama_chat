#!/usr/bin/env python3
"""
Test script to verify RAG assistant setup.
Checks all components and dependencies.
"""
import sys
from pathlib import Path
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
import config


def test_imports():
    """Test if all required packages are installed."""
    print("📦 Testing imports...")
    required_packages = [
        "fastapi",
        "uvicorn",
        "streamlit",
        "chromadb",
        "dotenv",
        "bcrypt",
        "requests",
        "pydantic"
    ]
    
    failed = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All packages installed\n")
    return True


def test_config():
    """Test if configuration is loaded."""
    print("⚙️  Testing configuration...")
    
    config_vars = [
        "CHROMA_PERSIST_DIR",
        "CHROMA_COLLECTION",
        "OLLAMA_BASE_URL",
        "OLLAMA_EMBED_MODEL",
        "OLLAMA_CHAT_MODEL",
        "TOP_K_DEFAULT",
        "RATE_LIMIT_RPM"
    ]
    
    for var in config_vars:
        value = getattr(config, var, None)
        print(f"  {var}: {value}")
    
    print("✅ Configuration loaded\n")
    return True


def test_ollama():
    """Test Ollama connection."""
    print("🤖 Testing Ollama connection...")
    
    try:
        response = requests.get(
            f"{config.OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )
        if response.status_code == 200:
            print(f"  ✅ Connected to Ollama at {config.OLLAMA_BASE_URL}")
            
            # Check for required models
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            
            embed_model = config.OLLAMA_EMBED_MODEL
            chat_model = config.OLLAMA_CHAT_MODEL
            
            if any(embed_model in m for m in models):
                print(f"  ✅ Embed model '{embed_model}' found")
            else:
                print(f"  ⚠️  Embed model '{embed_model}' not found")
                print(f"     Install with: ollama pull {embed_model}")
            
            if any(chat_model in m for m in models):
                print(f"  ✅ Chat model '{chat_model}' found")
            else:
                print(f"  ⚠️  Chat model '{chat_model}' not found")
                print(f"     Install with: ollama pull {chat_model}")
            
            print("✅ Ollama connection OK\n")
            return True
        else:
            print(f"  ❌ Ollama returned status code: {response.status_code}\n")
            return False
    except Exception as e:
        print(f"  ❌ Cannot connect to Ollama: {e}")
        print(f"     Make sure Ollama is running: ollama serve\n")
        return False


def test_chroma():
    """Test Chroma setup."""
    print("💾 Testing Chroma setup...")
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        print(f"  ✅ Chroma client initialized")
        
        # Try to get collection (might not exist yet)
        try:
            collection = client.get_collection(name=config.CHROMA_COLLECTION)
            count = collection.count()
            print(f"  ✅ Collection '{config.CHROMA_COLLECTION}' found with {count} chunks")
        except Exception:
            print(f"  ⚠️  Collection '{config.CHROMA_COLLECTION}' not found")
            print(f"     Run indexing: python scripts/index_codebase.py /path/to/repo")
        
        print("✅ Chroma setup OK\n")
        return True
    except Exception as e:
        print(f"  ❌ Chroma error: {e}\n")
        return False


def test_backend_running():
    """Test if backend is running."""
    print("🌐 Testing backend connection...")
    
    try:
        response = requests.get(
            f"http://{config.BACKEND_HOST}:{config.BACKEND_PORT}/health",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Backend is running")
            print(f"     Status: {data.get('status')}")
            print(f"     Chunks: {data.get('chunk_count')}")
            print("✅ Backend connection OK\n")
            return True
        else:
            print(f"  ⚠️  Backend responded with status: {response.status_code}\n")
            return False
    except Exception:
        print(f"  ⚠️  Backend not running")
        print(f"     Start with: python backend/main.py\n")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG Codebase Assistant - System Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Required tests
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Ollama", test_ollama()))
    results.append(("Chroma", test_chroma()))
    
    # Optional test (backend might not be running)
    results.append(("Backend", test_backend_running()))
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    print()
    
    # Check critical failures
    critical = results[:4]  # Imports, Config, Ollama, Chroma
    if all(passed for _, passed in critical):
        print("🎉 All critical tests passed! System is ready.")
        print()
        print("Next steps:")
        print("1. Index a codebase: python scripts/index_codebase.py /path/to/repo")
        print("2. Start backend: python backend/main.py")
        print("3. Start UI: streamlit run ui/app.py")
        return 0
    else:
        print("⚠️  Some critical tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
