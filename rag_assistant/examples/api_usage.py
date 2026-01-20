#!/usr/bin/env python3
"""
Example script showing how to use the RAG API programmatically.
"""
import requests
import json
import sys

# Configuration
BACKEND_URL = "http://localhost:8000"


def register_user(username: str, password: str):
    """Register a new user and get API key."""
    response = requests.post(
        f"{BACKEND_URL}/auth/register",
        json={"username": username, "password": password}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Registration successful!")
        print(f"API Key: {data['api_key']}")
        print(f"Save this key - you'll need it for queries!")
        return data['api_key']
    else:
        print(f"❌ Registration failed: {response.json()}")
        return None


def login_user(username: str, password: str):
    """Login and get API key."""
    response = requests.post(
        f"{BACKEND_URL}/auth/login",
        json={"username": username, "password": password}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Login successful!")
        print(f"API Key: {data['api_key']}")
        return data['api_key']
    else:
        print(f"❌ Login failed: {response.json()}")
        return None


def query_codebase(api_key: str, question: str, top_k: int = 5):
    """Query the codebase."""
    response = requests.post(
        f"{BACKEND_URL}/rag/query",
        headers={"X-API-Key": api_key},
        json={"question": question, "top_k": top_k}
    )
    
    if response.status_code == 200:
        data = response.json()
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)
        print(f"\nAnswer:\n{data['answer']}")
        print("\n" + "-" * 60)
        print("Sources:")
        for i, source in enumerate(data['sources'], 1):
            print(f"{i}. {source['file_path']}")
            print(f"   Lines {source['start_line']}-{source['end_line']}")
            print(f"   Distance: {source['distance']:.4f}")
        print("=" * 60 + "\n")
        return data
    elif response.status_code == 429:
        print("❌ Rate limit exceeded. Please wait a moment.")
        return None
    else:
        print(f"❌ Query failed: {response.json()}")
        return None


def check_health():
    """Check system health."""
    response = requests.get(f"{BACKEND_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print("System Health:")
        print(f"  Status: {data['status']}")
        print(f"  Collection: {data['collection']}")
        print(f"  Chunks: {data['chunk_count']}")
        print(f"  Ollama: {'✅' if data['ollama_connected'] else '❌'}")
        print(f"  Chroma: {'✅' if data['chroma_connected'] else '❌'}")
        return True
    else:
        print("❌ Cannot connect to backend")
        return False


def example_workflow():
    """Example workflow: register, query, multiple questions."""
    print("🚀 RAG API Example Workflow\n")
    
    # Check health first
    if not check_health():
        print("\n⚠️  Backend is not running!")
        print("Start it with: python backend/main.py")
        return
    
    print("\n" + "=" * 60)
    
    # Register or login
    username = "demo_user"
    password = "demo_password"
    
    print(f"\n1. Attempting to register user '{username}'...")
    api_key = register_user(username, password)
    
    if not api_key:
        print(f"\n2. User exists, logging in instead...")
        api_key = login_user(username, password)
    
    if not api_key:
        print("\n❌ Authentication failed!")
        return
    
    print("\n" + "=" * 60)
    
    # Example queries
    questions = [
        "What is the main purpose of this codebase?",
        "How does authentication work?",
        "Explain the RAG pipeline implementation",
    ]
    
    print(f"\n3. Running example queries...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuery {i}/{len(questions)}:")
        result = query_codebase(api_key, question, top_k=3)
        
        if result and i < len(questions):
            input("\nPress Enter to continue to next query...")
    
    print("\n✅ Example workflow complete!")
    print(f"\nYour API key: {api_key}")
    print("Use this key for future queries.")


def interactive_mode():
    """Interactive query mode."""
    print("🎯 Interactive Query Mode\n")
    
    # Get API key
    api_key = input("Enter your API key (or 'new' to register): ").strip()
    
    if api_key.lower() == 'new':
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        api_key = register_user(username, password)
        
        if not api_key:
            return
    
    print("\nReady! Enter questions (or 'quit' to exit)\n")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        top_k = input("Top-K chunks (default 5): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 5
        
        query_codebase(api_key, question, top_k)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        example_workflow()
        
        print("\n💡 Tip: Run with 'interactive' for interactive mode:")
        print("   python examples/api_usage.py interactive")
