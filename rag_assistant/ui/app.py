"""
Streamlit UI for the RAG Codebase Assistant.
Provides user interface for registration, login, and querying.
"""
import streamlit as st
import requests
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# Configuration
BACKEND_URL = f"http://{config.BACKEND_HOST}:{config.BACKEND_PORT}"


# Page configuration
st.set_page_config(
    page_title="RAG Codebase Assistant",
    page_icon="🔍",
    layout="wide"
)


# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "username" not in st.session_state:
    st.session_state.username = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []


def check_backend_health() -> dict:
    """Check if backend is accessible."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


def register_user(username: str, password: str) -> Optional[str]:
    """Register a new user."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/auth/register",
            json={"username": username, "password": password},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data["api_key"]
        else:
            st.error(f"Registration failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return None


def login_user(username: str, password: str) -> Optional[str]:
    """Login a user."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/auth/login",
            json={"username": username, "password": password},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data["api_key"]
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return None


def query_rag(question: str, top_k: int, api_key: str) -> Optional[dict]:
    """Query the RAG system."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/rag/query",
            json={"question": question, "top_k": top_k},
            headers={"X-API-Key": api_key},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.error("Rate limit exceeded. Please wait a moment and try again.")
            return None
        else:
            st.error(f"Query failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error querying backend: {e}")
        return None


def logout():
    """Logout the user."""
    st.session_state.api_key = None
    st.session_state.username = None
    st.session_state.query_history = []
    st.rerun()


def auth_page():
    """Display authentication page."""
    st.title("🔍 RAG Codebase Assistant")
    st.markdown("### Welcome! Please login or register to continue.")
    
    # Check backend health
    with st.spinner("Checking backend connection..."):
        health = check_backend_health()
    
    if not health:
        st.error("⚠️ Cannot connect to backend. Please ensure the FastAPI server is running.")
        st.code(f"Expected backend URL: {BACKEND_URL}")
        st.info("Start the backend with: `python backend/main.py` or `uvicorn backend.main:app`")
        return
    
    # Display backend status
    with st.expander("📊 Backend Status"):
        st.json(health)
    
    # Auth tabs
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    with st.spinner("Logging in..."):
                        api_key = login_user(username, password)
                        if api_key:
                            st.session_state.api_key = api_key
                            st.session_state.username = username
                            st.success("Login successful!")
                            st.rerun()
    
    with tab2:
        st.subheader("Register")
        with st.form("register_form"):
            username = st.text_input("Username (min 3 characters)")
            password = st.text_input("Password (min 6 characters)", type="password")
            password_confirm = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if not username or not password:
                    st.error("Please fill in all fields")
                elif len(username) < 3:
                    st.error("Username must be at least 3 characters")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif password != password_confirm:
                    st.error("Passwords do not match")
                else:
                    with st.spinner("Registering..."):
                        api_key = register_user(username, password)
                        if api_key:
                            st.session_state.api_key = api_key
                            st.session_state.username = username
                            st.success("Registration successful!")
                            st.rerun()


def main_page():
    """Display main query interface."""
    # Header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🔍 RAG Codebase Assistant")
    with col2:
        if st.button("Logout", type="secondary"):
            logout()
    
    st.markdown(f"**Logged in as:** {st.session_state.username}")
    st.markdown("---")
    
    # Sidebar with settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_k = st.slider(
            "Number of chunks to retrieve (top_k)",
            min_value=1,
            max_value=20,
            value=config.TOP_K_DEFAULT,
            help="More chunks = more context but slower response"
        )
        
        st.markdown("---")
        st.subheader("📊 System Info")
        
        if st.button("Refresh Status"):
            health = check_backend_health()
            if health:
                st.success("✅ Backend connected")
                st.metric("Indexed chunks", health.get("chunk_count", 0))
                st.metric("Collection", health.get("collection", "N/A"))
            else:
                st.error("❌ Backend disconnected")
    
    # Main query interface
    st.subheader("💬 Ask a Question")
    
    question = st.text_area(
        "Enter your question about the codebase:",
        height=100,
        placeholder="Example: How does the authentication system work?"
    )
    
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        query_button = st.button("🔍 Query", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.query_history = []
        st.rerun()
    
    if query_button:
        if not question.strip():
            st.error("Please enter a question")
        else:
            with st.spinner("Searching codebase and generating answer..."):
                result = query_rag(
                    question=question.strip(),
                    top_k=top_k,
                    api_key=st.session_state.api_key
                )
                
                if result:
                    # Add to history
                    st.session_state.query_history.insert(0, {
                        "question": question,
                        "result": result,
                        "top_k": top_k
                    })
    
    # Display results
    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("📝 Results")
        
        for idx, item in enumerate(st.session_state.query_history):
            with st.expander(
                f"Q: {item['question'][:100]}{'...' if len(item['question']) > 100 else ''}",
                expanded=(idx == 0)
            ):
                st.markdown("**Question:**")
                st.info(item['question'])
                
                st.markdown("**Answer:**")
                st.success(item['result']['answer'])
                
                st.markdown("**Sources:**")
                if item['result']['sources']:
                    for source in item['result']['sources']:
                        st.markdown(
                            f"- `{source['file_path']}` "
                            f"(lines {source['start_line']}-{source['end_line']}) "
                            f"- distance: {source['distance']:.4f}"
                        )
                else:
                    st.warning("No sources found")
                
                st.caption(f"Retrieved {item['top_k']} chunks")


def main():
    """Main application entry point."""
    # Check authentication
    if not st.session_state.api_key:
        auth_page()
    else:
        main_page()


if __name__ == "__main__":
    main()
