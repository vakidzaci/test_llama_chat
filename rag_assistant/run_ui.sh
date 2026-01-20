#!/bin/bash
# Script to start the Streamlit UI

echo "🎨 Starting RAG Codebase Assistant UI..."
echo "UI will be available at: http://localhost:8501"
echo ""

# Change to project root
cd "$(dirname "$0")"

# Start Streamlit
streamlit run ui/app.py
