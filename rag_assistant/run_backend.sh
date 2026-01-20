#!/bin/bash
# Script to start the FastAPI backend

echo "🚀 Starting RAG Codebase Assistant Backend..."
echo "Backend will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""

# Change to project root
cd "$(dirname "$0")"

# Start backend
python backend/main.py
