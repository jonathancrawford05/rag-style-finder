#!/bin/bash

# RAG Style Finder Launch Script
# This script sets up and launches the fashion analysis application

echo "🚀 Starting RAG Style Finder..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required. Found: $python_version"
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is required but not installed."
    echo "📥 Install from: https://ollama.ai/"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "🚀 Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Check for required model
if ! ollama list | grep -q "llava"; then
    echo "📥 Installing required vision model..."
    ollama pull llava:latest
fi

# Install Python dependencies if needed
if [ ! -d "venv" ] && [ ! -f "poetry.lock" ]; then
    echo "📦 Installing dependencies..."
    if command -v poetry &> /dev/null; then
        poetry install
    else
        pip install -r requirements.txt
    fi
fi

# Run setup if dataset is missing
if [ ! -f "swift-style-embeddings.pkl" ]; then
    echo "⚙️ Running initial setup..."
    python setup.py
fi

# Launch the application
echo "🎯 Launching Fashion Style Analyzer..."

if command -v poetry &> /dev/null && [ -f "poetry.lock" ]; then
    poetry run python main.py
else
    python main.py
fi
