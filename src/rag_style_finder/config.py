"""
Configuration settings for the RAG Style Finder application.
Open source implementation using Ollama instead of IBM Watson.
"""
import os
from typing import List, Tuple

# Ollama Configuration (replacing IBM Watson)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llava:latest"  # Default multimodal model, can be changed to llama3.2-vision when available
TEMPERATURE = 0.2
TOP_P = 0.6
MAX_TOKENS = 2000

# Alternative models you can use with Ollama:
# - "llava:latest" (most stable multimodal model)
# - "llama3.2-vision:latest" (if available)
# - "bakllava:latest" (another multimodal option)

# Image Processing Configuration
IMAGE_SIZE: Tuple[int, int] = (224, 224)
NORMALIZATION_MEAN: List[float] = [0.485, 0.456, 0.406]
NORMALIZATION_STD: List[float] = [0.229, 0.224, 0.225]

# Vector Similarity Configuration
SIMILARITY_THRESHOLD: float = 0.3  # Lowered from 0.8 for better matching
MAX_RESULTS: int = 10

# Dataset Configuration
DATASET_PATH = "swift-style-embeddings.pkl"

# Application Configuration
APP_TITLE = "Fashion Style Analyzer - Open Source"
APP_DESCRIPTION = """
Upload an image to analyze fashion elements and get detailed information about the items.
This application combines computer vision, vector similarity, and large language models 
to provide detailed fashion analysis using open source tools.
"""

# Gradio Configuration
GRADIO_THEME = "soft"
GRADIO_PORT = 7860
GRADIO_SHARE = False

# Logging Configuration
LOG_LEVEL = "INFO"

# Model device configuration
DEVICE = "mps" if hasattr(os, 'environ') and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') else "cpu"

# You can override these with environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
DATASET_PATH = os.getenv("DATASET_PATH", DATASET_PATH)
