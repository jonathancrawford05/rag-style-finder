"""
RAG Style Finder - Multimodal Fashion Analysis Application

A computer vision and LLM-powered application for analyzing fashion images
and finding similar items using open source tools.
"""

__version__ = "0.1.0"
__author__ = "V Crawford"

from .app import StyleFinderApp, main
from .image_processor import ImageProcessor
from .llm_service import OllamaVisionService
from .helpers import process_response, get_all_items_for_image

__all__ = [
    "StyleFinderApp",
    "ImageProcessor", 
    "OllamaVisionService",
    "main",
    "process_response",
    "get_all_items_for_image"
]
