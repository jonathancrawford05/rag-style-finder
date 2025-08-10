#!/usr/bin/env python3
"""
Setup script for RAG Style Finder application.
Downloads dataset and example images, sets up Ollama models.
"""

import os
import sys
import subprocess
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Ollama found: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ Ollama not found or not working")
            return False
    except FileNotFoundError:
        logger.error("❌ Ollama not installed")
        return False

def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama server is running")
            return True
        else:
            logger.error("❌ Ollama server not responding")
            return False
    except:
        logger.error("❌ Ollama server not running")
        return False

def install_ollama_model(model_name="llava:latest"):
    """Install required Ollama model."""
    try:
        logger.info(f"📥 Installing Ollama model: {model_name}")
        result = subprocess.run(['ollama', 'pull', model_name], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Model {model_name} installed successfully")
            return True
        else:
            logger.error(f"❌ Failed to install model {model_name}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error installing model: {e}")
        return False

def download_dataset():
    """Download the fashion dataset."""
    dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/95eJ0YJVtqTZhEd7RaUlew/processed-swift-style-embeddings.pkl"
    dataset_path = "swift-style-embeddings.pkl"
    
    if os.path.exists(dataset_path):
        logger.info("✅ Dataset already exists")
        return True
    
    try:
        logger.info("📥 Downloading fashion dataset...")
        response = requests.get(dataset_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(dataset_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✅ Dataset downloaded: {dataset_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download dataset: {e}")
        return False

def create_example_images():
    """Create example image placeholders."""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Example image URLs (you can replace with actual fashion images)
    example_urls = [
        "https://via.placeholder.com/400x600/FF0000/FFFFFF?text=Fashion+Example+1",
        "https://via.placeholder.com/400x600/00FF00/FFFFFF?text=Fashion+Example+2", 
        "https://via.placeholder.com/400x600/0000FF/FFFFFF?text=Fashion+Example+3"
    ]
    
    for i, url in enumerate(example_urls, 1):
        img_path = examples_dir / f"test-{i}.png"
        if not img_path.exists():
            try:
                logger.info(f"📥 Creating example image {i}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                img_path.write_bytes(response.content)
                logger.info(f"✅ Created {img_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not create example image {i}: {e}")

def main():
    """Main setup function."""
    logger.info("🚀 Setting up RAG Style Finder...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        logger.error("❌ Python 3.11+ required")
        return False
    
    # Check Ollama
    if not check_ollama():
        logger.error("Please install Ollama: https://ollama.ai/")
        return False
    
    # Start Ollama if not running
    if not check_ollama_running():
        logger.info("🚀 Starting Ollama server...")
        try:
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(3)  # Give it time to start
            if not check_ollama_running():
                logger.error("❌ Could not start Ollama server")
                return False
        except:
            logger.error("❌ Could not start Ollama server")
            return False
    
    # Install model
    if not install_ollama_model("llava:latest"):
        logger.warning("⚠️ Model installation failed, but you can install manually with: ollama pull llava:latest")
    
    # Download dataset
    if not download_dataset():
        logger.error("❌ Dataset download failed")
        return False
    
    # Create examples
    create_example_images()
    
    logger.info("✅ Setup complete!")
    logger.info("🚀 Run the app with: python main.py")
    
    return True

if __name__ == "__main__":
    main()
