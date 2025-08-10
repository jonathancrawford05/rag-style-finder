#!/usr/bin/env python3
"""
Test script for RAG Style Finder application.
Verifies all components are working correctly.
"""

import sys
import os
import logging
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    try:
        import torch
        import torchvision
        import transformers
        import faiss
        import gradio
        import ollama
        import langchain
        logger.info("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection and model availability."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            logger.info(f"✅ Ollama connected. Available models: {model_names}")
            
            # Check for vision models
            vision_models = [m for m in model_names if any(vm in m for vm in ['llava', 'vision', 'bakllava'])]
            if vision_models:
                logger.info(f"✅ Vision models found: {vision_models}")
                return True
            else:
                logger.warning("⚠️ No vision models found. Run: ollama pull llava:latest")
                return False
        else:
            logger.error("❌ Ollama server responded with error")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to Ollama: {e}")
        logger.info("💡 Make sure Ollama is running: ollama serve")
        return False

def test_torch_device():
    """Test PyTorch device availability."""
    try:
        import torch
        
        # Test CUDA
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            device = "cuda"
        # Test MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("✅ MPS (Apple Silicon) available")
            device = "mps"
        else:
            logger.info("✅ Using CPU")
            device = "cpu"
        
        # Test tensor creation
        test_tensor = torch.randn(10).to(device)
        logger.info(f"✅ PyTorch device test passed: {device}")
        return True
    except Exception as e:
        logger.error(f"❌ PyTorch device test failed: {e}")
        return False

def test_image_processing():
    """Test image processing components."""
    try:
        from rag_style_finder.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        logger.info("✅ Image processor initialized")
        
        # Test with a simple test image
        import numpy as np
        from PIL import Image
        
        # Create a test image
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_path = "test_image.jpg"
        test_img.save(test_path)
        
        # Test encoding
        result = processor.encode_image(test_path, is_url=False)
        if result['vector'] is not None and result['base64'] is not None:
            logger.info("✅ Image encoding test passed")
            os.remove(test_path)
            return True
        else:
            logger.error("❌ Image encoding failed")
            return False
    except Exception as e:
        logger.error(f"❌ Image processing test failed: {e}")
        return False

def test_dataset():
    """Test dataset loading."""
    dataset_path = "swift-style-embeddings.pkl"
    if os.path.exists(dataset_path):
        try:
            import pandas as pd
            data = pd.read_pickle(dataset_path)
            logger.info(f"✅ Dataset loaded: {len(data)} items")
            
            required_cols = ['Item Name', 'Price', 'Link', 'Image URL', 'Embedding']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"⚠️ Missing columns: {missing_cols}")
            else:
                logger.info("✅ Dataset structure is valid")
            return True
        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {e}")
            return False
    else:
        logger.warning(f"⚠️ Dataset not found: {dataset_path}")
        logger.info("💡 Run setup.py to download the dataset")
        return False

def test_ollama_vision():
    """Test Ollama vision model."""
    try:
        from rag_style_finder.llm_service import OllamaVisionService
        
        service = OllamaVisionService()
        
        # Create a simple test image in base64
        import base64
        from PIL import Image
        import io
        
        # Create test image
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Test response (short timeout for testing)
        response = service.generate_response(img_b64, "Describe this image briefly.")
        
        if response and not response.startswith("Error"):
            logger.info("✅ Ollama vision model test passed")
            return True
        else:
            logger.error(f"❌ Ollama vision test failed: {response}")
            return False
    except Exception as e:
        logger.error(f"❌ Ollama vision test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Running RAG Style Finder tests...")
    
    tests = [
        ("Package Imports", test_imports),
        ("Ollama Connection", test_ollama_connection),
        ("PyTorch Device", test_torch_device),
        ("Image Processing", test_image_processing),
        ("Dataset Loading", test_dataset),
        ("Ollama Vision Model", test_ollama_vision),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("🏁 TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! Your application is ready to use.")
        logger.info("🚀 Run: python main.py")
    else:
        logger.info("⚠️ Some tests failed. Check the setup instructions.")
        logger.info("💡 Run: python setup.py")

if __name__ == "__main__":
    main()
