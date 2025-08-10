#!/usr/bin/env python3
"""
Test to verify the embedding dimension fix.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_dimension_fix():
    """Test that embedding dimensions now match between dataset and ResNet50."""
    print("🔧 Testing Embedding Dimension Fix")
    print("="*40)
    
    try:
        # Test 1: Check dataset embedding dimensions
        import pandas as pd
        import numpy as np
        import os
        
        if not os.path.exists("swift-style-embeddings.pkl"):
            print("❌ Dataset file not found!")
            return False
            
        data = pd.read_pickle("swift-style-embeddings.pkl")
        embeddings = data['Embedding'].dropna()
        
        if len(embeddings) == 0:
            print("❌ No embeddings in dataset!")
            return False
            
        first_emb = embeddings.iloc[0]
        dataset_array = np.array(first_emb)
        
        print(f"📊 Dataset embedding shape: {dataset_array.shape}")
        print(f"📊 Dataset embedding type: {dataset_array.dtype}")
        
        # Test 2: Check ResNet50 output dimensions
        from rag_style_finder.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Test with example image
        test_image = "examples/test-1.png"
        if not os.path.exists(test_image):
            print(f"❌ Test image not found: {test_image}")
            return False
            
        result = processor.encode_image(test_image, is_url=False)
        if result['vector'] is None:
            print("❌ Image encoding failed")
            return False
            
        resnet_vector = result['vector']
        print(f"🧠 ResNet50 output shape: {resnet_vector.shape}")
        print(f"🧠 ResNet50 output type: {resnet_vector.dtype}")
        
        # Test 3: Compare dimensions
        print(f"\n⚖️ Dimension Comparison:")
        if dataset_array.shape == resnet_vector.shape:
            print(f"✅ DIMENSIONS MATCH! {dataset_array.shape}")
            
            # Test 4: Try similarity computation
            print(f"\n🧮 Testing similarity computation...")
            
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Stack a few dataset embeddings
            test_embeddings = embeddings.head(5)
            dataset_vectors = np.vstack(test_embeddings.values)
            
            # Compute similarity
            similarities = cosine_similarity(resnet_vector.reshape(1, -1), dataset_vectors)
            
            print(f"✅ Similarity computation successful!")
            print(f"📈 Similarity scores: {similarities[0]}")
            print(f"📈 Max similarity: {similarities.max():.4f}")
            print(f"📈 Min similarity: {similarities.min():.4f}")
            
            return True
            
        else:
            print(f"❌ DIMENSIONS STILL DON'T MATCH!")
            print(f"   Dataset: {dataset_array.shape}")
            print(f"   ResNet50: {resnet_vector.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the dimension fix test."""
    print("🔧 DIMENSION FIX VERIFICATION")
    print("="*50)
    
    if test_dimension_fix():
        print(f"\n🎉 SUCCESS! Dimensions are now compatible!")
        print(f"🚀 Now test the app: python main.py")
        print(f"📷 Try the same image that failed before")
    else:
        print(f"\n❌ Dimension fix didn't work")
        print(f"🔍 Check the error messages above")

if __name__ == "__main__":
    main()
