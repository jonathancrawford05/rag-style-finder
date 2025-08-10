#!/usr/bin/env python3
"""
Quick test to verify the similarity threshold fix works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def quick_match_test():
    """Test if image matching now works with lower threshold."""
    print("🧪 Quick Match Test")
    print("="*30)
    
    try:
        # Import our modules
        from rag_style_finder.image_processor import ImageProcessor
        from rag_style_finder.config import SIMILARITY_THRESHOLD
        import pandas as pd
        import os
        
        print(f"📏 Current similarity threshold: {SIMILARITY_THRESHOLD}")
        
        # Load dataset
        if not os.path.exists("swift-style-embeddings.pkl"):
            print("❌ Dataset file not found!")
            return False
            
        data = pd.read_pickle("swift-style-embeddings.pkl")
        print(f"📊 Dataset loaded: {len(data)} items")
        
        # Initialize image processor
        processor = ImageProcessor()
        print("🧠 Image processor initialized")
        
        # Test with example image
        test_image = "examples/test-1.png"
        if not os.path.exists(test_image):
            print(f"❌ Test image not found: {test_image}")
            return False
            
        print(f"🖼️ Testing with: {test_image}")
        
        # Encode image
        result = processor.encode_image(test_image, is_url=False)
        if result['vector'] is None:
            print("❌ Image encoding failed")
            return False
            
        print("✅ Image encoded successfully")
        
        # Find closest match
        closest_row, similarity_score = processor.find_closest_match(result['vector'], data)
        
        if closest_row is not None:
            print(f"🎉 MATCH FOUND!")
            print(f"  - Item: {closest_row.get('Item Name', 'Unknown')}")
            print(f"  - Similarity: {similarity_score:.4f}")
            print(f"  - Threshold: {SIMILARITY_THRESHOLD}")
            
            if similarity_score >= SIMILARITY_THRESHOLD:
                print(f"✅ Above threshold - would show as EXACT match")
            else:
                print(f"⚠️ Below threshold - would show as SIMILAR items")
            
            return True
        else:
            print("❌ Still no match found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run the quick test."""
    print("🚀 Testing Similarity Threshold Fix")
    print("="*40)
    
    if quick_match_test():
        print("\n🎉 SUCCESS! The fix works!")
        print("🚀 Now restart your app: python main.py")
        print("📷 Try the same image again")
    else:
        print("\n❌ Still having issues")
        print("🔍 Run the deep debug: python deep_debug.py")

if __name__ == "__main__":
    main()
