#!/usr/bin/env python3
"""
Quick dataset downloader for RAG Style Finder.
Downloads the fashion dataset with progress bar.
"""

import requests
import os
import sys
from pathlib import Path

def download_with_progress(url, filename):
    """Download file with progress bar."""
    print(f"📥 Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0:
            print(f"File size: {total_size / (1024*1024):.1f} MB")
        
        downloaded = 0
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 50
                        filled_length = int(bar_length * downloaded // total_size)
                        bar = '█' * filled_length + '-' * (bar_length - filled_length)
                        print(f'\r|{bar}| {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)', end='', flush=True)
        
        print(f"\n✅ Downloaded: {filename}")
        print(f"Final size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def verify_dataset(filename):
    """Verify the downloaded dataset."""
    try:
        import pandas as pd
        print(f"🔍 Verifying {filename}...")
        
        data = pd.read_pickle(filename)
        print(f"✅ Dataset verified!")
        print(f"  - Rows: {len(data)}")
        print(f"  - Columns: {list(data.columns)}")
        
        # Check embeddings
        if 'Embedding' in data.columns:
            embeddings = data['Embedding'].dropna()
            print(f"  - Valid embeddings: {len(embeddings)}")
            if len(embeddings) > 0:
                first_emb = embeddings.iloc[0]
                print(f"  - Embedding type: {type(first_emb)}")
                try:
                    import numpy as np
                    emb_array = np.array(first_emb)
                    print(f"  - Embedding shape: {emb_array.shape}")
                except:
                    print(f"  - Embedding shape: Could not determine")
        
        return True
    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        return False

def main():
    """Main download function."""
    dataset_file = "swift-style-embeddings.pkl"
    
    # Check if already exists
    if os.path.exists(dataset_file):
        print(f"✅ Dataset already exists: {dataset_file}")
        if verify_dataset(dataset_file):
            print("🎉 Dataset is ready to use!")
            return True
        else:
            print("⚠️ Existing dataset appears corrupted, re-downloading...")
            os.remove(dataset_file)
    
    # URLs to try
    urls = [
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/95eJ0YJVtqTZhEd7RaUlew/processed-swift-style-embeddings.pkl",
        # Backup URL if needed
        "https://github.com/HaileyTQuach/style-finder/raw/main/swift-style-embeddings.pkl"
    ]
    
    for i, url in enumerate(urls, 1):
        print(f"\n🌐 Trying URL {i}/{len(urls)}...")
        if download_with_progress(url, dataset_file):
            if verify_dataset(dataset_file):
                print("\n🎉 Dataset downloaded and verified successfully!")
                print("🚀 You can now run: python main.py")
                return True
            else:
                print("⚠️ Downloaded file appears corrupted, trying next URL...")
                if os.path.exists(dataset_file):
                    os.remove(dataset_file)
        else:
            print(f"❌ URL {i} failed, trying next...")
    
    print("\n❌ All download attempts failed!")
    print("\n💡 Manual alternatives:")
    print("1. Check if you have the dataset file from the course materials")
    print("2. Contact the course instructor for the dataset")
    print("3. Use a different fashion dataset with similar structure")
    
    return False

if __name__ == "__main__":
    main()
