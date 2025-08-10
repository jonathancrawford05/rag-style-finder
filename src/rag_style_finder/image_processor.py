"""
Image processing module for the RAG Style Finder application.
Handles image encoding, vector similarity, and fashion dataset matching.
"""
import base64
import logging
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Tuple, Optional

import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Processes images for fashion analysis using ResNet50 embeddings."""
    
    def __init__(self, image_size=(224, 224), 
                 norm_mean=[0.485, 0.456, 0.406],
                 norm_std=[0.229, 0.224, 0.225]):
        """
        Initialize the image processor with a pre-trained ResNet50 model.
        
        Args:
            image_size (tuple): Target size for input images
            norm_mean (list): Normalization mean values for RGB channels
            norm_std (list): Normalization standard deviation values for RGB channels
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained ResNet50 model (KEEP classification layer for dataset compatibility)
        # IMPORTANT: The dataset was created using ResNet50 WITH the classification layer,
        # which produces 1000-dimensional vectors (ImageNet classes). 
        # DO NOT remove the classification layer or we get 2048-dim vectors that don't match!
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    def encode_image(self, image_input: str, is_url: bool = True) -> Dict[str, Any]:
        """
        Encode an image and extract its feature vector.
        
        Args:
            image_input: URL or local path to the image
            is_url: Whether the input is a URL (True) or a local file path (False)
        
        Returns:
            dict: Contains 'base64' string and 'vector' (feature embedding)
        """
        try:
            if is_url:
                # Fetch the image from URL
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # Load the image from a local file
                image = Image.open(image_input).convert("RGB")
            
            # Convert image to Base64 for LLM input
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Preprocess the image for ResNet50
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Extract features using ResNet50 (with classification layer)
            with torch.no_grad():
                features = self.model(input_tensor)
            
            # Convert to NumPy array (already 1000-dimensional from classification layer)
            feature_vector = features.cpu().numpy().flatten()
            
            return {
                "base64": base64_string, 
                "vector": feature_vector
            }
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return {"base64": None, "vector": None}

    def find_closest_match(self, user_vector: np.ndarray, 
                          dataset) -> Tuple[Optional[Any], Optional[float]]:
        """
        Find the closest match in the dataset based on cosine similarity.
        
        Args:
            user_vector: Feature vector of the user-uploaded image
            dataset: DataFrame containing precomputed feature vectors
        
        Returns:
            tuple: (Closest matching row, similarity score)
        """
        try:
            # Extract embeddings from dataset
            embeddings = dataset['Embedding'].dropna()
            if embeddings.empty:
                logger.error("No embeddings found in dataset")
                return None, None
            
            # Stack all embeddings into a single array
            dataset_vectors = np.vstack(embeddings.values)
            
            # Compute cosine similarity
            similarities = cosine_similarity(user_vector.reshape(1, -1), dataset_vectors)
            
            # Find the index of the most similar vector
            closest_index = np.argmax(similarities)
            similarity_score = similarities[0][closest_index]
            
            # Debug: Log similarity statistics
            logger.info(f"Similarity scores - Max: {similarities.max():.4f}, Min: {similarities.min():.4f}, Mean: {similarities.mean():.4f}")
            logger.info(f"Top 5 similarities: {np.sort(similarities[0])[-5:][::-1]}")
            
            # Get the actual index in the original dataset
            original_index = embeddings.index[closest_index]
            closest_row = dataset.loc[original_index]
            
            logger.info(f"Found closest match with similarity: {similarity_score:.3f}")
            return closest_row, similarity_score
            
        except Exception as e:
            logger.error(f"Error finding closest match: {e}")
            return None, None

    def batch_encode_images(self, image_paths: list) -> list:
        """
        Encode multiple images in batch for efficiency.
        
        Args:
            image_paths: List of image paths or URLs
            
        Returns:
            list: List of encoded image dictionaries
        """
        results = []
        for i, path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}")
            is_url = path.startswith(('http://', 'https://'))
            result = self.encode_image(path, is_url=is_url)
            results.append(result)
        return results
