"""
LLM service module for the RAG Style Finder application.
Uses Ollama instead of IBM Watson for multimodal fashion analysis.
"""
import logging
import base64
from typing import Dict, Any, Optional
import requests
import json

logger = logging.getLogger(__name__)

class OllamaVisionService:
    """Service for interacting with Ollama multimodal models."""
    
    def __init__(self, base_url: str = "http://localhost:11434", 
                 model: str = "llava:latest",
                 temperature: float = 0.2, 
                 top_p: float = 0.6, 
                 max_tokens: int = 2000):
        """
        Initialize the Ollama vision service.
        
        Args:
            base_url: Ollama server URL
            model: Multimodal model name (llava, llama3.2-vision, etc.)
            temperature: Controls randomness in generation
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens in response
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if Ollama is running and model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                logger.info(f"To install the model, run: ollama pull {self.model}")
                # Don't fail here, just warn
            else:
                logger.info(f"Successfully connected to Ollama with model: {self.model}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            return False

    def generate_response(self, encoded_image: str, prompt: str) -> str:
        """
        Generate a response from the model based on an image and prompt.
        
        Args:
            encoded_image: Base64-encoded image string
            prompt: Text prompt to guide the model's response
        
        Returns:
            str: Model's response
        """
        try:
            logger.info(f"Sending request to Ollama with prompt length: {len(prompt)}")
            
            # Prepare the request payload for Ollama
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [encoded_image],  # Ollama expects images as base64 strings
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens
                }
            }
            
            # Send request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # Increased timeout for vision models
            )
            
            response.raise_for_status()
            result = response.json()
            
            content = result.get('response', '')
            
            logger.info(f"Received response with length: {len(content)}")
            
            # Check if response appears to be truncated
            if len(content) >= self.max_tokens * 0.95:
                logger.warning(f"Response may be truncated (length: {len(content)})")
            
            return content
            
        except requests.exceptions.Timeout:
            logger.error("Request timed out - vision models can be slow")
            return "Error: Request timed out. The vision model may be processing slowly."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return f"Error communicating with Ollama: {e}"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"

    def generate_fashion_response(self, user_image_base64: str, matched_row: Any, 
                                all_items: Any, similarity_score: float, 
                                threshold: float = 0.8) -> str:
        """
        Generate a fashion-specific response using role-based prompts.
        
        Args:
            user_image_base64: Base64-encoded user-uploaded image
            matched_row: The closest match row from the dataset
            all_items: DataFrame with all items related to the matched image
            similarity_score: Similarity score between user and matched images
            threshold: Minimum similarity for considering an exact match
        
        Returns:
            str: Detailed fashion response
        """
        # Generate a formatted list of items with prices and links
        items_list = []
        for _, row in all_items.iterrows():
            item_str = f"{row['Item Name']} (${row['Price']}): {row['Link']}"
            items_list.append(item_str)
        
        # Join items with clear separators
        items_description = "\\n".join([f"- {item}" for item in items_list])
        
        if similarity_score >= threshold:
            # Prompt for exact matches
            assistant_prompt = f"""You are a professional fashion analyst conducting a retail catalog analysis. 

Analyze this fashion image and provide a detailed assessment. The image shows clothing items available in department stores.

Focus on:
1. Identify and describe the clothing items objectively (colors, patterns, materials, cuts)
2. Categorize the overall style (business, casual, formal, etc.)
3. Note any distinctive design elements or styling details
4. Use professional, clinical language appropriate for a retail catalog

ITEM DETAILS (include this section in your response):
{items_description}

Provide a comprehensive analysis followed by the ITEM DETAILS section."""

        else:
            # Prompt for similar but not exact matches
            assistant_prompt = f"""You are a professional fashion analyst conducting a retail catalog analysis.

Analyze this fashion image and provide a detailed assessment. Note that the items shown below are visually similar but not exact matches to what's in the image.

Focus on:
1. Identify clothing elements in the image objectively (colors, patterns, materials)
2. Explain how the image relates to the similar items found
3. Categorize the overall style and aesthetic
4. Use professional, clinical language appropriate for a retail catalog

SIMILAR ITEMS FOUND (include this section in your response):
{items_description}

Provide a comprehensive analysis followed by the SIMILAR ITEMS FOUND section."""

        # Send the prompt to the model
        response = self.generate_response(user_image_base64, assistant_prompt)
        
        # Check if response is incomplete and add failsafe
        if len(response) < 100:
            logger.info("Response appears incomplete, creating basic response")
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS FOUND:"
            response = f"""# Fashion Analysis

This outfit features a collection of carefully coordinated pieces suitable for various styling approaches.

{section_header}
{items_description}"""
        
        # Ensure the items list is included - this is crucial for user value
        elif "ITEM DETAILS:" not in response and "SIMILAR ITEMS FOUND:" not in response:
            logger.info("Item details section missing from response")
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS FOUND:"
            response += f"\\n\\n{section_header}\\n{items_description}"
        
        return response

    def list_available_models(self) -> list:
        """List all available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
