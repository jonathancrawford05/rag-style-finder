"""
Helper functions for the RAG Style Finder application.
Includes response processing, dataset utilities, and formatting functions.
"""
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def process_response(response: str) -> str:
    """
    Process and clean the AI-generated response for better display.
    
    Args:
        response: Raw response from the AI model
        
    Returns:
        str: Cleaned and formatted response
    """
    if not response:
        return "No response generated. Please try again with a different image."
    
    # Handle common AI refusal patterns
    refusal_phrases = [
        "I'm not able to provide",
        "I cannot",
        "I apologize, but",
        "I don't feel comfortable",
        "I'm unable to",
        "I can't"
    ]
    
    # Check if the response contains refusal patterns
    if any(phrase in response for phrase in refusal_phrases):
        # Extract any useful information that might be in the response
        useful_parts = []
        lines = response.split('\n')
        
        for line in lines:
            # Keep lines that contain actual fashion analysis
            if any(keyword in line.lower() for keyword in 
                  ['item', 'fashion', 'style', 'clothing', 'outfit', 'garment', 'fabric', 'color']):
                useful_parts.append(line)
            # Keep lines that look like structured data
            elif line.strip().startswith(('- ', '* ', '1.', '2.', '#')):
                useful_parts.append(line)
        
        if useful_parts:
            response = '\n'.join(useful_parts)
        else:
            response = "## Fashion Analysis\n\nI've analyzed your image and found relevant fashion items. Please see the details below."
    
    # Clean up formatting
    response = response.strip()
    
    # Ensure proper markdown formatting
    if not response.startswith('#'):
        response = "## Fashion Analysis\n\n" + response
    
    # Fix common markdown issues
    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)  # Remove excessive line breaks
    response = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\n\2', response)  # Add breaks after sentences
    
    return response

def get_all_items_for_image(image_url: str, dataset) -> Any:
    """
    Get all items related to a specific image from the dataset.
    
    Args:
        image_url: The URL of the matched image
        dataset: DataFrame containing outfit information
    
    Returns:
        DataFrame: All items related to the image
    """
    try:
        related_items = dataset[dataset['Image URL'] == image_url]
        logger.info(f"Found {len(related_items)} items related to image URL: {image_url}")
        return related_items
    except Exception as e:
        logger.error(f"Error getting items for image: {e}")
        return dataset.iloc[0:0]  # Return empty DataFrame with same structure

def format_alternatives_response(user_response: str, alternatives: Dict[str, List[Dict]], 
                                similarity_score: float, threshold: float = 0.8) -> str:
    """
    Append alternatives to the user response in a formatted way.
    
    Args:
        user_response: Original response from the model
        alternatives: Dictionary of alternatives for each item
        similarity_score: Similarity score of the match
        threshold: Threshold for determining match quality
    
    Returns:
        str: Enhanced response with alternatives
    """
    # Check if user_response is problematic
    if not user_response or any(phrase in user_response for phrase in [
        "I'm not able to provide",
        "I cannot",
        "I apologize, but", 
        "I don't feel comfortable"]):
        # Create a basic response if the model refused
        user_response = "## Fashion Analysis Results\n\nHere are the items detected in your image:"
    
    if similarity_score >= threshold:
        enhanced_response = user_response + "\n\n## Similar Items Found\n\nHere are some similar items we found:\n"
    else:
        enhanced_response = user_response + "\n\n## Similar Items Found\n\nHere are some visually similar items:\n"
    
    # Count items added to ensure we're not exceeding reasonable limits
    items_added = 0
    max_items = 10
    
    for item, alts in alternatives.items():
        enhanced_response += f"\n### {item}:\n"
        if alts:
            for alt in alts[:3]:  # Limit to 3 alternatives per item
                if items_added < max_items:
                    enhanced_response += f"- {alt['title']} for {alt['price']} from {alt['source']} ([Buy it here]({alt['link']}))\n"
                    items_added += 1
        else:
            enhanced_response += "- No alternatives found.\n"
    
    return enhanced_response

def validate_dataset(dataset) -> bool:
    """
    Validate that the dataset has the required columns and structure.
    
    Args:
        dataset: DataFrame to validate
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    required_columns = ['Item Name', 'Price', 'Link', 'Image URL', 'Embedding']
    
    if dataset.empty:
        logger.error("Dataset is empty")
        return False
    
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        logger.error(f"Dataset missing required columns: {missing_columns}")
        return False
    
    # Check if embeddings exist
    if dataset['Embedding'].isna().all():
        logger.error("No embeddings found in dataset")
        return False
    
    logger.info(f"Dataset validation passed. {len(dataset)} items loaded.")
    return True

def format_price(price_str: str) -> str:
    """
    Format price strings consistently.
    
    Args:
        price_str: Raw price string
        
    Returns:
        str: Formatted price string
    """
    if not price_str:
        return "Price not available"
    
    # Remove any extra whitespace
    price_str = str(price_str).strip()
    
    # If it already has a $, just return it
    if price_str.startswith('$'):
        return price_str
    
    # Try to convert to float and format
    try:
        # Remove any non-numeric characters except decimal point
        clean_price = re.sub(r'[^\d.]', '', price_str)
        if clean_price:
            price_float = float(clean_price)
            return f"${price_float:.2f}"
    except ValueError:
        pass
    
    # If all else fails, return original with $ prefix
    return f"${price_str}"

def extract_brand_from_item_name(item_name: str) -> str:
    """
    Extract brand name from item name if possible.
    
    Args:
        item_name: Full item name
        
    Returns:
        str: Extracted brand name or empty string
    """
    if not item_name:
        return ""
    
    # Common brand patterns in fashion item names
    # This is a simple implementation - could be enhanced with a brand database
    words = item_name.split()
    
    # First word is often the brand
    if len(words) > 1:
        first_word = words[0]
        # If first word is capitalized and not a common clothing term
        common_terms = ['black', 'white', 'blue', 'red', 'cotton', 'silk', 'vintage']
        if first_word.lower() not in common_terms and first_word[0].isupper():
            return first_word
    
    return ""

def create_item_summary(item_row) -> str:
    """
    Create a formatted summary of a single item.
    
    Args:
        item_row: DataFrame row containing item information
        
    Returns:
        str: Formatted item summary
    """
    try:
        name = item_row.get('Item Name', 'Unknown Item')
        price = format_price(item_row.get('Price', ''))
        link = item_row.get('Link', '#')
        
        return f"**{name}** - {price} [View Item]({link})"
    except Exception as e:
        logger.error(f"Error creating item summary: {e}")
        return "Item information unavailable"
