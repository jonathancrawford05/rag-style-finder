"""
Main application for the RAG Style Finder.
Combines computer vision, vector similarity, and LLMs for fashion analysis.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
import gradio as gr

# Import our modules
from .config import *
from .image_processor import ImageProcessor
from .llm_service import OllamaVisionService
from .helpers import (
    process_response, 
    get_all_items_for_image, 
    validate_dataset,
    create_item_summary
)

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StyleFinderApp:
    """Main application class for the Style Finder."""
    
    def __init__(self, dataset_path: str = DATASET_PATH):
        """
        Initialize the Style Finder application.
        
        Args:
            dataset_path: Path to the dataset file
        """
        logger.info("Initializing Style Finder application...")
        
        # Load the dataset
        self.dataset_path = dataset_path
        self.data = self._load_dataset()
        
        # Initialize components
        self.image_processor = ImageProcessor(
            image_size=IMAGE_SIZE,
            norm_mean=NORMALIZATION_MEAN,
            norm_std=NORMALIZATION_STD
        )
        
        self.llm_service = OllamaVisionService(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS
        )
        
        logger.info("Style Finder application initialized successfully!")
    
    def _load_dataset(self):
        """Load and validate the fashion dataset."""
        logger.info(f"Attempting to load dataset from: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset file not found: {self.dataset_path}")
            logger.info("You need to download the dataset first!")
            logger.info("Run: python download_dataset.py")
            # Create a minimal empty dataset structure
            empty_df = pd.DataFrame(columns=['Item Name', 'Price', 'Link', 'Image URL', 'Embedding'])
            logger.warning("Created empty dataset - app will not work until real dataset is loaded")
            return empty_df
        
        try:
            logger.info("Loading dataset file...")
            data = pd.read_pickle(self.dataset_path)
            logger.info(f"Loaded {len(data)} rows from dataset")
            
            if not validate_dataset(data):
                raise ValueError("Dataset validation failed")
            
            # Additional embedding validation
            embeddings = data['Embedding'].dropna()
            logger.info(f"Found {len(embeddings)} valid embeddings")
            
            if len(embeddings) > 0:
                first_emb = embeddings.iloc[0]
                logger.info(f"First embedding type: {type(first_emb)}")
                try:
                    emb_array = np.array(first_emb)
                    logger.info(f"First embedding shape: {emb_array.shape}")
                except Exception as e:
                    logger.error(f"Cannot convert first embedding to array: {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Return empty dataset to allow app to start
            empty_df = pd.DataFrame(columns=['Item Name', 'Price', 'Link', 'Image URL', 'Embedding'])
            return empty_df
    
    def process_image(self, image) -> str:
        """
        Process a user-uploaded image and generate a fashion response.
        
        Args:
            image: PIL image uploaded through Gradio
        
        Returns:
            str: Formatted response with fashion analysis
        """
        if image is None:
            return "Please upload an image to analyze."
        
        if self.data.empty:
            return """
            ⚠️ **Dataset not loaded!**
            
            The fashion dataset is missing. To fix this:
            
            1. **Download the dataset:** Run `python download_dataset.py` 
            2. **Restart the app:** The app needs `swift-style-embeddings.pkl`
            3. **Check the logs** for more details
            
            Without the dataset, the app cannot find similar fashion items.
            """
        
        temp_file = None
        try:
            # Save the image temporarily if it's not already a file path
            if not isinstance(image, str):
                temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
                image_path = temp_file.name
                image.save(image_path, format='JPEG', quality=95)
                temp_file.close()
            else:
                image_path = image
            
            # Step 1: Encode the image
            logger.info("Encoding user image...")
            user_encoding = self.image_processor.encode_image(image_path, is_url=False)
            if user_encoding['vector'] is None:
                return "❌ Error: Unable to process the image. Please try another image."
            
            # Step 2: Find the closest match
            logger.info("Finding closest match in dataset...")
            closest_row, similarity_score = self.image_processor.find_closest_match(
                user_encoding['vector'], self.data
            )
            if closest_row is None:
                return "❌ Error: Unable to find a match. Please try another image."
            
            logger.info(f"Found match: {closest_row['Item Name']} (similarity: {similarity_score:.3f})")
            
            # Step 3: Get all related items
            all_items = get_all_items_for_image(closest_row['Image URL'], self.data)
            if all_items.empty:
                return "❌ Error: No items found for the matched image."
            
            # Step 4: Generate fashion response
            logger.info("Generating fashion analysis...")
            bot_response = self.llm_service.generate_fashion_response(
                user_image_base64=user_encoding['base64'],
                matched_row=closest_row,
                all_items=all_items,
                similarity_score=similarity_score,
                threshold=SIMILARITY_THRESHOLD
            )
            
            return process_response(bot_response)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"❌ Error processing image: {str(e)}"
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

def create_gradio_interface(app: StyleFinderApp) -> gr.Blocks:
    """
    Create and configure the Gradio interface.
    
    Args:
        app: Instance of the StyleFinderApp
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(theme=GRADIO_THEME, title=APP_TITLE) as demo:
        
        # Header
        gr.Markdown(f"""
        # {APP_TITLE}
        
        {APP_DESCRIPTION}
        
        **Powered by:** Open Source AI (Ollama + {OLLAMA_MODEL})
        """)
        
        # Model status check
        models = app.llm_service.list_available_models()
        if OLLAMA_MODEL not in models:
            gr.Markdown(f"""
            ⚠️ **Setup Required:** Model `{OLLAMA_MODEL}` not found in Ollama.
            
            To install it, run: `ollama pull {OLLAMA_MODEL}`
            
            Available models: {', '.join(models) if models else 'None'}
            """)
        
        # Example images section
        gr.Markdown("### 📸 Example Images")
        gr.Markdown("Click the buttons below to load example images for testing:")
        
        # Dynamically find all example images
        example_images = []
        example_buttons = []
        
        # Check for example images
        for i in range(1, 7):  # Check for test-1.png through test-6.png
            img_path = f"examples/test-{i}.png"
            if os.path.exists(img_path):
                example_images.append((i, img_path))
        
        # Display example images in rows of 3
        for row_start in range(0, len(example_images), 3):
            with gr.Row():
                for i in range(row_start, min(row_start + 3, len(example_images))):
                    idx, img_path = example_images[i]
                    with gr.Column():
                        gr.Image(
                            value=img_path,
                            label=f"Example {idx}", 
                            show_label=True, 
                            interactive=False,
                            height=200
                        )
                        btn = gr.Button(f"Use Example {idx}", variant="secondary")
                        example_buttons.append((btn, img_path, idx))
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                # Image input
                image_input = gr.Image(
                    type="pil",
                    label="📤 Upload Fashion Image",
                    height=400
                )
                
                # Submit button
                submit_btn = gr.Button("🔍 Analyze Style", variant="primary", size="lg")
                
                # Status indicator
                status = gr.Markdown("**Status:** Ready to analyze images")
                
                # Dataset info
                if not app.data.empty:
                    valid_embeddings = app.data['Embedding'].dropna()
                    gr.Markdown(f"""
                    **Dataset Info:**
                    - {len(app.data)} fashion items loaded
                    - {len(valid_embeddings)} valid embeddings
                    - {app.data['Image URL'].nunique()} unique outfits
                    """)
                else:
                    gr.Markdown("""
                    ⚠️ **Dataset not loaded** 
                    
                    **To fix this:**
                    1. Run: `python download_dataset.py`
                    2. Or download manually from the course materials
                    3. Restart the app
                    
                    The app needs the `swift-style-embeddings.pkl` file.
                    """)
            
            with gr.Column(scale=2):
                # Output
                output = gr.Markdown(
                    label="🎯 Style Analysis Results",
                    value="Upload an image and click 'Analyze Style' to get started!",
                    height=600
                )
        
        # Event handlers
        def update_status(message):
            return f"**Status:** {message}"
        
        # Submit button workflow
        submit_btn.click(
            fn=lambda: update_status("🔄 Analyzing image... This may take a few moments."),
            outputs=status
        ).then(
            fn=app.process_image,
            inputs=[image_input],
            outputs=output
        ).then(
            fn=lambda: update_status("✅ Analysis complete!"),
            outputs=status
        )
        
        # Dynamic example button handlers
        for btn, img_path, idx in example_buttons:
            def create_click_handler(path, example_idx):
                def click_handler():
                    return gr.Image(value=path)
                def status_handler():
                    return update_status(f"📷 Example {example_idx} loaded. Click 'Analyze Style' to process.")
                return click_handler, status_handler
            
            click_fn, status_fn = create_click_handler(img_path, idx)
            btn.click(
                fn=click_fn,
                outputs=image_input
            ).then(
                fn=status_fn,
                outputs=status
            )
        
        # Footer information
        gr.Markdown("""
        ### 🔧 How It Works
        
        This application uses state-of-the-art AI to analyze fashion images:
        
        1. **Image Encoding**: Converts your image into numerical vectors using ResNet50
        2. **Similarity Matching**: Finds visually similar items using cosine similarity
        3. **AI Analysis**: Generates detailed descriptions using Ollama's multimodal models
        
        ### 💡 Tips for Best Results
        
        - Use clear, well-lit fashion photos
        - Ensure clothing items are clearly visible
        - Try different angles and outfits
        
        ### 🛠️ Technical Stack
        
        - **Computer Vision**: PyTorch + ResNet50
        - **Vector Search**: FAISS + Scikit-learn
        - **Language Model**: Ollama ({OLLAMA_MODEL})
        - **Interface**: Gradio
        """)
    
    return demo

def main():
    """Main function to run the application."""
    # Initialize the application
    app = StyleFinderApp()
    
    # Create Gradio interface
    demo = create_gradio_interface(app)
    
    # Launch the application
    demo.launch(
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        server_name="0.0.0.0"  # Allow external connections
    )

if __name__ == "__main__":
    main()
