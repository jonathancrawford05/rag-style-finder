# RAG Style Finder - Fashion Analysis Application

A multimodal RAG application that analyzes fashion images using **open source AI tools** instead of IBM Watson.

## 🎯 **What This App Does**

- **Upload fashion images** and get detailed style analysis
- **Find similar items** from a curated dataset of fashion outfits  
- **Get detailed descriptions** of clothing, colors, materials, and style
- **View purchase links** for similar items

## 🔧 **Open Source Tech Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Ollama + Llama Vision | Multimodal image analysis |
| **Image Processing** | PyTorch + ResNet50 | Feature extraction |
| **Vector Search** | FAISS + Scikit-learn | Similarity matching |
| **Interface** | Gradio | Web UI |
| **Dataset** | Fashion embeddings | Style database |

## 🚀 **Quick Start**

### 1. **Install Dependencies**

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

### 2. **Setup Application**

```bash
# Run the setup script to download dataset and configure Ollama
python setup.py
```

This will:
- ✅ Check if Ollama is installed and running
- 📥 Download the fashion dataset (~100MB)
- 🤖 Install the Llama Vision model
- 📸 Create example images

### 3. **Run the Application**

```bash
# Start the Gradio app
python main.py
```

The app will be available at: `http://localhost:7860`

## 📋 **Prerequisites**

### Required:
- **Python 3.11+**
- **Ollama** installed and running
- **8GB+ RAM** (for running vision models)

### Install Ollama:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

### Start Ollama:
```bash
ollama serve
```

## 🎨 **Usage**

1. **Upload Image**: Drag and drop or click to upload a fashion photo
2. **Analyze**: Click "Analyze Style" button
3. **View Results**: Get detailed fashion analysis with similar items
4. **Explore**: Try different images and styles

## 📊 **Dataset**

The application uses a curated dataset of fashion items with:
- **Item descriptions** and prices
- **Purchase links** to retailers
- **Pre-computed embeddings** for fast similarity search
- **Taylor Swift outfit collections** (as demonstrated in the original tutorial)

## 🔧 **Configuration**

Edit `src/rag_style_finder/config.py` to customize:

```python
# Model settings
OLLAMA_MODEL = "llava:latest"  # Or "llama3.2-vision:latest"
TEMPERATURE = 0.2

# Similarity threshold
SIMILARITY_THRESHOLD = 0.8

# UI settings
GRADIO_PORT = 7860
```

## 🛠️ **Available Models**

You can experiment with different vision models:

```bash
# Install different models
ollama pull llava:latest          # Most stable
ollama pull llama3.2-vision:latest # Latest Llama vision
ollama pull bakllava:latest       # Alternative option
```

Update the model in `config.py`:
```python
OLLAMA_MODEL = "llama3.2-vision:latest"
```

## 📁 **Project Structure**

```
rag-style-finder/
├── src/rag_style_finder/
│   ├── app.py              # Main Gradio application
│   ├── config.py           # Configuration settings
│   ├── image_processor.py  # ResNet50 + vector similarity
│   ├── llm_service.py      # Ollama integration
│   └── helpers.py          # Utility functions
├── examples/               # Example images
├── setup.py               # Setup script
├── main.py                # Entry point
└── swift-style-embeddings.pkl  # Dataset (downloaded)
```

## 🔍 **How It Works**

1. **Image Encoding**: ResNet50 converts images to 2048-dimensional vectors
2. **Similarity Search**: FAISS finds closest matches in the dataset using cosine similarity
3. **Context Retrieval**: Related fashion items are retrieved for the matched outfit
4. **AI Analysis**: Ollama's vision model analyzes the image and generates detailed descriptions
5. **Response Formatting**: Results are formatted with item details and purchase links

## 🚨 **Troubleshooting**

### Model Issues:
```bash
# Check available models
ollama list

# Pull vision model if missing
ollama pull llava:latest

# Test model
ollama run llava:latest "Describe this image" --image test.jpg
```

### Memory Issues:
- **Reduce model size**: Use smaller models like `llava:7b`
- **Close other applications**: Vision models need substantial RAM
- **Use CPU mode**: Set `CUDA_VISIBLE_DEVICES=""` if GPU issues

### Dataset Issues:
```bash
# Re-download dataset
rm swift-style-embeddings.pkl
python setup.py
```

## 🆚 **vs IBM Watson Version**

| Feature | IBM Watson | Our Open Source Version |
|---------|------------|--------------------------|
| **Cost** | $$$ API costs | ✅ Free |
| **Privacy** | Cloud processing | ✅ Local processing |
| **Customization** | Limited | ✅ Full control |
| **Offline** | No | ✅ Works offline |
| **Model Choice** | Fixed | ✅ Multiple options |

## 🔮 **Next Steps**

- 🎯 **Add more datasets**: Expand beyond Taylor Swift outfits
- 🔍 **Improve search**: Add semantic text search
- 📱 **Mobile optimization**: Responsive design
- 🛒 **Real-time prices**: API integration with retailers
- 🎨 **Style recommendations**: Personal style learning

## 📝 **License**

MIT License - feel free to use and modify!

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Built with ❤️ using Open Source AI**
