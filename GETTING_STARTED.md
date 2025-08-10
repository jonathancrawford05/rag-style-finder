# 🚀 Getting Started with RAG Style Finder

## Quick Setup (5 minutes)

### 1. **Install Ollama** (if not already installed)
```bash
# macOS
brew install ollama

# Linux  
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve
```

### 2. **Install Dependencies**
```bash
# Navigate to project directory
cd rag-style-finder

# Install with Poetry (recommended)
poetry install
poetry shell

# OR install with pip
pip install -r requirements.txt
```

### 3. **Run Setup**
```bash
# Download dataset and install models
python setup.py
```

### 4. **Test Everything**
```bash
# Verify installation
python test.py
```

### 5. **Launch App**
```bash
# Start the application
python main.py

# OR use the launch script
chmod +x launch.sh
./launch.sh
```

App will be available at: **http://localhost:7860**

## 🎯 **What You Can Do**

1. **Upload fashion images** (photos of outfits, clothing items)
2. **Get AI analysis** of style, colors, materials, and fit
3. **Find similar items** with prices and purchase links
4. **Explore fashion trends** through the curated dataset

## 🔧 **Customization**

### Change Models:
```bash
# Install different vision models
ollama pull llama3.2-vision:latest
ollama pull bakllava:latest

# Update config.py
OLLAMA_MODEL = "llama3.2-vision:latest"
```

### Adjust Settings:
Edit `src/rag_style_finder/config.py`:
- `SIMILARITY_THRESHOLD`: How similar items must be (0.0-1.0)
- `TEMPERATURE`: AI creativity level (0.0-1.0)
- `GRADIO_PORT`: Web interface port

## 🔍 **Example Queries**

Try uploading images of:
- ✨ **Complete outfits** (best results)
- 👕 **Individual clothing items**
- 👗 **Formal wear** and accessories
- 👖 **Casual clothing** combinations

## 📱 **Usage Tips**

- **Good lighting** improves analysis quality
- **Clear, unobstructed** clothing items work best
- **High resolution** images give better results
- **Multiple angles** can provide more insights

## 🆘 **Troubleshooting**

### Common Issues:

**Ollama not running:**
```bash
ollama serve
```

**Model not found:**
```bash
ollama pull llava:latest
```

**Memory issues:**
```bash
# Use smaller model
ollama pull llava:7b
```

**Dataset missing:**
```bash
python setup.py  # Re-download
```

### Check Status:
```bash
python test.py  # Run diagnostics
```

## 🏗️ **Architecture Overview**

```
User Image → ResNet50 → Vector Search → LLM Analysis → Formatted Results
     ↓           ↓            ↓             ↓              ↓
   Upload    Features    Similar Items    Ollama      Web Interface
```

## 🌟 **What Makes This Special**

- **100% Open Source** - No API costs or vendor lock-in
- **Privacy First** - All processing happens locally
- **Highly Customizable** - Swap models, adjust parameters
- **Production Ready** - Robust error handling and logging
- **Modern Stack** - Latest PyTorch, Gradio, and LLM tech

## 🎓 **Learning Opportunities**

This project demonstrates:
- **Multimodal AI** (vision + language models)
- **Vector similarity search** with FAISS
- **RAG (Retrieval-Augmented Generation)** patterns
- **Open source LLM integration** with Ollama
- **Computer vision** with PyTorch
- **Web interface development** with Gradio

Perfect for understanding modern AI application development!

---

**Ready to analyze some fashion? 🎨**

Run: `python main.py` and visit `http://localhost:7860`
