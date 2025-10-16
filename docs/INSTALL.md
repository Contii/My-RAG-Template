# Installation Guide

Guide to install and configure the My-RAG-Template system.

---
## 📋 Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **RAM**: 8GB minimum (16GB+ recommended for large models)
- **Storage**: 10GB+ for models and embeddings
- **GPU** (optional): CUDA-compatible GPU for faster inference
  - NVIDIA GPU with CUDA 11.8+ recommended
  - PyTorch with CUDA support

---
## 🚀 Installation Steps

### 1. Clone Repository

```bash
git clone <repository-url>
cd My-RAG-Template
```

### 2. Create Virtual Environment

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

#### Core Dependencies:
- `torch` - Deep learning framework
- `transformers` - Hugging Face models
- `sentence-transformers` - Embedding models
- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `pyyaml` - Configuration management
- `psutil` - System metrics
- `python-docx`, `pypdf2` - Document parsing
- `beautifulsoup4` - HTML parsing
- `docling` - Advanced PDF parsing

### 4. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability (if GPU)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---
## ⚙️ Configuration

### 1. Configure Models

Edit `config/config.yaml` to set your preferred models:

```yaml
# Default model (change to your preference)
default_model: "bitnet"

# Available models
models:
  bitnet:
    path: "microsoft/bitnet-b1.58-2B-4T"
    quantization: "4bit"
    max_new_tokens: 250
    temperature: 0.7
  
  gemma:
    path: "google/gemma-2b-it"
    quantization: null
    max_new_tokens: 512
    temperature: 0.7
```

### 2. Configure Retriever

```yaml
retriever:
  model_name: "all-MiniLM-L6-v2"
  embeddings_path: "data/embeddings.json"
  top_k: 5
  
  # Reranking settings
  use_reranking: true
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  rerank_top_k: 3
  
  # Cache settings
  cache_enabled: true
  cache_ttl: 3600  # 1 hour
  
  # Metrics
  metrics_enabled: true
```

### 3. Add Documents

Place your documents in the `data/documents/` folder:

```
data/
├── documents/
│   ├── paper1.pdf
│   ├── report.docx
│   ├── article.txt
│   └── notes.md
└── embeddings.json  # Generated automatically
```

**Supported formats:**
- Text: `.txt`, `.md`
- Documents: `.pdf`, `.docx`, `.pptx`
- Web: `.html`, `.htm`
- Data: `.json`

---
## 🏃 First Run

### 1. Generate Embeddings

On first run, the system will automatically:
1. Scan `data/documents/` folder
2. Parse all supported files
3. Generate embeddings
4. Save to `data/embeddings.json`

```bash
python main.py
```

**Output:**
```
Initializing SemanticRetriever with model: all-MiniLM-L6-v2
No existing embeddings found. Processing documents...
Processing documents: 100%|██████| 25/25 [00:15<00:00]
Embeddings saved to data/embeddings.json
```

### 2. Choose Operational Mode

```
=========== My-RAG-Template ===========
1 - Direct question to LLM.   
2 - Feed prompt with RAG.
3 - RAG with advanced filters.
Choose mode: 2
```

**Mode 1**: Direct LLM inference (no retrieval)
**Mode 2**: Full RAG pipeline (recommended)
**Mode 3**: RAG with metadata filtering

### 3. Test Query

```
Question: What is retrieval-augmented generation?

🤖 Using huggingface generator: microsoft/bitnet-b1.58-2B-4T
   Max tokens: 250
   Temperature: 0.7

✨ Generated tokens: 250   
⏱️ Generation time: 12.45s
⚡ Speed: 20.1 tokens/s

❓ Question: What is retrieval-augmented generation?

💬 Answer: Retrieval-Augmented Generation (RAG) is an AI technique...
```

---
## 🔄 Updating

### Update Dependencies

```bash
# Pull latest changes
git pull origin main

# Update packages
pip install -r requirements.txt --upgrade
```

### Regenerate Embeddings

If you add/remove documents:

```bash
# Delete old embeddings
rm data/embeddings.json

# Run system to regenerate
python main.py
```

---
## 💡 Tips

### Performance Optimization

1. **Use GPU**: 10-20x faster inference
2. **Enable Quantization**: 2-4x memory reduction
3. **Enable Caching**: Reduces redundant retrievals
4. **Tune `top_k`**: Lower values = faster retrieval

### Resource Management

```yaml
# For low-memory systems (<8GB RAM)
max_new_tokens: 128
quantization: "4bit"
use_reranking: false

# For high-performance systems (16GB+ RAM, GPU)
max_new_tokens: 512
quantization: null
use_reranking: true
```

---
## 12. Documentation and Support

### Further Reading
- [System Architecture](./ARCHITECTURE.md) - Technical implementation details
- [Usage Examples](./USAGE_EXAMPLES.md) - Code samples and integration patterns
- [HuggingFace Models Configuration](./HUGGINGFACE_CONFIGS.md) - Detailed HuggingFace configuration parameters

- [Hugging Face CLI Documentation](https://huggingface.co/docs/huggingface_hub/cli)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Transformers Library](https://github.com/huggingface/transformers)

### Community Resources
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [PyTorch Community](https://pytorch.org/community/)
- [RAG Research Papers](https://huggingface.co/docs/transformers/model_doc/rag)

For model-specific questions and best practices, consult the official documentation of the respective libraries and models.