# Installation and Usage Guide

## Prerequisites

- Python 3.8+ (recommended: Python 3.11)
- CUDA-compatible GPU (recommended for optimal performance)
- Hugging Face account (for model access)

---
## 1. Create Virtual Environment

```bash
py -3.11 -m venv venv
venv\Scripts\Activate.ps1   # Windows PowerShell
# or
venv\Scripts\activate.bat   # Windows Command Prompt
# or
source venv/bin/activate    # Linux/macOS
```

---
## 2. Install Core Dependencies

### Essential Dependencies
```bash
pip install -U "huggingface_hub[cli]"
pip install git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4
pip install accelerate
```

### RAG System Dependencies
```bash
# Core ML libraries
pip install torch torchvision  # See step 4 for CUDA-specific installation
pip install sentence-transformers>=2.2.0
pip install scikit-learn>=1.3.0
pip install numpy>=1.24.0

# Document processing
pip install docling>=1.0.0
pip install beautifulsoup4>=4.12.0
pip install python-docx>=0.8.11

# System utilities
pip install pyyaml>=6.0
pip install psutil>=5.9.0
pip install tqdm>=4.65.0

# Optional: Web interface
pip install flask>=2.3.0
pip install requests>=2.31.0
```

### Alternative: Requirements File
```bash
pip install -r requirements.txt
```

---
## 3. Hugging Face Authentication

```bash
huggingface-cli login
```
When prompted, paste your Hugging Face token. **Do not add this token as a credential in the project repository.**
**Note**: Some models require acceptance of license terms on the Hugging Face website before download.

---
## 4. Install PyTorch with CUDA Support

Visit [pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select the appropriate version for your system.

### Example for CUDA 12.8:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Example for CPU-only installation:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---
## 5. Verify PyTorch Installation

Create a test file `test_pytorch.py`:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

Run the test:
```bash
python test_pytorch.py
```

---
## 6. Test Model Download and Loading

Create a test file `test_model.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test with the default model
model_name = "microsoft/bitnet-b1.58-2B-4T"

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model for {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",  # Automatic device placement
    torch_dtype=torch.float16  # Use half precision for memory efficiency
)

print("Model loaded successfully!")

# Simple test generation
input_text = "The future of artificial intelligence is"
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
    
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")
```

Execute using the virtual environment interpreter:
```bash
python test_model.py
```

---
## 7. Initialize RAG System

### Prepare Document Directory
```bash
# Create data directories
mkdir -p data/documents
mkdir -p data/embeddings
mkdir -p logs

# Add sample documents to data/documents/
# Supported formats: .txt, .json, .html, .pdf, .docx
```

### Test Basic RAG Functionality
```bash
python main.py
```

Select mode and test basic functionality:
1. Choose mode 2 (RAG mode)
2. Ask a question about your documents
3. Verify context retrieval and answer generation

### Test Advanced Features
```bash
python main.py
```

1. Choose mode 3 (Advanced RAG with filters)
2. Test filtering by file type and source
3. During execution, type `metrics` to view performance dashboard
4. Type `exit` to generate metrics report

---
## 8. Configuration Customization

### Basic Configuration (`config/config.yaml`)
```yaml
# Adjust based on your hardware
llm_model: "microsoft/bitnet-b1.58-2B-4T"
max_gpu_memory: "3.6GB"  # Adjust based on your GPU memory (recommended: 90% of max capacity)

# Enable/disable features based on needs
retrieval:
  use_reranking: true      # Better accuracy, slower
  use_cache: true          # Better performance
  use_filters: true        # Advanced filtering
  use_metrics: true        # Performance monitoring
```

### Development vs Production
```yaml
# Development (faster iteration)
retrieval:
  use_reranking: false
  use_cache: false
  top_k: 5

# Production (optimized performance)
retrieval:
  use_reranking: true
  use_cache: true
  cache_ttl_hours: 168  # 1 week
  top_k: 3
```

---
## 9. Troubleshooting

### Common Issues and Solutions

**GPU Memory Issues:**
```yaml
# Reduce GPU memory usage in config.yaml
max_gpu_memory: "2GB"  # Adjust downward

# Or use CPU-only mode
device: "cpu"
```

**Model Download Failures:**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/transformers/
huggingface-cli download microsoft/bitnet-b1.58-2B-4T
```

**Slow Performance:**
```yaml
# Enable caching and reduce reranking candidates
retrieval:
  use_cache: true
  rerank_top_k: 5  # Reduce from default 10
```

**Import Errors:**
```bash
# Verify all dependencies are installed
pip check
pip install --upgrade -r requirements.txt
```

### Performance Optimization

**For Low-Memory Systems:**
```yaml
# Use smaller models
retrieval:
  model_name: "all-MiniLM-L6-v2"  # Smaller embedding model
  reranker_model: "cross-encoder/ms-marco-TinyBERT-L-2-v2"  # Smaller reranker
```

**For High-Performance Systems:**
```yaml
# Use larger, more accurate models
retrieval:
  model_name: "all-mpnet-base-v2"  # Larger embedding model
  reranker_model: "cross-encoder/ms-marco-electra-base"  # Larger reranker
```

---
## 10. System Verification

### Complete System Test
```python
# Create system_test.py
from pipeline.rag_pipeline import RAGPipeline
import time

def test_complete_system():
    print("Testing complete RAG system...")
    
    # Initialize pipeline
    pipeline = RAGPipeline(use_rag=True)
    
    # Test basic retrieval
    start_time = time.time()
    context, answer = pipeline.run("What is artificial intelligence?")
    end_time = time.time()
    
    print(f"✅ Basic RAG test completed in {end_time - start_time:.2f}s")
    print(f"Context retrieved: {len(context)} documents")
    print(f"Answer generated: {len(answer)} characters")
    
    # Test metrics
    if hasattr(pipeline.retriever, 'print_metrics_dashboard'):
        print("✅ Metrics system working")
        pipeline.retriever.print_metrics_dashboard()
    
    # Test filters
    filters = {'file_types': ['.txt', '.json'], 'min_score': 0.5}
    context, answer = pipeline.run("Test question", filters)
    print(f"✅ Filter system working: {len(context)} filtered results")
    
    print("🎉 All systems operational!")

if __name__ == "__main__":
    test_complete_system()
```

Run the complete test:
```bash
python system_test.py
```

---
## 11. Expansion Notes

### Adding New LLMs
1. Update `config.yaml` with new model name
2. Test model compatibility with `test_model.py`
3. Adjust generation parameters as needed
4. Update memory settings based on model size

### Integrating New Data Sources
1. Create custom parsers in `utils/document_parsers.py`
2. Add new document types to supported formats
3. Extend metadata extraction for new fields
4. Update filtering logic for new metadata types

### Custom Components
1. Extend `SmartRetriever` class for new functionality
2. Add new metrics to `RetrievalMetrics` class
3. Create custom caching strategies in `QueryCache`
4. Implement new retrieval algorithms

### Production Deployment
1. Configure logging for production environment
2. Set up external caching (Redis/Memcached)
3. Implement API endpoints using Flask/FastAPI
4. Add monitoring and alerting systems

## 12. Documentation and Support

### Further Reading
- [System Architecture](./ARCHITECTURE.md) - Technical implementation details
- [Usage Examples](./USAGE_EXAMPLES.md) - Code samples and integration patterns
- [Hugging Face CLI Documentation](https://huggingface.co/docs/huggingface_hub/cli)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Transformers Library](https://github.com/huggingface/transformers)

### Community Resources
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [PyTorch Community](https://pytorch.org/community/)
- [RAG Research Papers](https://huggingface.co/docs/transformers/model_doc/rag)

For model-specific questions and best practices, consult the official documentation of the respective libraries and models.