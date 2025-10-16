# My-RAG Template

Laboratory for developing Retrieval-Augmented Generation (RAG) systems with lightweight LLMs, including semantic search, cross-encoder reranking, caching, metadata filtering and metrics tracking. The default template is integrating [Hugging Face](https://huggingface.co/) models with PyTorch.

## 🚀 Features

### Core Capabilities
- **Multi-Model Support**: Switch between different LLM models (Bitnet, Gemma, Llama, etc.) without restarting
- **Smart Retriever**: All-in-one retrieval system combining semantic search, reranking, caching, and filtering
- **Advanced Semantic Search**: FAISS-based vector similarity search with configurable models
- **Cross-Encoder Reranking**: Improved result relevance using cross-encoder models
- **Intelligent Query Caching**: File-based cache with configurable TTL to reduce redundant retrievals
- **Metadata Filtering**: Filter documents by file type, source, date range, and custom metadata
- **Flexible Document Parsing**: Support for TXT, PDF, DOCX, PPTX, HTML, JSON, and Markdown
- **Quantization Support**: 4-bit and 8-bit quantization for efficient model inference

### Metrics & Monitoring
- **Unified Metrics System**: Centralized tracking of all system components
  - System resources (CPU, RAM, GPU)
  - Retrieval performance (queries, latency, results)
  - Cache effectiveness (hit rate, time saved)
  - Generator performance (speed, success rate, token throughput)
- **Real-time Dashboard**: Interactive metrics display with performance insights
- **Persistent Metrics**: JSON-based metric storage for analysis and reporting

### Document Processing
- **Multi-format Support**: TXT, JSON, HTML, PDF, DOCX with automatic format detection
- **Docling Integration**: Advanced PDF parsing with layout and structure preservation
- **Metadata Extraction**: Automatic file type, source, and date detection for filtering
- **Embedding Management**: Efficient storage, loading and caching of vector embeddings

## Applications

- **Advanced Chatbots**: Context-aware conversational AI with retrieval augmentation
- **Semantic Search**: Large-scale corpus exploration with natural language queries
- **Intelligent Recommendation Systems**: Content discovery based on natural language understanding
- **Document Analysis**: Enterprise knowledge base querying and analysis
- **Research Assistant**: Academic paper and technical document exploration

---
## 🔧 System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for large models)
- **Storage**: 10GB+ for models and embeddings
- **GPU**: CUDA-compatible GPU for faster inference

## 📋 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd My-RAG-Template

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

See [INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

Main parameters are configured in [config/config.yaml](config/config.yaml).

#### Run the System
```bash
# Interactive mode with full RAG pipeline
python main.py

# Choose mode:
# 1 - Direct question to LLM (no retrieval)
# 2 - Feed prompt with RAG (semantic search + generation)
# 3 - RAG with advanced filters (metadata + date filtering)
```

---
## 📊 Metrics System

### Unified Dashboard

The system includes a comprehensive metrics tracking system accessible via the `metrics` command:

```
============================================================
              UNIFIED RAG METRICS DASHBOARD
============================================================

🖥️  SYSTEM RESOURCES:
   CPU: Avg 45.2%, Max 78.5%
   RAM: Avg 4.23GB, Max 6.15GB
   GPU: NVIDIA RTX 3080
        Avg 65.4%, Max 92.3%

🔍 RETRIEVAL:
   Total Queries: 15
   Avg Response Time: 0.234s
   Avg Results/Query: 5.0

💾 CACHE:
   Total Requests: 15
   Hit Rate: 40.0%
   Hits: 6, Misses: 9
   Time Saved: 1.42s

🤖 GENERATOR:
   Total Generations: 15
   Successful: 15
   Failed: 0
   Avg Duration: 12.45s
   Avg Speed: 8.2 tokens/s

============================================================
```

### Performance Insights

The system provides intelligent insights based on collected metrics:

```
💡 PERFORMANCE INSIGHTS:
   ✅ Excellent retrieval speed (<0.5s)
   ⚠️  Low cache hit rate - consider increasing TTL
   ✅ High generation success rate (100%)
   ℹ️  Moderate token speed - GPU acceleration recommended
```

---
## 📚 Documentation

- [Installation Guide](docs/INSTALL.md) - Detailed setup instructions
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Usage Examples](docs/USAGE_EXAMPLES.md) - Practical examples and use cases
- [HuggingFace Models Configuration](docs/HUGGINGFACE_CONFIGS.md) - Detailed HuggingFace configuration parameters
---
## 📦 Project Structure

```
My-RAG-Template/
├── config/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── documents/               # Source documents
│   └── embeddings/              # Processed embeddings
├── docs/                        # Documentation
├── generator/
│   ├── base_generator.py        # Generator interface
│   ├── generator.py             # Inference motors implementation
│   └── generator_factory.py     # Multi-model factory
├── logger/
│   └── logger.py                # Centralized logging
├── pipeline/
│   └── rag_pipeline.py          # Main orchestration
├── retriever/
│   ├── smart_retriever.py       # Unified retriever
│   ├── semantic_retriever.py    # Base semantic search
│   ├── faiss_index.py           # Vector index
│   └── [legacy retrievers]      # Backward compatibility
├── utils/
│   ├── metrics/                 # Metrics system
│   │   ├── unified_metrics.py   # Unified collector
│   │   ├── system_metrics.py
│   │   ├── retrieval_metrics.py
│   │   ├── cache_metrics.py
│   │   └── generator_metrics.py
│   ├── document_parsers.py      # Document parsing
│   └── query_cache.py           # Query caching
├── main.py                      # CLI entry point
└── requirements.txt
```
See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

---
## References

- [Hugging Face](https://huggingface.co/) - Transformer models and tokenizers
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Retrieval-Augmented Generation (RAG)](https://huggingface.co/docs/transformers/model_doc/rag) - RAG architecture
- [FAISS](https://faiss.ai/index.html) - In memory tool for embeddings similarity
- [Sentence Transformers](https://www.sbert.net/) - Semantic search embeddings
- [Docling](https://github.com/DS4SD/docling) - Advanced document parsing
- [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/) - HTML text processing

---
## 📄 License

© 2025 **Joao Conti**. This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
