# My-RAG Template

Laboratory for developing Retrieval-Augmented Generation (RAG) systems with lightweight LLMs, integrating [Hugging Face](https://huggingface.co/) models and PyTorch.

## 🚀 Features

### Smart Retrieval System
- **Modular Architecture**: Enable/disable components independently (reranking, caching, filtering, metrics)
- **Semantic Search**: Using sentence-transformers embeddings with configurable models
- **Advanced Filtering**: By file type, source, score threshold, and date range
- **Intelligent Caching**: TTL-based query caching with smart cache key generation
- **Cross-Encoder Reranking**: Improved relevance scoring with configurable reranker models

### Performance Monitoring & Analytics
- **Real-time Metrics**: CPU, RAM, and GPU monitoring during operations
- **Component Timing**: Detailed breakdown of retrieval pipeline performance
- **Interactive Dashboard**: Live metrics display with performance insights and recommendations
- **Structured Logging**: JSON metrics alongside traditional log files for analysis

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

## Installation

Consult the [detailed installation instructions](./docs/INSTALL.md) for initial environment setup and configuration.

## Configuration

Main parameters are configured in `config/config.yaml`.

### Basic Configuration Example:
```yaml
# Core settings
llm_model: "microsoft/bitnet-b1.58-2B-4T"
retriever_type: "smart"  # smart, semantic, filtered, cache, reranking, stub
generator_type: "llm"    # llm, stub
max_tokens: 250
data_path: "data/documents/"
temperature: 0.7
max_gpu_memory: "3.8GB"

# Smart Retriever Components (enable/disable as needed)
retrieval:
  model_name: "all-MiniLM-L6-v2"
  top_k: 3
  embeddings_path: "data/embeddings"
  
  # Component toggles
  use_reranking: true      # Cross-encoder reranking for better relevance
  use_cache: true          # Query result caching for performance
  use_filters: true        # Advanced metadata filtering
  use_metrics: true        # Detailed performance tracking
  
  # Component configurations
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  rerank_top_k: 10
  cache_ttl_hours: 24
  min_score_threshold: 0.3

# Logging configuration
logging:
  level: INFO
  log_to_file: true
  log_file: "logs/rag.log"
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

### Configuration Profiles

**Development Profile:**
```yaml
retrieval:
  use_reranking: false    # Faster iteration
  use_cache: false        # Always fresh results
  use_metrics: true       # Monitor performance
```

**Production Profile:**
```yaml
retrieval:
  use_reranking: true     # Maximum accuracy
  use_cache: true         # Optimized performance
  use_metrics: true       # Full monitoring
```

## Usage

### Basic Execution
```sh
python main.py
```

Choose from three operational modes:
1. **Direct LLM Mode**: Questions go directly to the language model
2. **RAG Mode**: Questions enhanced with retrieved context from documents
3. **Advanced RAG**: RAG with interactive filtering options

### Interactive Features

**Real-time Metrics Dashboard:**
```
Question: metrics
==================================================
           RETRIEVAL METRICS DASHBOARD
==================================================
📊 Total Queries: 5
⏱️  Avg Response Time: 2.143s
📄 Avg Results/Query: 3.0
💾 Cache Hit Rate: 40.0%

🔧 Component Average Times:
   semantic_search: 1.234s
   reranking: 0.567s
   filtering: 0.023s
   cache_check: 0.001s
==================================================
```

**Advanced Filtering Interface:**
```
=== FILTER OPTIONS ===
Available file types: ['.pdf', '.json', '.html', '.txt']
Available sources: ['research_paper.pdf', 'ai_overview.json']

Use filters? (y/n): y
Filter by file types (comma-separated): .pdf, .json
Filter by sources (comma-separated): research_paper.pdf
Minimum score threshold (0.0-1.0): 0.7
```

## Project Structure

```
My-RAG-Template/
├── config/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── documents/               # Source documents (auto-detected formats)
│   └── embeddings/              # Cached vector embeddings
├── docs/
│   ├── INSTALL.md              # Detailed installation guide
│   ├── ARCHITECTURE.md         # System architecture documentation
│   └── USAGE_EXAMPLES.md       # Code examples and use cases
├── generator/
│   └── generator.py            # LLM text generation logic
├── logger/
│   └── logger.py               # Enhanced logging with system metrics
├── logs/                       # Log files and structured metrics
├── pipeline/
│   └── rag_pipeline.py         # Main RAG orchestration
├── retriever/
│   ├── semantic_retriever.py   # Base semantic search functionality
│   ├── smart_retriever.py      # Advanced modular retrieval system
│   ├── filtered_retriever.py   # Legacy filtered retrieval
│   ├── cache_retriever.py      # Legacy cached retrieval
│   └── reranking_retriever.py  # Legacy reranking retrieval
├── utils/
│   ├── document_parsers.py     # Multi-format document parsing
│   ├── query_cache.py          # Intelligent query caching system
│   └── retrieval_metrics.py    # Comprehensive performance tracking
└── main.py                     # Application entry point
```

## Key Features Expanded

### Modular Architecture
The system implements a component-based design where each feature can be independently enabled or disabled:

- **Base**: Semantic search using sentence-transformers
- **Reranking**: Cross-encoder models for relevance improvement
- **Caching**: TTL-based result caching with intelligent key generation
- **Filtering**: Metadata-based document filtering (type, source, date, score)
- **Metrics**: Comprehensive performance monitoring and analysis

### Performance Optimization
- **Smart Caching**: Automatic query result caching with configurable TTL
- **Component Timing**: Individual pipeline stage performance tracking
- **Memory Management**: Efficient embedding storage and GPU memory monitoring
- **Batch Processing**: Optimized vector operations for better throughput

### Enterprise Ready
- **Structured Logging**: JSON metrics alongside traditional logs
- **Error Handling**: Graceful degradation when components fail
- **Configuration Management**: YAML-based centralized configuration
- **Monitoring**: Real-time performance insights and recommendations

## Advanced Usage

### Programmatic Access
```python
from pipeline.rag_pipeline import RAGPipeline

# Initialize with custom configuration
pipeline = RAGPipeline(use_rag=True, config_path="custom_config.yaml")

# Query with filters
filters = {
    'file_types': ['.pdf', '.json'],
    'min_score': 0.7,
    'sources': ['important_document.pdf']
}

context, answer = pipeline.run("Your question here", filters=filters)

# Access performance metrics
if hasattr(pipeline.retriever, 'print_metrics_dashboard'):
    pipeline.retriever.print_metrics_dashboard()
```

### Custom Component Configuration
```python
from retriever.smart_retriever import SmartRetriever

# Fine-grained component control
retriever = SmartRetriever(
    data_path="data/documents",
    use_reranking=True,
    use_cache=True,
    use_filters=True,
    use_metrics=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    cache_ttl_hours=48,
    min_score_threshold=0.5
)

results = retriever.retrieve(
    "machine learning applications",
    file_types=['.pdf'],
    min_score=0.8
)
```

## Documentation

- **[Installation Guide](./docs/INSTALL.md)**: Step-by-step setup instructions
- **[Architecture Documentation](./docs/ARCHITECTURE.md)**: System design and component details
- **[Usage Examples](./docs/USAGE_EXAMPLES.md)**: Code samples and integration patterns

## Expansion and Customization

- **Modular Design**: Easy integration with different data sources and models
- **Clean Architecture**: Organized codebase designed for professional evolution
- **Extensible Components**: Add custom retrievers, generators, or processing pipelines
- **Configuration Driven**: Minimal code changes for different deployment scenarios
- **Future Ready**: Prepared for multiple databases, model adaptation, and API integration

## Performance and Monitoring

The system provides comprehensive monitoring capabilities:

- **System Metrics**: Real-time CPU, RAM, and GPU usage tracking
- **Component Analysis**: Individual pipeline stage performance measurement
- **Cache Effectiveness**: Hit/miss rates and cache optimization insights
- **Query Analytics**: Response time analysis and performance recommendations
- **Structured Data**: JSON metrics export for external monitoring systems

## References

- [Hugging Face](https://huggingface.co/) - Transformer models and tokenizers
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Retrieval-Augmented Generation (RAG)](https://huggingface.co/docs/transformers/model_doc/rag) - RAG methodology
- [Sentence Transformers](https://www.sbert.net/) - Semantic search embeddings
- [Docling](https://github.com/DS4SD/docling) - Advanced document parsing

## License

This project is licensed under the MIT License - see the LICENSE file for details.