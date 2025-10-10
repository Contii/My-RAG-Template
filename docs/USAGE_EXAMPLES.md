# Usage Examples

## Basic Usage Examples

### 1. Simple Question Answering
```bash
python main.py
# Choose mode: 2 (Feed prompt with RAG)
# Question: What is artificial intelligence?
```

### 2. Advanced Filtering
```bash
python main.py
# Choose mode: 3 (RAG with advanced filters)
# Filter by file types: .pdf, .json
# Filter by sources: ai_research.pdf
# Minimum score: 0.7
```

### 3. Performance Monitoring
```bash
python main.py
# During execution, type: metrics
# View real-time performance dashboard
```

## Configuration Examples

### Development Configuration
```yaml
# config/dev_config.yaml
llm_model: "microsoft/bitnet-b1.58-2B-4T"
retriever_type: "smart"
retrieval:
  use_reranking: false    # Faster for development
  use_cache: false        # Always fresh results
  use_filters: true
  use_metrics: true       # Monitor performance
  top_k: 5
```

### Production Configuration
```yaml
# config/prod_config.yaml
llm_model: "microsoft/bitnet-b1.58-2B-4T"
retriever_type: "smart"
retrieval:
  use_reranking: true     # Maximum accuracy
  use_cache: true         # Optimized performance
  use_filters: true
  use_metrics: true
  top_k: 3
  cache_ttl_hours: 168    # 1 week cache
```

## Document Processing Examples

### Adding New Documents
```python
# Add documents to data/documents/
# Supported formats: .txt, .json, .html, .pdf, .docx

# Example JSON document
{
    "title": "AI Ethics Guidelines",
    "content": "Artificial intelligence ethics involves...",
    "metadata": {
        "author": "Dr. Smith",
        "date": "2024-01-15",
        "category": "ethics"
    }
}

# The system will automatically:
# 1. Detect file format
# 2. Extract text and metadata
# 3. Generate embeddings
# 4. Make available for retrieval
```

### Custom Document Metadata
```python
# Use metadata for advanced filtering
filters = {
    'sources': ['ethics_paper.json'],
    'date_from': datetime(2024, 1, 1),
    'date_to': datetime(2024, 12, 31)
}

context, answer = pipeline.run(
    "What are AI ethics principles?",
    filters=filters
)
```

## Best Practices

### Query Optimization
```python
# Good: Specific, focused questions
"What are the main types of machine learning algorithms?"

# Better: Questions that match document content
"Explain supervised learning algorithms with examples"

# Best: Questions with context hints
"What are neural network architectures used in computer vision?"
```

### Filter Usage
```python
# Use filters to improve relevance
filters = {
    'file_types': ['.pdf'],        # Academic papers
    'min_score': 0.6,              # High relevance only
    'sources': ['recent_papers/']   # Latest research
}
```

### Performance Optimization
```python
# For frequent queries, enable caching
config = {
    'retrieval': {
        'use_cache': True,
        'cache_ttl_hours': 24
    }
}

# For accuracy, enable reranking
config = {
    'retrieval': {
        'use_reranking': True,
        'rerank_top_k': 10
    }
}
```