# System Architecture

## Overview

My-RAG-Template implements a modular, extensible architecture for Retrieval-Augmented Generation with monitoring and performance optimization.

## Component Architecture

``` mermaid
graph TB
    A[User Query] --> B[RAG Pipeline]
    B --> C[Smart Retriever]
    C --> D[Semantic Search]
    C --> E[Reranker]
    C --> F[Cache]
    C --> G[Filters]
    C --> H[Metrics]
    B --> I[LLM Generator]
    H --> J[Performance Dashboard]
    F --> K[Query Cache]
    L[Documents] --> D
    M[Config] --> B
```

## Smart Retriever Design

The SmartRetriever implements a pipeline pattern with optional components:

1. **Cache Check** - O(1) lookup for repeated queries
2. **Semantic Search** - Vector similarity search
3. **Filtering** - Metadata-based filtering
4. **Reranking** - Cross-encoder relevance scoring
5. **Cache Save** - Store results for future use

### Component Flow
```python
def retrieve(query, filters):
    # 1. Check cache
    if cached_result := cache.get(query_key):
        return cached_result
    
    # 2. Semantic search
    candidates = semantic_search(query)
    
    # 3. Apply filters
    if filters:
        candidates = apply_filters(candidates, filters)
    
    # 4. Rerank if enabled
    if use_reranking:
        candidates = rerank(query, candidates)
    
    # 5. Cache and return
    cache.set(query_key, results)
    return results[:top_k]
```

## Metrics System

### Two-Layer Monitoring
1. **System Metrics** (logger.py)
   - CPU, RAM, GPU usage
   - Component timing
   - Structured JSON output

2. **Retrieval Metrics** (retrieval_metrics.py)
   - Query analysis
   - Cache performance
   - Result quality tracking

### Data Flow
```
Query → [Start Timer] → Components → [End Timer] → Metrics
                    ↓
               System Logger ← Component Times ← Performance Data
                    ↓
            JSON Metrics File + Traditional Logs
```

## Document Processing Pipeline

### Multi-Format Support
```
Input Documents → Format Detection → Appropriate Parser → Text Extraction → Metadata → Embeddings
```

### Parser Selection
- `.txt` → Direct text reading
- `.json` → JSON structure parsing
- `.html` → HTML tag extraction
- `.pdf` → Docling advanced parsing
- `.docx` → Docling document parsing

## Error Handling

### Graceful Degradation
1. **Component Failure**: Disable component, continue pipeline
2. **Model Loading**: Fallback to simpler models
3. **Cache Miss**: Continue without cache
4. **Parse Error**: Skip document, log warning

### Error Recovery
```python
try:
    results = rerank(candidates)
except Exception as e:
    logger.warning(f"Reranking failed: {e}")
    results = candidates  # Use unranked results
```

## Extensibility Points

### Adding New Components
```python
class CustomRetriever(SmartRetriever):
    def __init__(self, *args, use_custom=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_custom = use_custom
        if use_custom:
            self.custom_component = CustomComponent()
    
    def retrieve(self, query, **kwargs):
        # Standard pipeline
        results = super().retrieve(query, **kwargs)
        
        # Custom processing
        if self.use_custom:
            results = self.custom_component.process(results)
        
        return results
```

### Adding New Metrics
```python
class CustomMetrics(RetrievalMetrics):
    def log_custom_metric(self, query_data, metric_value):
        self.query_data[query_data['id']]['custom_metric'] = metric_value
```

## Future Enhancements

### Planned Features
1. **Vector Databases**: ChromaDB, Pinecone, Weaviate integration
2. **API Layer**: REST/GraphQL endpoints
3. **Distributed Processing**: Multi-node deployment
4. **Advanced Metrics**: MLflow integration
5. **A/B Testing**: Retrieval strategy comparison
