# Usage Examples

Practical examples demonstrating core system features and configurations.

---
## 🚀 CLI Usage

### 1. Standard RAG Query

```bash
python main.py
```

```
=========== My-RAG-Template ===========
1 - Direct question to LLM.   
2 - Feed prompt with RAG.
3 - RAG with advanced filters.
Choose mode: 2

Question: What is retrieval-augmented generation?

🤖 Using huggingface generator: microsoft/bitnet-b1.58-2B-4T
   🎯 Max tokens: 250
   🌡️  Temperature: 0.7

✨ Generated tokens: 250   
⏱️ Generation time: 12.45s
⚡ Speed: 20.1 tokens/s

💡 PERFORMANCE INSIGHTS:
   ✅ Fast retrieval (<0.5s)
   ℹ️  Moderate generation speed (12.5s avg)

❓ Question: What is retrieval-augmented generation?

💬 Answer: Retrieval-Augmented Generation (RAG) is an AI framework that 
combines information retrieval with text generation. The system first 
searches a knowledge base for relevant documents, then uses those documents 
as context to generate accurate, factually grounded responses...

========================================
```

---
### 2. Direct LLM (No Retrieval)

```bash
Choose mode: 1

Question: What is the capital of France?

💬 Answer: The capital of France is Paris, located in the north-central 
part of the country along the Seine River. Paris is known for its art, 
fashion, gastronomy, and culture...
```

**When to use:**
- General knowledge questions
- Math calculations
- Creative writing
- No need for specific document context

---
### 3. Filtered RAG

```bash
Choose mode: 3

Question: What are the latest research findings on neural networks?

📁 Available file types: .pdf, .docx, .txt, .md
Enter file types (comma-separated, or press Enter for all): .pdf

📄 Available source files:
  1. neural_networks_2024.pdf
  2. deep_learning_paper.pdf
  3. transformer_research.pdf
  4. meeting_notes.docx
  5. summary.txt
Enter file numbers (comma-separated, or press Enter for all): 1,2,3

💬 Answer: Based on the 2024 research papers, recent findings on neural 
networks show significant advances in efficiency and interpretability...

📊 Retrieved 3 documents from 3 sources
```

**When to use:**
- Query specific document types (only PDFs, only reports)
- Focus on particular files
- Filter by date ranges
- Exclude irrelevant sources

---
## 📊 Special Commands

### View Metrics Dashboard

```bash
Question: metrics
```

```
============================================================
              UNIFIED RAG METRICS DASHBOARD
============================================================
Session Start: 2025-10-15 14:30:00
============================================================

🖥️  SYSTEM RESOURCES:
   CPU: Avg 45.2%, Max 78.5%
   RAM: Avg 4.23GB / 16.00GB total
   GPU: NVIDIA RTX 3080
        Memory: Avg 65.4%, Max 92.3%
        Utilization: Avg 78.2%

🔍 RETRIEVAL:
   Total Queries: 15
   Avg Response Time: 0.234s
   Avg Results/Query: 5.0
   Component Times:
      - Semantic Search: 0.180s
      - Reranking: 0.054s

💾 CACHE:
   Total Requests: 15
   Hit Rate: 40.0%
   Hits: 6, Misses: 9
   Time Saved: 1.42s (avg 0.16s per hit)

🤖 GENERATOR:
   Total Generations: 15
   Successful: 15
   Failed: 0
   Avg Duration: 12.45s
   Avg Output Length: 487 characters
   Avg Speed: 8.2 tokens/s

============================================================
```

---
### Switch Models at Runtime

```bash
Question: switch

Available models:
  1. bitnet (current)
  2. gemma
  3. llama
Choose model number: 2

✅ Switched to model: gemma
🤖 Model: google/gemma-2b-it
   Quantization: None
   Max tokens: 512
   Temperature: 0.7

Model loaded successfully!
```

**Benefits:**
- Compare model outputs without restarting
- Test different configurations
- Optimize for speed vs quality
- No need to restart application

---
## 🎯 Programmatic Usage

### Basic Integration

```python
from pipeline.rag_pipeline import RAGPipeline

# Initialize pipeline (loads config automatically)
pipeline = RAGPipeline()

# Run a query
context, answer = pipeline.run("What is machine learning?")

# Display results
print(f"Answer: {answer}")
print(f"\nSources used:")
for doc in context:
    print(f"  - {doc['source']} (score: {doc['score']:.3f})")
```

**Output:**
```
Answer: Machine learning is a subset of artificial intelligence...

Sources used:
  - ml_intro.pdf (score: 0.892)
  - ai_handbook.docx (score: 0.845)
  - research_paper.pdf (score: 0.778)
```

---
### Query with Filters

```python
from datetime import datetime, timedelta

# Define filters
filters = {
    'file_types': ['.pdf', '.docx'],           # Only PDF and Word docs
    'sources': ['important.pdf', 'key.docx'],  # Specific files
    'min_score': 0.7,                          # High relevance only
    'date_from': datetime(2024, 1, 1),         # From Jan 1, 2024
    'date_to': datetime.now()                  # Until now
}

# Run filtered query
context, answer = pipeline.run(
    "What were the key decisions made this year?",
    filters=filters
)

print(f"Answer: {answer}")
print(f"Filtered to {len(context)} relevant documents")
```

---
### Process Multiple Queries

```python
from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

questions = [
    "What is deep learning?",
    "Explain transformer architecture",
    "What is RAG?",
    "How do attention mechanisms work?"
]

results = []
for question in questions:
    context, answer = pipeline.run(question)
    results.append({
        'question': question,
        'answer': answer,
        'num_sources': len(context)
    })

# Display summary
for result in results:
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer'][:100]}...")
    print(f"Sources: {result['num_sources']}")
```

---
### Switch Models Programmatically

```python
from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Compare models
models = ['bitnet', 'gemma']
question = "Explain neural networks in simple terms"

for model_name in models:
    # Switch model
    pipeline.switch_model(model_name)
    
    # Run query
    _, answer = pipeline.run(question)
    
    # Display
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(answer[:200])
```

---
## 🔧 Configuration Examples

### Speed-Optimized (Production)

```yaml
# config/config.yaml
default_model: "bitnet"

models:
  bitnet:
    path: "microsoft/bitnet-b1.58-2B-4T"
    quantization: "4bit"        # Faster inference
    max_new_tokens: 128         # Shorter responses
    temperature: 0.5            # More deterministic

retriever:
  model_name: "all-MiniLM-L6-v2"
  top_k: 3                      # Fewer candidates
  use_reranking: false          # Skip reranking step
  cache_enabled: true           # Enable caching
  cache_ttl: 7200              # 2 hours
```

**Results:**
- Generation: ~40s (CPU) or ~4s (GPU)
- Retrieval: ~100ms
- Total: ~40-45s per query

---
### Quality-Optimized (Research)

```yaml
default_model: "gemma"

models:
  gemma:
    path: "google/gemma-2b-it"
    quantization: null          # No quantization (better quality)
    max_new_tokens: 512         # Longer, detailed responses
    temperature: 0.7            # More creative

retriever:
  model_name: "all-MiniLM-L6-v2"
  top_k: 10                     # More candidates
  use_reranking: true           # Enable reranking
  rerank_top_k: 5              # Return top 5 after reranking
  cache_enabled: true
```

**Results:**
- Generation: ~120s (CPU) or ~12s (GPU)
- Retrieval: ~300ms
- Total: ~120-125s per query

---
### Low-Memory (<8GB RAM)

```yaml
default_model: "bitnet"

models:
  bitnet:
    quantization: "4bit"        # Essential for low memory
    max_new_tokens: 128
    temperature: 0.7

retriever:
  top_k: 3
  use_reranking: false          # Saves memory
  cache_enabled: true
```

**Memory usage:** ~2-3GB total

---
## 📁 Document Management

### Adding New Documents

```bash
# 1. Copy documents to folder
cp my_document.pdf data/documents/
cp research_paper.docx data/documents/

# 2. Delete old embeddings (will regenerate automatically)
rm data/embeddings.json

# 3. Run system (embeddings auto-generated on first query)
python main.py
```

**Automatic processing:**
```
Initializing SemanticRetriever...
No existing embeddings found. Processing documents...
Processing documents: 100%|████████| 27/27 [00:18<00:00, 1.48it/s]
✅ Embeddings saved to data/embeddings.json
Successfully processed 27 documents
```

---
### Supported Document Formats

```
✅ Text files:        .txt, .md
✅ Documents:         .pdf, .docx, .pptx
✅ Web content:       .html, .htm
✅ Data:              .json
```

**Example structure:**
```
data/documents/
├── research_paper.pdf       ✅ Parsed with docling
├── meeting_notes.docx       ✅ Parsed with python-docx
├── presentation.pptx        ✅ Parsed with python-pptx
├── summary.txt              ✅ Parsed as plain text
├── technical_spec.md        ✅ Parsed as markdown
├── report.html              ✅ Parsed with BeautifulSoup
└── data_export.json         ✅ Parsed as JSON
```

---
## 💡 Best Practices

### When to Use Each Mode

| Mode | Best For | Example Question |
|------|----------|------------------|
| **1. Direct** | General knowledge, math, creative tasks | "What is 15% of 200?" |
| **2. RAG** | Document-based queries, specific info | "What does our Q3 report say about revenue?" |
| **3. Filtered** | Specific document subsets | "What are the latest findings in research PDFs?" |

---
### Model Selection Guide

| Priority | Model | Speed (CPU) | Memory | Quality |
|----------|-------|-------------|--------|---------|
| **Speed** | Bitnet 4-bit | ~40s | 2GB | Good |
| **Balance** | Gemma 4-bit | ~80s | 4GB | Better |
| **Quality** | Gemma no quant | ~120s | 8GB | Best |

---
### Cache Strategy

```yaml
# Frequently updated documents (news, reports)
cache_ttl: 900  # 15 minutes

# Static reference documents (manuals, research)
cache_ttl: 86400  # 24 hours

# Development/testing
cache_enabled: false  # Disable for fresh results
```

---
## 📚 Documentation

- [Installation Guide](docs/INSTALL.md) - Detailed setup instructions
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [HuggingFace Models Configuration](docs/HUGGINGFACE_CONFIGS.md) - Detailed HuggingFace configuration parameters
---