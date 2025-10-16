# HuggingFace LLM Configuration Guide

Detailed reference for configuring HuggingFace models in the RAG pipeline.

> This documentation is not yet complete and may contain unnecessary or inaccurate information.


## Table of Contents

- [Overview](#overview)
- [Configuration Categories](#configuration-categories)
- [Loading Configurations](#loading-configurations)
- [Generation Configurations](#generation-configurations)
- [Quantization Options](#quantization-options)
- [Device Management](#device-management)
- [Model-Specific Examples](#model-specific-examples)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Overview

HuggingFace models can be configured at two stages:

1. **Loading Time**: How the model is loaded into memory (dtype, quantization, device placement)
2. **Generation Time**: How the model generates text (temperature, sampling, penalties)

### Configuration Structure

```yaml
generator:
  active_model: "model_name"
  
  models:
    model_name:
      type: "huggingface"
      model_id: "org/model-name"
      
      # Loading configs
      torch_dtype: "bfloat16"
      device_map: "auto"
      max_gpu_memory: "3.8GB"
      quantization: null
      
      # Generation configs
      max_tokens: 250
      temperature: 0.7
      do_sample: true
```
---

## Quick Reference Tables

### Loading Configurations

Complete reference for model loading parameters:

| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| **torch_dtype** | str/torch.dtype | Weight precision | `bfloat16`, `float16`, `float32` |
| **device_map** | str/dict | GPU/CPU distribution | `auto`, `balanced`, `sequential`, `{0: "20GB"}` |
| **max_memory** | dict | Memory limit per device | `{0: "3.8GB", "cpu": "8GB"}` |
| **quantization_config** | dict | Quantization settings | 8-bit, 4-bit (bitsandbytes) |
| **load_in_8bit** | bool | Enable 8-bit quantization | `True` / `False` |
| **load_in_4bit** | bool | Enable 4-bit quantization | `True` / `False` |
| **low_cpu_mem_usage** | bool | Reduce CPU RAM usage | `True` (default) |
| **trust_remote_code** | bool | Execute custom model code | `True` (for some models) |
| **use_flash_attention_2** | bool | Enable Flash Attention v2 | `True` (if available) |

### Generation Configurations

Complete reference for text generation parameters:

| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|-------|
| **max_new_tokens** | int | Maximum tokens to generate | 250 | 1 - ∞ |
| **temperature** | float | Sampling randomness | 0.7 | 0.0 - 2.0 |
| **top_p** | float | Nucleus sampling threshold | 1.0 | 0.0 - 1.0 |
| **top_k** | int | Top-k sampling threshold | 50 | 0 - ∞ |
| **repetition_penalty** | float | Penalize token repetition | 1.0 | 1.0 - 2.0 |
| **do_sample** | bool | Enable sampling vs greedy | `True` | - |
| **num_beams** | int | Beam search width | 1 | 1 - ∞ |
| **early_stopping** | bool | Stop at EOS in beam search | `False` | - |
| **no_repeat_ngram_size** | int | Prevent n-gram repetition | 0 | 0 - ∞ |
| **length_penalty** | float | Penalize sequence length | 1.0 | 0.0 - 2.0 |
| **pad_token_id** | int | Padding token ID | `eos_token_id` | - |
| **eos_token_id** | int | End-of-sequence token ID | Auto-detected | - |

---

## Configuration Categories

### Universal Configs (100% Support)

These work with **ALL** HuggingFace models:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | int | 250 | Maximum tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature (0.0 = greedy) |
| `do_sample` | bool | False | Enable sampling (vs greedy decoding) |
| `pad_token_id` | int | Auto | Padding token ID |
| `eos_token_id` | int | Auto | End-of-sequence token ID |

### Common Configs (95%+ Support)

Work with most modern models:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `top_k` | int | 50 | Top-k sampling threshold |
| `repetition_penalty` | float | 1.0 | Penalty for token repetition |

### Advanced Configs (Model-Dependent)

May not work with all models:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_beams` | int | 1 | Beam search width |
| `early_stopping` | bool | False | Stop at first EOS in beam search |
| `no_repeat_ngram_size` | int | 0 | Prevent n-gram repetition |
| `length_penalty` | float | 1.0 | Exponential penalty on length |

---

## Loading Configurations

### 1. Data Type (torch_dtype)

Controls precision and memory usage.

#### Options:

| Value | Memory | Speed | Quality | Use Case |
|-------|--------|-------|---------|----------|
| `bfloat16` | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | **Recommended** for modern GPUs (A100, H100, 4090) |
| `float16` | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Good for older GPUs (V100, 3090) |
| `float32` | ⭐ | ⭐ | ⭐⭐⭐ | Full precision (rarely needed) |

#### Configuration:

```yaml
# Best for Ampere+ (RTX 30/40 series, A100)
torch_dtype: "bfloat16"

# Best for older GPUs (V100, RTX 20 series)
torch_dtype: "float16"

# Full precision (debugging only)
torch_dtype: "float32"
```

#### Code Example:

```python
import torch

dtype_map = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32
}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype_map['bfloat16']
)
```

---

### 2. Device Mapping (device_map)

Controls how model layers are distributed across devices.

#### Options:

| Value | Description | Best For |
|-------|-------------|----------|
| `"auto"` | Automatic distribution | Single GPU, sufficient VRAM |
| `"balanced"` | Balance across GPUs | Multi-GPU, equal distribution |
| `"sequential"` | Fill GPUs sequentially | Multi-GPU, unequal sizes |
| `{"layer.0": 0, "layer.1": 1}` | Manual mapping | Custom control |

#### Configuration:

```yaml
# Single GPU (recommended)
device_map: "auto"

# Multi-GPU (balanced)
device_map: "balanced"

# Multi-GPU (sequential)
device_map: "sequential"
```

#### Manual Mapping Example:

```yaml
# Advanced: Manual layer distribution
device_map:
  "model.embed_tokens": 0
  "model.layers.0": 0
  "model.layers.1": 0
  "model.layers.2": 1
  "model.layers.3": 1
  "lm_head": 1
```

---

### 3. Memory Management (max_gpu_memory)

Limits VRAM usage per device.

#### Configuration:

```yaml
# Single GPU with 4GB VRAM
max_gpu_memory: "3.8GB"  # Leave ~200MB for system

# Multi-GPU setup
max_memory:
  0: "7GB"   # GPU 0
  1: "7GB"   # GPU 1
  "cpu": "20GB"  # CPU fallback
```

#### Memory Planning:

| GPU VRAM | Safe Limit | Model Size (FP16) |
|----------|-----------|-------------------|
| 4GB | 3.8GB | Up to 2B params |
| 8GB | 7.5GB | Up to 7B params |
| 12GB | 11GB | Up to 13B params |
| 16GB | 15GB | Up to 13B params + context |
| 24GB | 23GB | Up to 30B params |

---

### 4. Low CPU Memory Usage

Reduces CPU RAM during loading.

```yaml
low_cpu_mem_usage: true  # Default: true
```

**When to disable:**
- Debugging model loading
- Very fast NVMe SSD available
- Plenty of CPU RAM (64GB+)

---

### 5. Trust Remote Code

Allows execution of custom model code.

```yaml
trust_remote_code: false  # Default: false (secure)
```

**Enable for:**
- Models with custom architectures (Phi, Falcon)
- Models requiring special preprocessing

**Security Warning:** Only enable for trusted models!

```yaml
# Example: Phi models require this
microsoft/phi-2:
  trust_remote_code: true
```

---

### 6. Flash Attention 2

Optimized attention implementation (if available).

```yaml
use_flash_attention_2: true
```

**Requirements:**
- `flash-attn` package installed
- GPU with compute capability ≥ 8.0 (Ampere+)
- Model supports Flash Attention

**Speed Improvement:**
- 2-4x faster inference
- 30-50% lower VRAM usage

---

## Generation Configurations

### 1. Max Tokens (max_new_tokens)

Maximum number of tokens to generate.

```yaml
max_tokens: 250
```

**Guidelines:**

| Task | Recommended |
|------|-------------|
| Short answers | 50-100 |
| Paragraph responses | 150-300 |
| Long-form content | 500-2048 |
| Chat messages | 100-200 |

**Trade-offs:**
- Higher = slower generation
- Higher = more VRAM usage (context window)
- Higher = risk of rambling

---

### 2. Temperature

Controls randomness in sampling.

```yaml
temperature: 0.7
```

**Scale:**

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.0 | Deterministic (greedy) | Factual Q&A, code generation |
| 0.1-0.3 | Very focused | Technical documentation |
| 0.4-0.7 | **Balanced** | General chat, RAG answers |
| 0.8-1.0 | Creative | Storytelling, brainstorming |
| 1.0-2.0 | Very random | Creative writing, exploration |

**Formula:** `P(token) = softmax(logits / temperature)`

**Visual Guide:**
```
Temperature = 0.1    Temperature = 0.7    Temperature = 1.5
├─ ████████ (95%)    ├─ ████ (40%)        ├─ ██ (20%)
├─ █ (4%)            ├─ ███ (30%)         ├─ ██ (18%)
└─ (1%)              ├─ ██ (20%)          ├─ ██ (17%)
                     └─ █ (10%)           └─ ██ (45% spread)
```

---

### 3. Top-P Sampling (Nucleus Sampling)

Samples from smallest set of tokens with cumulative probability ≥ p.

```yaml
top_p: 0.9
```

**Scale:**

| Value | Effect | Best For |
|-------|--------|----------|
| 0.1 | Very conservative | Factual responses |
| 0.5 | Focused | Technical writing |
| 0.9 | **Balanced** | General use |
| 0.95 | Diverse | Creative tasks |
| 1.0 | All tokens (no filtering) | Maximum creativity |

**Example:**
```
Tokens: [the: 0.4, a: 0.3, an: 0.2, one: 0.05, some: 0.05]

top_p = 0.9:
├─ Sample from: [the, a, an]  (cumsum = 0.9)
└─ Exclude: [one, some]

top_p = 0.7:
├─ Sample from: [the, a]  (cumsum = 0.7)
└─ Exclude: [an, one, some]
```

**Recommendation:** Use `top_p` **OR** `top_k`, not both.

---

### 4. Top-K Sampling

Samples from top K most probable tokens.

```yaml
top_k: 50
```

**Scale:**

| Value | Effect | Use Case |
|-------|--------|----------|
| 1 | Greedy (deterministic) | Code generation |
| 10 | Very focused | Technical Q&A |
| 50 | **Balanced** | General chat |
| 100 | Diverse | Creative writing |
| 0 | Disabled (use top_p instead) | When using nucleus sampling |

**Visual:**
```
top_k = 5:
├─ Consider: Top 5 tokens only
└─ Ignore: All other tokens

top_k = 50:
├─ Consider: Top 50 tokens
└─ Ignore: Long tail tokens
```

---

### 5. Do Sample

Enables sampling (vs. greedy decoding).

```yaml
do_sample: true
```

**Comparison:**

| Mode | Behavior | Output |
|------|----------|--------|
| `do_sample: false` | Always picks highest probability token | Deterministic, repetitive |
| `do_sample: true` | Samples according to probability distribution | Diverse, natural |

**When to disable:**
- Code generation (want exact syntax)
- Mathematical calculations
- Structured output (JSON, SQL)

---

### 6. Repetition Penalty

Penalizes tokens that have already appeared.

```yaml
repetition_penalty: 1.15
```

**Scale:**

| Value | Effect | Use Case |
|-------|--------|----------|
| 1.0 | No penalty | Creative writing (allows repetition) |
| 1.05-1.1 | Subtle | General use |
| 1.1-1.2 | **Moderate** | Prevent boring repetition |
| 1.2-1.5 | Strong | Fight aggressive repetition |
| 1.5+ | Very strong | Last resort (may degrade quality) |

**Formula:** `score = score / (penalty ^ count)`

**Example:**
```
Without penalty (1.0):
"The cat sat on the mat. The cat was happy. The cat..."

With penalty (1.15):
"The cat sat on the mat. It was content. The feline purred..."
```

**Warning:** Too high can make output incoherent!

---

### 7. No Repeat N-Gram Size

Prevents exact n-gram repetition.

```yaml
no_repeat_ngram_size: 3
```

**Scale:**

| Value | Effect |
|-------|--------|
| 0 | Disabled (allow any repetition) |
| 2 | Prevent 2-word phrase repetition |
| 3 | **Recommended** - Prevent 3-word phrase repetition |
| 4-5 | Prevent longer phrases |

**Example (n=3):**
```
Blocked: "the cat sat on the mat" → "the cat sat..." (blocks "the cat sat")
Allowed: "the cat sat" → "the dog sat" (different middle word)
```

**Use Cases:**
- ✅ Summaries (prevent copy-paste)
- ✅ Q&A (avoid verbatim context repetition)
- ❌ Code (may break syntax)
- ❌ Poetry (intentional repetition)

---

### 8. Beam Search (num_beams)

Explores multiple generation paths.

```yaml
num_beams: 1  # Default: disabled (greedy or sampling)
```

**Scale:**

| Value | Speed | Quality | Memory |
|-------|-------|---------|--------|
| 1 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 2-3 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 4-5 | ⭐ | ⭐⭐⭐ | ⭐ |
| 6+ | ⭐ | ⭐⭐⭐ | ⭐ |

**When to use:**
- ✅ Translation (quality critical)
- ✅ Summarization (need best output)
- ❌ Chat (too slow)
- ❌ Real-time (latency matters)

**Visual:**
```
Greedy (num_beams=1):
Start → Token1 → Token2 → End

Beam Search (num_beams=3):
Start ┬→ Token1a → Token2a → End
      ├→ Token1b → Token2b → End
      └→ Token1c → Token2c → End
      ↓
   Pick best path
```

**Note:** Incompatible with `do_sample=true`

---

### 9. Early Stopping

Stop beam search when all beams finish.

```yaml
early_stopping: true
```

**Only relevant when `num_beams > 1`**

| Value | Behavior |
|-------|----------|
| `false` | Continue until max_tokens |
| `true` | Stop when all beams hit EOS |

---

### 10. Length Penalty

Exponential penalty on sequence length.

```yaml
length_penalty: 1.0
```

**Scale:**

| Value | Effect | Use Case |
|-------|--------|----------|
| < 1.0 | Favor shorter outputs | Concise answers |
| 1.0 | Neutral | Default |
| > 1.0 | Favor longer outputs | Detailed explanations |

**Formula:** `score = score / (length ^ penalty)`

**Example:**
```
length_penalty = 0.8 (shorter):
- "Paris." (3 tokens) → Score: 0.95
- "The capital of France is Paris." (7 tokens) → Score: 0.82

length_penalty = 1.2 (longer):
- "Paris." → Score: 0.85
- "The capital of France is Paris." → Score: 0.93
```

---

## Quantization Options

### Overview

Quantization reduces model precision to save memory and increase speed.

| Method | Memory | Speed | Quality | Hardware |
|--------|--------|-------|---------|----------|
| None (FP16/BF16) | 100% | 100% | 100% | Any |
| 8-bit | 50% | 80-90% | 98% | CUDA 7.0+ |
| 4-bit (NF4) | 25% | 70-80% | 95% | CUDA 7.0+ |

---

### 1. No Quantization (Baseline)

```yaml
quantization: null
torch_dtype: "bfloat16"  # or float16
```

**Pros:**
- ✅ Best quality
- ✅ Fastest inference (no dequantization)

**Cons:**
- ❌ Highest VRAM usage

**Use When:**
- Sufficient VRAM available
- Quality is critical
- Latency must be minimized

---

### 2. 8-bit Quantization

```yaml
quantization: "8bit"
```

**Equivalent to:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)
```

**Stats:**
- Memory: **~50% reduction**
- Speed: **10-20% slower**
- Quality: **~2% degradation**

**Requirements:**
- `bitsandbytes` library
- CUDA-capable GPU

**Example Memory:**
```
Llama-7B:
- FP16: 14GB VRAM
- 8-bit: 7GB VRAM ✅ Fits on 8GB GPU!
```

---

### 3. 4-bit Quantization (NF4)

```yaml
quantization: "4bit"
```

**Equivalent to:**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Stats:**
- Memory: **~75% reduction**
- Speed: **20-30% slower**
- Quality: **~5% degradation**

**Example Memory:**
```
Llama-7B:
- FP16: 14GB VRAM
- 4-bit: 3.5GB VRAM ✅ Fits on 4GB GPU!
```

**Advanced Options:**

```yaml
quantization: "4bit"
quantization_config:
  bnb_4bit_compute_dtype: "bfloat16"  # Compute in BF16
  bnb_4bit_use_double_quant: true     # Extra compression
  bnb_4bit_quant_type: "nf4"          # NormalFloat4 (best quality)
```

---

### Quantization Comparison

```
Model: Llama-2-7B (14GB FP16)

┌─────────────┬──────────┬────────┬─────────┬──────────┐
│ Method      │ VRAM     │ Speed  │ Quality │ Hardware │
├─────────────┼──────────┼────────┼─────────┼──────────┤
│ BF16        │ 14GB     │ 100%   │ 100%    │ Any      │
│ 8-bit       │ 7GB      │ 85%    │ 98%     │ CUDA 7+  │
│ 4-bit (NF4) │ 3.5GB    │ 75%    │ 95%     │ CUDA 7+  │
└─────────────┴──────────┴────────┴─────────┴──────────┘
```

**Recommendation Matrix:**

| VRAM | Best Option |
|------|-------------|
| < 4GB | 4-bit (or smaller model) |
| 4-8GB | 4-bit or 8-bit |
| 8-16GB | 8-bit or FP16 |
| 16GB+ | FP16/BF16 (no quantization) |

---

## Device Management

### Single GPU

```yaml
# Simple: Let PyTorch handle it
device_map: "auto"
max_gpu_memory: "7GB"

# Explicit: Force GPU 0
device_map: 0
```

---

### Multi-GPU (Balanced)

```yaml
# Automatic balancing
device_map: "balanced"
max_memory:
  0: "10GB"
  1: "10GB"
```

**Result:**
```
GPU 0: Layers 0-15
GPU 1: Layers 16-31
```

---

### Multi-GPU (Sequential)

```yaml
# Fill GPU 0, then GPU 1
device_map: "sequential"
max_memory:
  0: "8GB"
  1: "12GB"
```

**Result:**
```
GPU 0: Layers 0-12 (fills 8GB)
GPU 1: Layers 13-31 (uses remaining)
```

---

### CPU Offloading

```yaml
device_map: "auto"
max_memory:
  0: "6GB"      # GPU
  "cpu": "20GB" # CPU fallback
```

**When to use:**
- Model larger than GPU VRAM
- Batch processing (throughput > latency)

**Trade-off:**
- ✅ Fits huge models
- ❌ 10-100x slower for CPU layers

---

### Manual Layer Placement

```yaml
device_map:
  "model.embed_tokens": 0
  "model.layers.0": 0
  "model.layers.1": 0
  # ... layers 2-29 on GPU 0
  "model.layers.30": 1
  "model.layers.31": 1
  "lm_head": 1
```

**Use cases:**
- Debugging bottlenecks
- Optimizing cross-GPU communication
- Special hardware setups

---

## Model-Specific Examples

### 1. BitNet (2B - Efficient)

```yaml
bitnet:
  type: "huggingface"
  model_id: "microsoft/bitnet-b1.58-2B-4T"
  
  # Loading (optimized for 4GB GPU)
  torch_dtype: "bfloat16"
  device_map: "auto"
  max_gpu_memory: "3.8GB"
  quantization: null  # Already efficient
  
  # Generation (balanced)
  max_tokens: 250
  temperature: 0.7
  do_sample: true
  top_p: 0.9
  repetition_penalty: 1.0
```

**Stats:**
- VRAM: ~2GB
- Speed: ~20 tokens/sec (RTX 4090)
- Quality: Good for size

---

### 2. Gemma-2 9B (Quality)

```yaml
gemma3n:
  type: "huggingface"
  model_id: "google/gemma-2-9b-it"
  
  # Loading (8-bit for 8GB GPU)
  torch_dtype: "float16"
  device_map: "balanced"
  max_gpu_memory: "7GB"
  quantization: "8bit"
  
  # Generation (creative)
  max_tokens: 512
  temperature: 0.8
  do_sample: true
  top_p: 0.95
  repetition_penalty: 1.1
  no_repeat_ngram_size: 3
```

**Stats:**
- VRAM: ~7GB (8-bit)
- Speed: ~15 tokens/sec
- Quality: Excellent

---

### 3. Llama-2 7B Chat (Popular)

```yaml
llama7b:
  type: "huggingface"
  model_id: "meta-llama/Llama-2-7b-chat-hf"
  
  # Loading (4-bit for 4GB GPU)
  torch_dtype: "float16"
  device_map: "auto"
  max_gpu_memory: "3.8GB"
  quantization: "4bit"
  
  # Generation (chat optimized)
  max_tokens: 512
  temperature: 0.7
  do_sample: true
  top_p: 0.9
  top_k: 40
  repetition_penalty: 1.1
```

**Stats:**
- VRAM: ~3.5GB (4-bit)
- Speed: ~10 tokens/sec
- Quality: Very good

---

### 4. Mistral 7B Instruct (Efficient + Quality)

```yaml
mistral7b:
  type: "huggingface"
  model_id: "mistralai/Mistral-7B-Instruct-v0.2"
  
  # Loading (8-bit sweet spot)
  torch_dtype: "bfloat16"
  device_map: "auto"
  max_gpu_memory: "7GB"
  quantization: "8bit"
  
  # Generation (instruction following)
  max_tokens: 1024
  temperature: 0.6
  do_sample: true
  top_p: 0.9
  repetition_penalty: 1.15
  no_repeat_ngram_size: 3
```

**Stats:**
- VRAM: ~7GB
- Speed: ~18 tokens/sec
- Quality: Excellent for instructions

---

### 5. Phi-2 (2.7B - Code/Reasoning)

```yaml
phi2:
  type: "huggingface"
  model_id: "microsoft/phi-2"
  
  # Loading (requires trust_remote_code)
  torch_dtype: "float16"
  device_map: "auto"
  max_gpu_memory: "4GB"
  quantization: null
  trust_remote_code: true  # Required for Phi!
  
  # Generation (precise for code)
  max_tokens: 512
  temperature: 0.3  # Low for code
  do_sample: true
  top_p: 0.9
  repetition_penalty: 1.05
```

**Stats:**
- VRAM: ~3GB
- Speed: ~25 tokens/sec
- Quality: Excellent for reasoning/code

---

## Performance Tuning

### Latency vs Throughput

| Priority | Config Strategy |
|----------|----------------|
| **Low Latency** (real-time chat) | No quantization, greedy decoding, low max_tokens |
| **High Throughput** (batch processing) | Quantization, CPU offload, high batch size |
| **Best Quality** (research) | No quantization, beam search, high temperature |

---

### Memory Optimization Checklist

```yaml
# Start here (baseline)
torch_dtype: "bfloat16"
device_map: "auto"
max_gpu_memory: "<90% of VRAM>"
quantization: null

# If OOM (out of memory):
quantization: "8bit"  # Try first

# Still OOM:
quantization: "4bit"

# Still OOM:
max_memory:
  0: "<current_gpu_memory>"
  "cpu": "20GB"  # Enable CPU offload

# Last resort:
# Use smaller model or upgrade GPU
```

---

### Speed Optimization Checklist

```yaml
# Fastest possible inference:
torch_dtype: "bfloat16"  # BF16 fastest on Ampere+
device_map: "auto"
quantization: null       # No dequantization overhead
use_flash_attention_2: true  # If available

# Generation:
do_sample: false         # Greedy (no sampling)
num_beams: 1             # No beam search
max_tokens: 100          # Shorter = faster
```

---

### Quality Optimization Checklist

```yaml
# Best possible quality:
torch_dtype: "bfloat16"
quantization: null       # No precision loss

# Generation:
num_beams: 4             # Beam search
length_penalty: 1.1      # Encourage completeness
temperature: 0.7         # Balanced randomness
no_repeat_ngram_size: 3  # Prevent repetition
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions (in order):**

1. **Reduce max_gpu_memory:**
   ```yaml
   max_gpu_memory: "3GB"  # Was 4GB
   ```

2. **Enable 8-bit quantization:**
   ```yaml
   quantization: "8bit"
   ```

3. **Enable 4-bit quantization:**
   ```yaml
   quantization: "4bit"
   ```

4. **Enable CPU offload:**
   ```yaml
   max_memory:
     0: "3GB"
     "cpu": "20GB"
   ```

5. **Use smaller model:**
   ```yaml
   model_id: "microsoft/phi-2"  # Instead of Llama-7B
   ```

---

### Issue: Slow Generation

**Symptoms:**
- < 5 tokens/second
- High GPU utilization but slow

**Solutions:**

1. **Check quantization:**
   ```yaml
   quantization: null  # Faster than 4-bit/8-bit
   ```

2. **Disable CPU offload:**
   - Ensure all layers on GPU

3. **Use Flash Attention:**
   ```yaml
   use_flash_attention_2: true
   ```

4. **Optimize dtype:**
   ```yaml
   torch_dtype: "bfloat16"  # Faster on Ampere+ than float16
   ```

5. **Reduce beam search:**
   ```yaml
   num_beams: 1  # Fastest
   ```

---

### Issue: Poor Quality Output

**Symptoms:**
- Repetitive text
- Incoherent responses
- Off-topic answers

**Solutions:**

1. **Tune temperature:**
   ```yaml
   temperature: 0.7  # Try 0.5-0.9
   ```

2. **Add repetition penalty:**
   ```yaml
   repetition_penalty: 1.15
   no_repeat_ngram_size: 3
   ```

3. **Enable sampling:**
   ```yaml
   do_sample: true
   top_p: 0.9
   ```

4. **Increase max_tokens:**
   ```yaml
   max_tokens: 512  # Allow complete answers
   ```

5. **Check quantization:**
   ```yaml
   # 4-bit may degrade quality
   quantization: "8bit"  # Or null
   ```

---

### Issue: Deterministic Output

**Symptoms:**
- Same answer every time
- No variation

**Solutions:**

1. **Enable sampling:**
   ```yaml
   do_sample: true
   ```

2. **Increase temperature:**
   ```yaml
   temperature: 0.8  # From 0.0
   ```

3. **Add top_p:**
   ```yaml
   top_p: 0.9
   ```

---

### Issue: Model Not Loading

**Symptoms:**
```
OSError: model.safetensors not found
```

**Solutions:**

1. **Check model_id:**
   ```yaml
   model_id: "meta-llama/Llama-2-7b-chat-hf"  # Correct format
   ```

2. **Authenticate (private models):**
   ```bash
   huggingface-cli login
   ```

3. **Enable trust_remote_code:**
   ```yaml
   trust_remote_code: true  # For Phi, Falcon
   ```

4. **Check internet connection:**
   - Models download on first use (~GB of data)

---

## Complete Example Configs

### Beginner (Stable)

```yaml
generator:
  active_model: "bitnet"
  
  models:
    bitnet:
      type: "huggingface"
      model_id: "microsoft/bitnet-b1.58-2B-4T"
      torch_dtype: "bfloat16"
      device_map: "auto"
      max_gpu_memory: "3.8GB"
      max_tokens: 250
      temperature: 0.7
      do_sample: true
```

---

### Intermediate (Optimized)

```yaml
generator:
  active_model: "mistral"
  
  models:
    mistral:
      type: "huggingface"
      model_id: "mistralai/Mistral-7B-Instruct-v0.2"
      
      # Loading
      torch_dtype: "bfloat16"
      device_map: "auto"
      max_gpu_memory: "7GB"
      quantization: "8bit"
      low_cpu_mem_usage: true
      
      # Generation
      max_tokens: 512
      temperature: 0.7
      do_sample: true
      top_p: 0.9
      top_k: 40
      repetition_penalty: 1.15
      no_repeat_ngram_size: 3
```

---

### Advanced (Multi-GPU + Quality)

```yaml
generator:
  active_model: "llama70b"
  
  models:
    llama70b:
      type: "huggingface"
      model_id: "meta-llama/Llama-2-70b-chat-hf"
      
      # Loading (multi-GPU)
      torch_dtype: "bfloat16"
      device_map: "balanced"
      max_memory:
        0: "40GB"
        1: "40GB"
      quantization: "8bit"
      use_flash_attention_2: true
      
      # Generation (quality)
      max_tokens: 2048
      temperature: 0.7
      do_sample: true
      top_p: 0.9
      num_beams: 2
      repetition_penalty: 1.1
      no_repeat_ngram_size: 3
      length_penalty: 1.05
      early_stopping: true
```

---

## References

### Official Documentation

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Quantization Guide](https://huggingface.co/docs/transformers/main_classes/quantization)
- [Model Parallelism](https://huggingface.co/docs/transformers/main_classes/model#large-model-loading)

### Related Libraries

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - Quantization
- [flash-attn](https://github.com/Dao-AILab/flash-attention) - Flash Attention
- [accelerate](https://huggingface.co/docs/accelerate) - Multi-GPU support