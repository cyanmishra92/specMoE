# MoE Trace Collection Guide

This guide covers the enhanced Switch Transformer trace collection system for gathering real MoE routing patterns from inference on benchmark datasets.

## 🎯 Overview

The collection system captures **real MoE routing traces** by running inference on Switch Transformers and extracting the router logits that determine which experts are activated for each token. This provides authentic routing patterns for training speculation models.

## 📊 What Are MoE Traces?

**MoE (Mixture of Experts) traces** capture the routing decisions made during inference:

- **Router Logits**: Probability distributions over experts for each token
- **Expert Selection**: Which experts are chosen (top-k routing)
- **Layer Information**: Traces from all MoE layers in the model
- **Sequence Context**: How routing decisions evolve across tokens

### Switch Transformer Architecture
- **128-Expert Model**: `google/switch-base-128` (recommended)
- **MoE Layers**: 6 layers with expert routing (layers 1, 3, 5, 7, 9, 11)
- **Traces per Sample**: 6 routing traces (one per MoE layer)
- **Expert Coverage**: All 128 experts tracked with routing probabilities

## 🚀 Quick Start

### **🎯 Recommended: Maximum Real Trace Collection**
```bash
# Collect diverse real traces from 60+ datasets (RECOMMENDED)
python scripts/collection/collect_maximum_real_traces.py

# This collects from diverse NLP tasks:
# - Core NLP: CNN/DailyMail, IMDB, WikiText, SQuAD, BillSum
# - GLUE: CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE, WNLI, STS-B
# - SuperGLUE: CB, COPA, BoolQ, WSC, MultiRC, ReCoRD, WiC
# - Classification: AG News, Yelp, Yahoo Answers, DBpedia, Amazon
# - Question Answering: SQuAD v2, MS MARCO, TriviaQA, QuAC
# - Summarization: XSum, NewsRoom, Multi-News, Reddit TIFU
# - Dialogue: Daily Dialog, Empathetic Dialogues, Blended Skill Talk
# - Reasoning: CosmosQA, CommonsenseQA, Social IQA, WinoGrande, HellaSwag
# - Scientific: SciTail, PubMed QA, Scientific Papers
# - Reviews: Amazon Reviews, App Reviews
# - Misc: Emotion, Tweet Eval, Financial PhraseBank

# Features:
# - 100% real data (no synthetic traces)
# - Randomized dataset order for diversity
# - Balanced sampling (100-200 traces per dataset)
# - ~10,000 traces from ~50 different datasets
# - Authentic routing patterns across NLP tasks

# After collection, analyze the diverse patterns
python scripts/analysis/comprehensive_expert_analysis.py
```

### **Alternative: Basic Collection**
```bash
# Basic collection with fewer datasets
python scripts/collection/collect_robust_traces.py --traces 3000

# Mixed mode (real + synthetic)
python scripts/collection/collect_robust_traces.py --traces 5000 --mode mixed --real-ratio 0.7

# Real data only mode
python scripts/collection/collect_robust_traces.py --traces 5000 --mode real

# Quick test with smaller model
python scripts/collection/collect_robust_traces.py --traces 100 --model google/switch-base-32
```

### Collection Modes
```bash
# Real traces only (recommended for authentic patterns)
python scripts/collection/collect_robust_traces.py --traces 2000 --mode real

# Synthetic traces (fast generation)
python scripts/collection/collect_robust_traces.py --traces 1000 --mode synthetic  

# Mixed mode (50% real, 50% synthetic)
python scripts/collection/collect_robust_traces.py --traces 2000 --mode mixed --real-ratio 0.5
```

## 📚 Benchmark Datasets

The collection system uses diverse benchmark datasets to capture varied routing patterns:

### Available Datasets

| Dataset | Domain | Samples Available | Task Type | Routing Pattern |
|---------|--------|------------------|-----------|-----------------|
| **CNN/DailyMail** | News | 287,113 | Summarization | Complex reasoning |
| **IMDB** | Movies | 25,000 | Sentiment Analysis | Opinion/emotion |
| **WikiText** | Encyclopedia | 36,718 | Language Modeling | Factual knowledge |
| **SQuAD** | Q&A | 87,599 | Reading Comprehension | Question answering |

**Total**: 436,430+ samples available (far more than needed)

### Dataset Processing
- **Text Preprocessing**: Proper formatting for seq2seq models
- **Length Limits**: Truncated to 256 tokens for efficiency
- **Task-Specific Prompts**: 
  - CNN: `"summarize: {article}"`
  - IMDB: `"analyze sentiment: {review}"`
  - WikiText: `"summarize: {text}"`
  - SQuAD: `"answer: {question} {context}"`

## 🛠️ Command-Line Options

### Basic Parameters
```bash
--traces, -t        Number of traces to collect (default: 3000)
--model, -m         Switch model to use (128/64/32 experts)  
--output, -o        Output file path (default: routing_data/robust_traces.pkl)
```

### Collection Modes
```bash
--mode              Collection strategy:
                    real      - Real dataset inference only
                    synthetic - Generated traces only  
                    mixed     - Combination of both

--real-ratio        For mixed mode: ratio of real traces (default: 0.5)
```

### Model Selection
```bash
# Auto-select largest available (recommended)
python scripts/collection/collect_robust_traces.py --traces 1000

# Force specific model
python scripts/collection/collect_robust_traces.py --traces 1000 --model google/switch-base-128
python scripts/collection/collect_robust_traces.py --traces 500 --model google/switch-base-64  
python scripts/collection/collect_robust_traces.py --traces 200 --model google/switch-base-32
```

## 📊 Collection Capacity

### Real Trace Capacity
- **Script Default**: 1,200 samples × 6 layers = **7,200 traces maximum**
- **Your Target**: 3,000 traces easily achievable
- **Scalability**: Can collect 10,000+ traces if needed

### Performance Estimates
```bash
# Fast collection (~5 minutes)
python scripts/collection/collect_robust_traces.py --traces 500 --mode real

# Medium collection (~15 minutes)  
python scripts/collection/collect_robust_traces.py --traces 2000 --mode real

# Large collection (~30 minutes)
python scripts/collection/collect_robust_traces.py --traces 5000 --mode real
```

## 🔍 Output Format

### Trace Structure
Each trace contains:
```python
GatingDataPoint(
    layer_id=1,                    # MoE layer (1,3,5,7,9,11)
    hidden_states=tensor,          # [seq_len, hidden_size]
    target_routing=tensor,         # [seq_len, 128] - expert probabilities
    target_top_k=tensor,          # [seq_len, 1] - selected expert indices
    prev_layer_gates=[],          # Previous layer routing (for context)
    sequence_length=14,           # Sequence length
    token_ids=tensor,             # Input token IDs
    dataset_name="cnn_dailymail", # Source dataset
    sample_id="cnn_dailymail_42"  # Unique identifier
)
```

### Files Generated
```
routing_data/
├── robust_traces.pkl      # Pickled trace data
├── robust_traces.json     # Metadata (stats, expert usage, timing)
└── ...
```

### Metadata Example
```json
{
  "total_traces": 3000,
  "num_experts": 128,
  "expert_distribution": {"0": 45, "1": 38, ...},
  "experts_used": 98,
  "diversity_percentage": 76.6,
  "dataset_distribution": {
    "cnn_dailymail": 1500,
    "imdb": 900,
    "wikitext": 600
  },
  "collection_time": 1842.3,
  "device": "cuda"
}
```

## 🎯 Use Cases

### For Training Speculation Models
```bash
# Collect diverse real patterns for training
python scripts/collection/collect_robust_traces.py --traces 5000 --mode real
```

### For Quick Prototyping
```bash
# Fast synthetic generation for testing
python scripts/collection/collect_robust_traces.py --traces 1000 --mode synthetic
```

### For Balanced Datasets
```bash
# Mix real and synthetic for larger datasets
python scripts/collection/collect_robust_traces.py --traces 8000 --mode mixed --real-ratio 0.6
```

## 🔧 Advanced Usage

### Custom Output Location
```bash
python scripts/collection/collect_robust_traces.py \
  --traces 2000 \
  --mode real \
  --output routing_data/my_custom_traces.pkl
```

### Specific Model with Large Collection
```bash
python scripts/collection/collect_robust_traces.py \
  --traces 10000 \
  --mode mixed \
  --real-ratio 0.7 \
  --model google/switch-base-128
```

### Memory-Constrained Collection
```bash
# Use smaller model for limited GPU memory
python scripts/collection/collect_robust_traces.py \
  --traces 3000 \
  --mode real \
  --model google/switch-base-32
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Use smaller model or reduce batch processing
python scripts/collection/collect_robust_traces.py --model google/switch-base-32
```

**No traces collected**
```bash
# Check if model loaded successfully - should see "Model loaded successfully!"
# Verify CUDA availability if using GPU
```

**Slow collection**
```bash
# Use synthetic mode for faster generation during testing
python scripts/collection/collect_robust_traces.py --mode synthetic
```

### Debug Mode
For detailed logging, edit the script to use `logging.DEBUG`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Integration with Pipeline

The collection script integrates with the full pipeline:

```bash
# Use in complete pipeline
python scripts/pipelines/run_working_pipeline.py --use-128-experts

# Or run individual stages
python scripts/collection/collect_robust_traces.py --traces 3000 --mode real
python scripts/training/proper_train_test.py
python scripts/evaluation/test_individual_approaches.py
```

## 📈 Expected Results

### Expert Diversity
- **Real traces**: ~70-90% expert coverage
- **Synthetic traces**: 100% expert coverage (by design)
- **Mixed traces**: Balanced diversity with authentic patterns

### Collection Statistics
```
✅ Expert diversity: 98/128 experts (76.6%)
📊 Dataset distribution: {
  'cnn_dailymail': 1500,
  'imdb': 900, 
  'wikitext': 600
}
⏱️ Collection time: 30.7 minutes
💾 Output: routing_data/robust_traces.pkl (156 MB)
```

## 🔬 Technical Details

### MoE Layer Detection
The system automatically detects MoE layers by:
1. Checking for `encoder_router_logits` in model outputs
2. Identifying layers with tuple structure `(router_logits, expert_indices)`
3. Extracting router logits with shape `[batch, seq_len, num_experts]`

### Expert Selection Process
1. **Router Logits**: Raw scores for each expert
2. **Softmax**: Convert to probabilities
3. **Top-K Selection**: Choose best expert(s)
4. **Trace Storage**: Save routing decisions and context

### Memory Management
- **Model Loading**: Automatic device mapping for GPU memory
- **Batch Processing**: Process samples individually to avoid OOM
- **Cleanup**: Automatic GPU memory cleanup between datasets

---

This collection system provides the foundation for training accurate MoE speculation models by capturing real routing patterns from diverse benchmark tasks.