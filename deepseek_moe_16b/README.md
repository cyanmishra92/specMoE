# DeepSeek-MoE-16B Expert Speculation Training

Complete implementation of expert speculation training for DeepSeek-MoE-16B model.

## Model Overview

**DeepSeek-MoE-16B** is a sophisticated small MoE model:
- **16.4B total parameters** (64 experts × ~256M each)
- **2.8B active parameters** per token (top-2 routing)
- **64 experts per MoE layer** (fine-grained expert specialization)
- **Top-2 routing** with more expert diversity
- **RTX 3090/A6000 compatible** with quantization

## Key Advantages

- **Fine-grained experts**: 64 experts vs 8 in Mixtral/Qwen
- **Efficient activation**: Only 2.8B active parameters
- **Better specialization**: More experts = better task specialization
- **Moderate size**: Fits on RTX 3090 with 4-bit quantization
- **Research-friendly**: Open weights from DeepSeek-AI

## Hardware Requirements

- **A6000 (48GB)**: Optimal performance with 4-bit quantization
- **RTX 3090 (24GB)**: Good performance with 4-bit + CPU offload
- **A100 (40GB/80GB)**: Excellent performance
- **Minimum**: 20GB VRAM with aggressive quantization

## Quick Start

1. **Collect Traces**:
   ```bash
   cd deepseek_moe_16b
   python scripts/collection/collect_deepseek_moe_traces.py
   ```

2. **Train Speculation Model**:
   ```bash
   python scripts/training/train_deepseek_moe_speculation.py
   ```

3. **Analyze Results**:
   ```bash
   python scripts/analysis/visualize_deepseek_moe_routing.py
   ```

## Performance Expectations

### Trace Collection
- **A6000**: ~1500-2500 traces/minute
- **RTX 3090**: ~1000-1500 traces/minute (with CPU offload)
- **Target**: 20,000 traces (~3,333 per dataset)
- **Collection time**: 15-30 minutes

### Memory Usage
- **Model loading**: ~16-20GB VRAM (4-bit quantization)
- **Inference**: ~20-24GB VRAM peak
- **Batch processing**: Up to 12 samples (A6000), 8 samples (RTX 3090)

## Model Architecture

```
DeepSeek-MoE-16B Architecture:
├── 28 Transformer layers
├── 14 MoE layers (every 2nd layer)
├── 64 experts per MoE layer
├── Top-2 expert selection
├── 2048 hidden dimensions
└── 32,000 vocabulary size
```

## Expert Specialization

With 64 experts, DeepSeek-MoE achieves finer specialization:
- **Domain experts**: Math, Code, Science, Literature
- **Task experts**: Generation, Analysis, Translation
- **Style experts**: Formal, Casual, Technical
- **Language experts**: English, Chinese, Multilingual

## Training Configuration

```python
# A6000 Optimized
config = {
    'batch_size': 12,
    'max_length': 512,
    'quantization': '4bit',
    'device_map': 'auto',
    'cpu_offload': False
}

# RTX 3090 Optimized
config = {
    'batch_size': 8,
    'max_length': 512,
    'quantization': '4bit',
    'device_map': 'auto',
    'cpu_offload': True
}
```

## Dataset Coverage

- **IMDB**: Movie reviews (~3,333 traces)
- **Yelp**: User reviews (~3,333 traces)
- **AG News**: News articles (~3,333 traces)
- **Squad**: Q&A contexts (~3,333 traces)
- **Amazon**: Product reviews (~3,333 traces)
- **DBpedia**: Structured knowledge (~3,333 traces)

## Expected Accuracy

With 64 experts, prediction is more challenging but potentially more rewarding:
- **Random baseline**: ~1.56% (1/64 experts)
- **Most frequent**: ~8-12%
- **Pattern-based**: ~25-35% (target)
- **Statistics-aware**: ~30-40% (target)

## Comparison with Other Models

| Model | Active Params | Total Params | Experts | VRAM Needed | Expert Diversity |
|-------|---------------|--------------|---------|-------------|------------------|
| DeepSeek-MoE-16B | 2.8B | 16.4B | 64 | 16-20GB | ✅ High |
| Qwen1.5-MoE-A2.7B | 2.7B | 14.3B | 8 | 8-12GB | ⚠️ Medium |
| Mixtral-8x7B | 14B | 45B | 8 | 24GB+ | ⚠️ Medium |

## Expert Routing Challenge

DeepSeek-MoE presents unique challenges:
- **64-way classification**: Much harder than 8-way
- **Sparse activation patterns**: More diverse routing
- **Fine-grained specialization**: Experts are more specialized
- **Hierarchical routing**: Some experts may be sub-specialized

## Features

- **Fine-grained MoE**: 64 experts for better specialization
- **Efficient Quantization**: 4-bit quantization for RTX 3090
- **Balanced Sampling**: 20,000 traces from 6 clean datasets
- **Progress Tracking**: Real-time monitoring with expert diversity metrics
- **Memory Optimization**: Aggressive memory management for large expert count

## Files Structure

```
deepseek_moe_16b/
├── scripts/
│   ├── collection/          # Trace collection scripts
│   ├── training/           # Speculation model training
│   └── analysis/           # Expert diversity analysis
├── models/                 # Trained speculation models
├── routing_data/          # Collected MoE traces
└── README.md             # This file
```

## Research Opportunities

DeepSeek-MoE-16B offers unique research opportunities:
- **Expert clustering**: How do 64 experts organize?
- **Hierarchical routing**: Are there expert hierarchies?
- **Specialization analysis**: What does each expert learn?
- **Routing patterns**: More complex than 8-expert systems

## Getting Started

1. **Setup Environment**:
   ```bash
   pip install torch transformers datasets tqdm bitsandbytes accelerate
   ```

2. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```

3. **Run Collection**:
   ```bash
   python scripts/collection/collect_deepseek_moe_traces.py
   ```

Perfect for researchers interested in fine-grained MoE expert speculation and specialization analysis!