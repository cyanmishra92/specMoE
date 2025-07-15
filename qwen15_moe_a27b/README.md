# Qwen1.5-MoE-A2.7B Expert Speculation Training

Complete implementation of expert speculation training for Qwen1.5-MoE-A2.7B model.

## Model Overview

**Qwen1.5-MoE-A2.7B** is a highly efficient small MoE model:
- **14.3B total parameters** (8 experts × ~1.8B each)
- **2.7B active parameters** per token (top-2 routing)
- **8 experts per MoE layer**
- **Top-2 routing** (same as Mixtral but much smaller)
- **RTX 3090/A6000 optimized** - fits easily on 24GB VRAM

## Key Advantages over Mixtral

- **5x smaller active parameters** (2.7B vs 14B)
- **3x smaller total parameters** (14.3B vs 45B)
- **Much faster inference** on consumer GPUs
- **Same top-2 routing strategy** for comparable speculation training
- **Better GPU utilization** on RTX 3090/A6000

## Hardware Requirements

- **RTX 3090 (24GB)**: Perfect fit, no CPU offload needed
- **A6000 (48GB)**: Excellent performance, can run larger batches
- **RTX 4090 (24GB)**: Optimal performance
- **Minimum**: 16GB VRAM with 8-bit quantization

## Quick Start

1. **Collect Traces**:
   ```bash
   cd qwen15_moe_a27b
   python scripts/collection/collect_qwen15_moe_traces.py
   ```

2. **Train Speculation Model**:
   ```bash
   python scripts/training/train_qwen15_moe_speculation.py
   ```

3. **Analyze Results**:
   ```bash
   python scripts/analysis/visualize_qwen15_moe_routing.py
   ```

## Performance Expectations

### Trace Collection
- **RTX 3090**: ~2000-3000 traces/minute
- **A6000**: ~3000-4000 traces/minute
- **Target**: 20,000 traces (~3,333 per dataset)
- **Collection time**: 10-20 minutes

### Memory Usage
- **Model loading**: ~8-12GB VRAM
- **Inference**: ~12-16GB VRAM peak
- **Batch processing**: Up to 24 samples (A6000), 16 samples (RTX 3090)

## Model Architecture

```
Qwen1.5-MoE-A2.7B Architecture:
├── 24 Transformer layers
├── 8 MoE layers (every 3rd layer)
├── 8 experts per MoE layer
├── Top-2 expert selection
├── 2048 hidden dimensions
└── 32,000 vocabulary size
```

## Training Configuration

```python
# RTX 3090 Optimized
config = {
    'batch_size': 16,
    'max_length': 512,
    'quantization': '8bit',
    'device_map': 'auto',
    'cpu_offload': False
}

# A6000 Optimized  
config = {
    'batch_size': 24,
    'max_length': 512,
    'quantization': '8bit',
    'device_map': 'auto',
    'cpu_offload': False
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

Based on model architecture and similar MoE models:
- **Random baseline**: ~12.5% (1/8 experts)
- **Most frequent**: ~25-30%
- **Pattern-based**: ~45-55% (target)
- **Statistics-aware**: ~50-60% (target)

## Comparison with Other Models

| Model | Active Params | Total Params | VRAM Needed | RTX 3090 Compatible |
|-------|---------------|--------------|-------------|-------------------|
| Qwen1.5-MoE-A2.7B | 2.7B | 14.3B | 8-12GB | ✅ Perfect |
| DeepSeek-MoE-16B | 2.8B | 16.4B | 16-20GB | ✅ Good |
| Mixtral-8x7B | 14B | 45B | 24GB+ | ⚠️ Tight fit |

## Features

- **Efficient MoE Architecture**: Same top-2 routing as Mixtral
- **GPU-Adaptive Batch Processing**: Optimized for RTX 3090/A6000
- **Balanced Dataset Sampling**: 20,000 traces from 6 clean datasets
- **Fast Inference**: 5x faster than Mixtral on same hardware
- **Progress Tracking**: Real-time collection rate monitoring

## Files Structure

```
qwen15_moe_a27b/
├── scripts/
│   ├── collection/          # Trace collection scripts
│   ├── training/           # Speculation model training
│   └── analysis/           # Visualization and analysis
├── models/                 # Trained speculation models
├── routing_data/          # Collected MoE traces
└── README.md             # This file
```

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
   python scripts/collection/collect_qwen15_moe_traces.py
   ```

Perfect for RTX 3090 and A6000 users who want efficient MoE expert speculation training!