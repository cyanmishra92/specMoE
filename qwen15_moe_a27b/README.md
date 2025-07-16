# Qwen1.5-MoE-A2.7B Expert Speculation Training

Complete implementation of multi-expert prediction training for Qwen1.5-MoE-A2.7B model with **RTX 3090 memory optimization**.

## Model Overview

**Qwen1.5-MoE-A2.7B** is a highly efficient small MoE model:
- **14.3B total parameters** (60 routing + 4 shared experts)
- **2.7B active parameters** per token (top-4 routing)
- **60 routing experts + 4 shared experts** per MoE layer
- **Top-4 routing** (4 out of 60 experts selected per token)
- **RTX 3090/A6000 optimized** with data sharding

## Key Advantages over Mixtral

- **5x smaller active parameters** (2.7B vs 14B)
- **3x smaller total parameters** (14.3B vs 45B)
- **Much faster inference** on consumer GPUs
- **Same top-2 routing strategy** for comparable speculation training
- **Better GPU utilization** on RTX 3090/A6000

## 🚀 RTX 3090 Memory Optimization

**NEW**: Data sharding for memory-efficient training on RTX 3090 (24GB)!

### Hardware Requirements
- **RTX 3090 (24GB)**: ✅ Perfect with data sharding
- **A6000 (48GB)**: ✅ Excellent performance, larger batches
- **RTX 4090 (24GB)**: ✅ Optimal performance
- **Minimum**: 16GB VRAM with 4-bit quantization + sharding

### Memory-Efficient Workflow

1. **Collect & Shard Traces**:
   ```bash
   # Automatic sharding for RTX 3090
   python scripts/collection/collect_qwen15_moe_traces_medium.py \
     --target_traces 5000 \
     --shard_data \
     --shard_size_mb 400
   ```

2. **Train with Sharded Data**:
   ```bash
   python scripts/train_multi_expert_predictor.py \
     --shard_dir routing_data/qwen15_moe_a27b_traces_medium_shards \
     --batch_size 4 \
     --epochs 50
   ```

3. **Run RTX 3090 Example**:
   ```bash
   python scripts/examples/rtx3090_training_example.py
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
├── 60 routing experts per MoE layer
├── 4 shared experts (always active)
├── Top-4 expert selection from routing experts
├── 2048 hidden dimensions
├── 1408 intermediate size per routing expert
└── 32,000 vocabulary size
```

## Training Configuration

```python
# RTX 3090 Optimized (with sharding)
config = {
    'batch_size': 4,
    'max_length': 256,
    'quantization': '4bit',
    'device_map': 'auto',
    'cpu_offload': True,
    'shard_size_mb': 400
}

# A6000 Optimized  
config = {
    'batch_size': 8,
    'max_length': 256,
    'quantization': '8bit',
    'device_map': 'auto',
    'cpu_offload': False,
    'shard_size_mb': 500
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

Based on model architecture and top-4 routing:
- **Random baseline**: ~6.7% (4/60 experts)
- **Most frequent**: ~15-20%
- **Multi-expert prediction**: ~35-45% (target)
- **Pattern-based**: ~40-50% (target)

## Comparison with Other Models

| Model | Active Params | Total Params | VRAM Needed | RTX 3090 Compatible |
|-------|---------------|--------------|-------------|-------------------|
| Qwen1.5-MoE-A2.7B | 2.7B | 14.3B | 8-12GB | ✅ Perfect (sharded) |
| DeepSeek-MoE-16B | 2.8B | 16.4B | 16-20GB | ✅ Good (sharded) |
| Mixtral-8x7B | 14B | 45B | 24GB+ | ⚠️ Tight fit |

## 🆕 New Features

- **Multi-Expert Prediction**: Predicts top-4 experts simultaneously
- **Data Sharding**: Automatic memory-efficient training
- **GPU-Adaptive Configuration**: RTX 3090/A6000 optimized
- **Balanced Dataset Sampling**: Configurable trace counts
- **Progress Tracking**: Real-time collection rate monitoring
- **Memory Monitoring**: Automatic GPU cache management

## Files Structure

```
qwen15_moe_a27b/
├── scripts/
│   ├── collection/          # Trace collection scripts
│   │   ├── collect_qwen15_moe_traces_small.py    # 10 traces (dev)
│   │   ├── collect_qwen15_moe_traces_medium.py   # Configurable (2000 default)
│   │   └── collect_qwen15_moe_traces.py          # Full traces
│   ├── utils/
│   │   └── data_sharding.py                      # Memory-efficient sharding
│   ├── examples/
│   │   └── rtx3090_training_example.py          # RTX 3090 workflow
│   ├── train_multi_expert_predictor.py          # Multi-expert training
│   └── analysis/                                 # Visualization and analysis
├── models/
│   └── multi_expert_predictor.py               # Multi-expert model
├── routing_data/                                # Collected MoE traces
│   └── *_shards/                               # Sharded data directories
└── README.md                                   # This file
```

## Getting Started

1. **Setup Environment**:
   ```bash
   pip install torch transformers datasets tqdm bitsandbytes accelerate
   pip install GPUtil seaborn matplotlib numpy scipy
   ```

2. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```

3. **Quick Test (RTX 3090)**:
   ```bash
   python scripts/examples/rtx3090_training_example.py
   ```

4. **Full Training Pipeline**:
   ```bash
   # Collect traces with sharding
   python scripts/collection/collect_qwen15_moe_traces_medium.py \
     --target_traces 5000 --shard_data --shard_size_mb 400
   
   # Train predictor
   python scripts/train_multi_expert_predictor.py \
     --shard_dir routing_data/qwen15_moe_a27b_traces_medium_shards \
     --batch_size 4 --epochs 50
   ```

## 🎯 Command Line Options

### Trace Collection
```bash
# Configurable trace count
--target_traces 5000              # Number of traces to collect
--output_suffix rtx3090           # Custom output filename
--shard_data                      # Enable automatic sharding
--shard_size_mb 400              # Shard size in MB
```

### Training
```bash
# Memory-efficient training
--shard_dir path/to/shards        # Use sharded data
--batch_size 4                    # Batch size for RTX 3090
--epochs 50                       # Training epochs
--lr 1e-4                         # Learning rate
--device cuda                     # Training device
```

Perfect for RTX 3090 and A6000 users who want efficient MoE expert speculation training with **memory optimization**!