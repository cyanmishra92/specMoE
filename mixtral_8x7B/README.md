# Mixtral 8x7B Expert Speculation Training

Complete implementation of expert speculation training for Mixtral 8x7B MoE model with **memory optimization**.

## Model Overview

**Mixtral 8x7B** is a state-of-the-art Mixture of Experts (MoE) model:
- **45B total parameters** (8 experts × 7B each)
- **14B active parameters** per token (top-2 routing)
- **8 experts per MoE layer**
- **Top-2 routing** (vs Switch Transformer's top-1)
- **Multi-GPU optimized** with automatic GPU selection
- **Data sharding** for RTX 3090 compatibility

## Project Structure

```
mixtral_8x7B/
├── scripts/
│   ├── collection/          # Trace collection from public datasets
│   ├── training/           # Speculation model training
│   └── analysis/           # Visualization and analysis
├── models/                 # Trained speculation models
├── docs/                  # Documentation and guides
├── routing_data/          # Collected MoE traces
└── README.md             # This file
```

## 🚀 Memory-Efficient Training

**NEW**: Data sharding for RTX 3090 compatibility!

### Hardware Requirements
- **RTX 3090 (24GB)**: ✅ Supported with data sharding
- **A6000 (48GB)**: ✅ Recommended, larger batches
- **A100 (40/80GB)**: ✅ Optimal performance
- **RAM**: 32GB+ recommended (64GB+ for A100)
- **Storage**: 50GB+ for traces and models

### Quick Start

1. **Collect Traces with Sharding**:
   ```bash
   python scripts/collection/collect_mixtral_traces_medium.py \
     --target_traces 3000 \
     --shard_data \
     --shard_size_mb 500
   ```

2. **Train with Sharded Data**:
   ```bash
   python scripts/training/train_mixtral_speculation.py \
     --shard_dir routing_data/mixtral_8x7b_traces_medium_shards \
     --batch_size 4
   ```

3. **Analyze Results**:
   ```bash
   python scripts/analysis/visualize_mixtral_routing.py
   ```

## 🆕 Features

- **Advanced MoE Architecture**: Top-2 routing vs Switch Transformer's top-1
- **Data Sharding**: Automatic memory-efficient training for RTX 3090
- **Multi-GPU Support**: Automatic GPU selection and optimal configuration
- **Optimized Data Collection**: 
  - Batch processing for 4-16x speedup (GPU-adaptive)
  - Balanced sampling from 8 diverse datasets
  - Configurable trace counts (default: 1500 medium traces)
  - Automatic sharding with customizable shard sizes
- **Multiple Speculation Models**: InterLayer, Statistics-Aware, and Ensemble
- **Rich Visualizations**: Token journeys, expert usage, routing patterns
- **Performance Metrics**: Top-1/3/5/10 accuracy tracking
- **Hardware Optimization**: RTX 3090, A6000, A100 support with memory management

## 🎯 Command Line Options

### Trace Collection
```bash
# Configurable trace count and sharding
--target_traces 3000              # Number of traces to collect
--output_suffix rtx3090           # Custom output filename
--shard_data                      # Enable automatic sharding
--shard_size_mb 500              # Shard size in MB
```

### Training
```bash
# Memory-efficient training options
--shard_dir path/to/shards        # Use sharded data
--batch_size 4                    # Batch size for RTX 3090
--gradient_accumulation_steps 4   # Effective larger batch size
--device cuda                     # Training device
```

## Getting Started

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for detailed setup instructions.