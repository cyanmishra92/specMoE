# Mixtral 8x7B Expert Speculation Training

Complete implementation of expert speculation training for Mixtral 8x7B MoE model.

## Model Overview

**Mixtral 8x7B** is a state-of-the-art Mixture of Experts (MoE) model:
- **45B total parameters** (8 experts × 7B each)
- **14B active parameters** per token (top-2 routing)
- **8 experts per MoE layer**
- **Top-2 routing** (vs Switch Transformer's top-1)
- **Multi-GPU optimized** with automatic GPU selection

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

## Quick Start

1. **Collect Traces**:
   ```bash
   python scripts/collection/collect_mixtral_traces.py
   ```

2. **Train Speculation Model**:
   ```bash
   python scripts/training/train_mixtral_speculation.py
   ```

3. **Analyze Results**:
   ```bash
   python scripts/analysis/visualize_mixtral_routing.py
   ```

## Hardware Requirements

- **GPU**: RTX 3090 (24GB VRAM) minimum, A6000 (48GB) recommended, A100 (40GB/80GB) optimal
- **RAM**: 32GB+ recommended (64GB+ for A100)
- **Storage**: 50GB+ for traces and models
- **Multi-GPU**: Automatic selection of best available GPU

## Features

- **Advanced MoE Architecture**: Top-2 routing vs Switch Transformer's top-1
- **Multi-GPU Support**: Automatic GPU selection and optimal configuration
- **Optimized Data Collection**: 
  - Batch processing for 4-16x speedup (GPU-adaptive)
  - Balanced sampling from 8 diverse datasets
  - 50,000 total traces with max 200 traces per sample
- **Multiple Speculation Models**: InterLayer, Statistics-Aware, and Ensemble
- **Rich Visualizations**: Token journeys, expert usage, routing patterns
- **Performance Metrics**: Top-1/3/5/10 accuracy tracking
- **Hardware Optimization**: RTX 3090, A6000, A100 support

## Getting Started

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for detailed setup instructions.