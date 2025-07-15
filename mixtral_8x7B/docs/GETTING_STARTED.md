# Getting Started with Mixtral 8x7B Expert Speculation

## Overview

This project implements expert speculation training for Mixtral 8x7B, focusing on predicting which experts will be activated for upcoming tokens.

## Hardware Requirements

### Minimum Requirements (RTX 3090)
- **GPU**: RTX 3090 (24GB VRAM)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space
- **CUDA**: 11.8 or higher

### Recommended Requirements
- **GPU**: RTX 4090 or A100
- **RAM**: 64GB system RAM
- **Storage**: 100GB free space

## Installation

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets accelerate bitsandbytes
   pip install numpy matplotlib seaborn tqdm
   ```

2. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd mixtral_8x7B
   ```

## Quick Start

### Step 1: Collect MoE Traces

```bash
python scripts/collection/collect_mixtral_traces.py
```

This will:
- Load Mixtral 8x7B with 8-bit quantization
- Collect expert routing traces from 10 public datasets
- Save ~150 samples per dataset
- Generate `routing_data/mixtral_8x7b_traces.pkl`

### Step 2: Train Speculation Models

```bash
python scripts/training/train_mixtral_speculation.py
```

This will train multiple speculation models:
- InterLayer speculation model
- Statistics-aware model
- Ensemble model (if enabled)

### Step 3: Analyze Results

```bash
python scripts/analysis/visualize_mixtral_routing.py
```

This generates:
- Expert usage heatmaps
- Token journey visualizations
- Routing pattern analysis
- Performance metrics

## Project Structure

```
mixtral_8x7B/
├── scripts/
│   ├── collection/
│   │   └── collect_mixtral_traces.py    # Trace collection
│   ├── training/
│   │   └── train_mixtral_speculation.py # Model training
│   └── analysis/
│       └── visualize_mixtral_routing.py # Analysis & viz
├── models/                              # Saved models
├── routing_data/                        # Collected traces
└── docs/                               # Documentation
```

## Key Differences from Switch Transformer

| Feature | Switch Transformer | Mixtral 8x7B |
|---------|-------------------|--------------|
| **Routing** | Top-1 | Top-2 |
| **Experts** | 128 | 8 |
| **Parameters** | 7B total | 45B total, 14B active |
| **Load Balancing** | Auxiliary loss | Built-in |
| **Architecture** | Encoder-decoder | Decoder-only |

## Memory Optimization

### RTX 3090 Settings
- 8-bit model loading
- Gradient checkpointing
- Smaller batch sizes (16-32)
- Memory cleanup between datasets

### Tips for Success
1. **Monitor GPU memory**: `nvidia-smi` during collection
2. **Use smaller sequences**: Max 512 tokens
3. **Clear cache regularly**: `torch.cuda.empty_cache()`
4. **Adjust batch size**: Start with 16, increase if stable

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size in training script
   batch_size = 16  # instead of 32
   ```

2. **Model Loading Fails**:
   ```bash
   # Install latest transformers
   pip install transformers --upgrade
   ```

3. **Dataset Loading Issues**:
   ```bash
   # Clear HuggingFace cache
   rm -rf ~/.cache/huggingface/
   ```

## Next Steps

1. **Collect traces**: Start with trace collection
2. **Train models**: Run speculation training
3. **Compare results**: Analyze vs Switch Transformer
4. **Experiment**: Try different architectures
5. **Scale up**: Move to larger models if needed

## Support

For issues or questions:
- Check the troubleshooting section
- Review the code comments
- Compare with the Switch Transformer implementation