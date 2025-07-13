# Switch Transformer Training on Custom Research Papers

## Project Overview

This subdirectory contains the complete pipeline for training a Switch Transformer model on custom research papers using RTX 3090 hardware.

## Directory Structure

```
switch_training/
├── data/
│   ├── raw_pdfs/           # Place your PDF research papers here
│   ├── processed/          # Extracted and cleaned text
│   ├── train/             # Training data splits
│   ├── val/               # Validation data splits
│   └── test/              # Test data splits
├── models/                # Trained model checkpoints
├── scripts/               # Training and processing scripts
├── logs/                  # Training logs and metrics
└── benchmarks/            # Evaluation results
```

## Hardware Requirements

- **GPU**: RTX 3090 (24GB VRAM) ✅
- **Model Size**: Switch-Base (256M-512M parameters)
- **Expert Count**: 8-32 experts (optimized for single GPU)
- **Batch Size**: 4-8 (depending on sequence length)

## Quick Start

1. **Place PDFs**: Copy research papers to `data/raw_pdfs/`
2. **Process Data**: Run `scripts/process_pdfs.py`
3. **Train Model**: Run `scripts/train_switch.py`
4. **Evaluate**: Run `scripts/benchmark_model.py`

## Training Configuration

### Model Architecture
- **Base Model**: Switch Transformer
- **Hidden Size**: 768
- **Expert Count**: 16 experts
- **Layers**: 12 
- **Attention Heads**: 12
- **Expert Capacity**: 2.0
- **Total Parameters**: ~400M

### Training Setup
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 (with warmup)
- **Batch Size**: 6
- **Sequence Length**: 512
- **Training Steps**: 50,000-100,000
- **Validation Frequency**: Every 1,000 steps

## Expected Timeline

- **Data Processing**: 30-60 minutes
- **Training Setup**: 15 minutes
- **Training Time**: 12-24 hours (depending on data size)
- **Evaluation**: 30 minutes

## Memory Usage

- **Model**: ~1.5GB VRAM
- **Training**: ~18-20GB VRAM (with gradient accumulation)
- **Inference**: ~2-3GB VRAM

Perfect fit for RTX 3090! 🚀