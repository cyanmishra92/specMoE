# Switch Transformer Training and Evaluation Guide

This document provides comprehensive instructions for training and evaluating Switch Transformer models on research paper data.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Training Options](#training-options)
3. [Evaluation Methods](#evaluation-methods)
4. [Hardware Requirements](#hardware-requirements)
5. [Troubleshooting](#troubleshooting)
6. [Results Interpretation](#results-interpretation)

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you're in the scripts directory
cd switch_training/scripts
```

### Basic Training
```bash
# Stable Switch Transformer training (recommended)
python train_switch_stabilized.py \
  --model_name google/switch-base-8 \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --patience 10
```

### Basic Evaluation
```bash
# Quick evaluation
python simple_switch_eval.py --model_path ../models/switch_stabilized/final_model

# Comprehensive evaluation
python evaluate_model.py --model_path ../models/switch_stabilized/final_model --model_type switch
```

## 🎯 Training Options

### 1. Stabilized Switch Transformer (Primary Method)

**Purpose**: Fine-tune pre-trained Switch Transformers with extensive stability measures.

**Script**: `train_switch_stabilized.py`

**Key Features**:
- Router weight stabilization to prevent routing chaos
- Ultra-safe data filtering (ASCII only, proper sentence structure)
- FP32 training for maximum numerical stability
- Aggressive gradient clipping and NaN detection
- Emergency checkpointing

**Basic Usage**:
```bash
python train_switch_stabilized.py [OPTIONS]
```

**Available Options**:
- `--model_name`: Switch model to use (default: `google/switch-base-8`)
- `--batch_size`: Training batch size (default: 1)
- `--learning_rate`: Learning rate (default: 5e-6)
- `--num_epochs`: Number of training epochs (default: 2)
- `--max_length`: Maximum sequence length (default: 256)
- `--patience`: Early stopping patience (default: 3)
- `--warmup_ratio`: Warmup ratio for scheduler (default: 0.2)
- `--weight_decay`: Weight decay for optimizer (default: 0.001)
- `--data_dir`: Directory containing training data (default: "../data")
- `--output_dir`: Output directory for trained models (default: "../models/switch_stabilized")

### 2. Small MoE Model (Alternative)

**Purpose**: Train a custom MoE model based on GPT-2 with added expert layers.

**Script**: `train_small_moe.py`

**Usage**:
```bash
python train_small_moe.py \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --num_epochs 3
```

### 3. Mixtral MoE (Large Scale)

**Purpose**: Fine-tune Mixtral models (requires significant memory).

**Script**: `train_mixtral_moe.py`

**Usage**:
```bash
python train_mixtral_moe.py \
  --model_name mistralai/Mixtral-8x7B-v0.1 \
  --batch_size 1
```

## 🎛️ Hardware-Specific Configurations

### RTX 3090 (24GB VRAM)
```bash
# Conservative settings for stability
python train_switch_stabilized.py \
  --model_name google/switch-base-8 \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --patience 10
```

### A6000 (48GB VRAM)
```bash
# Larger model with higher batch size
CUDA_VISIBLE_DEVICES=0 python train_switch_stabilized.py \
  --model_name google/switch-base-16 \
  --batch_size 4 \
  --learning_rate 8e-6 \
  --num_epochs 4 \
  --patience 8 \
  --max_length 512 \
  --output_dir ../models/switch_large
```

### Multi-GPU Systems
```bash
# Force single GPU to avoid peer mapping issues
CUDA_VISIBLE_DEVICES=0 python train_switch_stabilized.py [OPTIONS]
```

## 📊 Evaluation Methods

### 1. Simple Evaluation (Recommended First)

**Purpose**: Quick assessment of model functionality and basic metrics.

**Script**: `simple_switch_eval.py`

**What it measures**:
- Basic text generation capability
- Perplexity on training data format
- Model loading verification
- Generation success rate

**Usage**:
```bash
python simple_switch_eval.py --model_path PATH_TO_MODEL
```

**Output**: 
- JSON results file with detailed metrics
- Text summary with human-readable results

### 2. Comprehensive Evaluation

**Purpose**: Detailed analysis including domain-specific understanding.

**Script**: `evaluate_model.py`

**What it measures**:
- Perplexity calculation with proper seq2seq handling
- Token-level prediction accuracy
- Research domain concept understanding
- Text generation quality assessment
- Model parameter statistics

**Usage**:
```bash
python evaluate_model.py \
  --model_path PATH_TO_MODEL \
  --model_type MODEL_TYPE
```

**Supported Model Types**:
- `switch`: Switch Transformer models
- `small_moe`: Custom small MoE models
- `mixtral`: Mixtral-based models

### 3. Custom Evaluation

**Purpose**: Evaluate specific aspects or create custom metrics.

**Example**:
```python
from simple_switch_eval import simple_evaluation

# Custom evaluation
results = simple_evaluation("../models/switch_stabilized/final_model")
print(f"Perplexity: {results.get('perplexity', 'N/A')}")
```

## 📈 Model Configurations

### Available Switch Transformer Models

| Model | Parameters | Experts | Memory Req | Recommended Hardware |
|-------|------------|---------|------------|---------------------|
| `google/switch-base-8` | 619M | 8 | ~6GB | RTX 3090+ |
| `google/switch-base-16` | 1.07B | 16 | ~10GB | A6000+ |
| `google/switch-base-32` | 2.1B | 32 | ~18GB | A100+ |

### Training Time Estimates

| Model | Hardware | Batch Size | Estimated Time |
|-------|----------|------------|----------------|
| switch-base-8 | RTX 3090 | 2 | 15-30 min |
| switch-base-16 | A6000 | 4 | 30-60 min |
| switch-base-32 | A100 | 6 | 60-120 min |

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. NaN Loss During Training
**Symptoms**: Loss becomes NaN, training stops
**Causes**: Learning rate too high, numerical instability
**Solutions**:
- Reduce learning rate (try 5e-6 instead of 1e-5)
- Ensure FP32 training (avoid mixed precision)
- Check router weight initialization

#### 2. CUDA Out of Memory
**Symptoms**: `RuntimeError: CUDA out of memory`
**Solutions**:
```bash
# Reduce batch size
--batch_size 1

# Increase gradient accumulation
--gradient_accumulation 8

# Use smaller model
--model_name google/switch-base-8
```

#### 3. Multi-GPU Issues
**Symptoms**: `peer mapping resources exhausted`
**Solutions**:
```bash
# Force single GPU
CUDA_VISIBLE_DEVICES=0 python train_switch_stabilized.py [OPTIONS]
```

#### 4. Poor Generation Quality
**Symptoms**: Gibberish output, repetitive text
**Causes**: Insufficient training, poor data quality
**Solutions**:
- Increase training epochs
- Reduce early stopping patience
- Check data filtering settings

#### 5. Slow Training
**Symptoms**: Very slow iteration speed
**Solutions**:
- Increase batch size if memory allows
- Use mixed precision (but less stable)
- Check for CPU bottlenecks in data loading

### Data Issues

#### File Not Found Errors
```bash
# Ensure correct directory structure
ls ../data/train/finetuning_train.json
ls ../data/val/finetuning_val.json
ls ../data/test/finetuning_test.json
```

#### Empty or Insufficient Data
**Check data statistics**:
```bash
python debug_data.py
```

## 📋 Results Interpretation

### Perplexity Guidelines

| Perplexity Range | Quality | Interpretation |
|------------------|---------|----------------|
| 10-30 | Excellent | Well-trained, coherent generation |
| 30-80 | Good | Reasonable performance, some issues |
| 80-200 | Poor | Undertrained, many coherence problems |
| 200+ | Very Poor | Severe training issues, mostly gibberish |

### Generation Quality Assessment

**Good Signs**:
- Coherent sentence structure
- Research paper terminology usage
- Logical flow of ideas
- Minimal repetition

**Warning Signs**:
- Repetitive phrases
- Broken sentence structure
- Random punctuation
- Non-English output

### Training Progress Indicators

**Healthy Training**:
- Decreasing loss over time
- Stable gradient norms
- No NaN occurrences
- Validation loss tracking training loss

**Problematic Training**:
- Loss plateaus immediately
- Frequent NaN warnings
- Extremely high or low gradient norms
- Large gap between training and validation loss

## 🗂️ File Structure

```
switch_training/
├── scripts/
│   ├── train_switch_stabilized.py    # Main training script
│   ├── simple_switch_eval.py         # Quick evaluation
│   ├── evaluate_model.py             # Comprehensive evaluation
│   ├── train_small_moe.py           # Alternative MoE training
│   ├── train_mixtral_moe.py         # Mixtral training
│   ├── debug_data.py                # Data debugging
│   └── config.yaml                  # Configuration file
├── data/
│   ├── train/finetuning_train.json  # Training data
│   ├── val/finetuning_val.json      # Validation data
│   └── test/finetuning_test.json    # Test data
├── models/                          # Saved models (gitignored)
├── evaluations/                     # Evaluation results
├── requirements.txt                 # Python dependencies
└── TRAINING_GUIDE.md               # This document
```

## 🔄 Workflow Example

### Complete Training and Evaluation Workflow

1. **Prepare Environment**:
```bash
cd switch_training/scripts
pip install -r ../requirements.txt
```

2. **Start Training**:
```bash
python train_switch_stabilized.py \
  --model_name google/switch-base-8 \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --patience 10
```

3. **Quick Evaluation**:
```bash
python simple_switch_eval.py --model_path ../models/switch_stabilized/final_model
```

4. **Comprehensive Evaluation**:
```bash
python evaluate_model.py \
  --model_path ../models/switch_stabilized/final_model \
  --model_type switch
```

5. **Review Results**:
```bash
# Check evaluation outputs
ls ../evaluations/
cat ../evaluations/simple_switch_eval_*.txt
```

6. **Iterate if Needed**:
```bash
# If results are poor, retrain with different settings
python train_switch_stabilized.py \
  --learning_rate 5e-6 \
  --num_epochs 8 \
  --patience 15
```

## 📚 Additional Resources

- **Switch Transformer Paper**: [Switch Transformer: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- **HuggingFace Documentation**: [Switch Transformers](https://huggingface.co/docs/transformers/model_doc/switch_transformers)
- **MoE Training Best Practices**: [Mixture of Experts Guide](https://huggingface.co/blog/moe)

## 🐛 Bug Reports and Issues

If you encounter issues not covered in this guide:

1. Check the emergency checkpoint directory for partial training state
2. Review the full error traceback
3. Verify hardware compatibility and memory requirements
4. Test with smaller models or reduced batch sizes
5. Ensure all dependencies are correctly installed

---

*Last updated: July 12, 2025*