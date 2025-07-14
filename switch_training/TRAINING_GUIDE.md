# Switch Transformer Training and Evaluation Guide

This document provides comprehensive instructions for training and evaluating Switch Transformer models on research paper data.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training Options](#training-options)
4. [Evaluation Methods](#evaluation-methods)
5. [Hardware Requirements](#hardware-requirements)
6. [Troubleshooting](#troubleshooting)
7. [Results Interpretation](#results-interpretation)

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install PyPDF2 PyMuPDF pdfplumber nltk transformers torch wandb

# Ensure you're in the scripts directory
cd switch_training/scripts
```

### Complete Workflow
1. **Process PDFs → Generate Dataset**
2. **Train Switch Transformer**
3. **Evaluate Model**

## 📄 Data Preparation

### PDF Processing Pipeline

The data preparation follows this workflow:
1. **Raw PDFs** → Individual processed JSONs → **Combined train/val/test datasets**

### Step 1: Process PDFs
```bash
# Process all PDFs in raw_pdfs directory
python process_pdfs.py --raw_pdf_dir ../data/raw_pdfs --output_dir ../data
```

**What this does**:
- Extracts text from PDFs using PyMuPDF, pdfplumber, and PyPDF2
- Cleans and normalizes text (removes URLs, page numbers, etc.)
- Segments text into training-suitable chunks
- Creates train/val/test splits (80%/10%/10%)
- Generates both JSON and TXT files for inspection

**Output Structure**:
```
data/
├── processed/          # Individual PDF processing results
│   ├── paper1.txt     # Raw extracted text
│   └── paper1_processed.json  # Processed segments
├── train/
│   ├── train_data.json    # Training dataset
│   └── train_data.txt     # Human-readable version
├── val/
│   ├── val_data.json      # Validation dataset
│   └── val_data.txt       # Human-readable version
└── test/
    ├── test_data.json     # Test dataset
    └── test_data.txt      # Human-readable version
```

### Step 2: Verify Data Quality
```bash
# Check dataset statistics
wc -l ../data/train/train_data.json
wc -l ../data/val/val_data.json
wc -l ../data/test/test_data.json

# Inspect samples
head -n 10 ../data/train/train_data.txt
```

## 🎯 Training Options

### 1. Easy Training Launcher (Recommended)

**Purpose**: Simple launcher with automatic GPU detection and selection.

**Script**: `launch_training.py`

**Features**:
- Automatic GPU detection and selection
- Single GPU and multi-GPU support
- Data availability checking
- GPU status display
- Confirmation prompts

**Usage**:
```bash
# Check GPU status
python launch_training.py --status

# Train with 1 GPU (single GPU training)
python launch_training.py --experts 128 --gpus 1

# Train with 2 GPUs (distributed training)
python launch_training.py --experts 128 --gpus 2

# Train with 4 GPUs (distributed training)
python launch_training.py --experts 256 --gpus 4 --num_epochs 5
```

### 2. Distributed Training (Multi-GPU)

**Purpose**: Multi-GPU distributed training with automatic GPU selection.

**Script**: `train_switch_distributed.py`

**Features**:
- Intelligent GPU selection based on memory and utilization
- Automatic fallback to fewer GPUs if requested count unavailable
- Linear learning rate scaling for multi-GPU
- Distributed data loading and synchronization
- Proper cleanup and error handling

**Usage**:
```bash
# Auto-select 4 best GPUs
python train_switch_distributed.py --experts 128 --gpus 4

# Train with 2 GPUs, custom settings
python train_switch_distributed.py --experts 256 --gpus 2 \
  --batch_size 4 --learning_rate 4e-6 --num_epochs 5

# Force single GPU training
python train_switch_distributed.py --experts 128 --single_gpu
```

### 3. Single GPU Training (Unified)

**Purpose**: Single GPU training with auto-optimized settings.

**Script**: `train_switch_unified.py`

**Available Models**:
- `switch-base-8` (8 experts, ~7.4B params)
- `switch-base-16` (16 experts, ~7.4B params)
- `switch-base-32` (32 experts, ~7.4B params)
- `switch-base-64` (64 experts, ~7.4B params)
- `switch-base-128` (128 experts, ~7.4B params)
- `switch-base-256` (256 experts, ~7.4B params) ← **Largest base model**

**Key Features**:
- Auto-optimized settings per expert count
- Router weight stabilization
- FP32 training for maximum stability
- Gradient clipping and NaN detection
- Progressive difficulty handling

**Usage**:
```bash
# Train Switch with 128 experts (auto-optimized)
python train_switch_unified.py --experts 128

# Train Switch with 256 experts (largest base model)
python train_switch_unified.py --experts 256

# Override auto-optimization if needed
python train_switch_unified.py --experts 128 \
  --batch_size 4 \
  --learning_rate 3e-6 \
  --num_epochs 5
```

**Expert-Specific Auto-Optimization**:
| Experts | Batch Size | Learning Rate | Grad Accum | Warmup | Max Grad Norm |
|---------|------------|---------------|------------|---------|---------------|
| 8       | 6          | 5e-6         | 2          | 0.1     | 1.0          |
| 16      | 5          | 4e-6         | 3          | 0.1     | 1.0          |
| 32      | 4          | 3e-6         | 4          | 0.12    | 0.8          |
| 64      | 4          | 3e-6         | 4          | 0.12    | 0.8          |
| 128     | 3          | 2e-6         | 5          | 0.15    | 0.6          |
| 256     | 3          | 2e-6         | 6          | 0.15    | 0.5          |

### 2. Legacy Training Scripts

**For specific models**:
- `train_switch_128.py` - Switch-128 specific
- `train_switch_256.py` - Switch-256 specific
- `train_switch_stabilized.py` - Original stabilized training

### 3. Alternative Models

**Small MoE**: `train_small_moe.py`
```bash
python train_small_moe.py --batch_size 4 --learning_rate 2e-5 --num_epochs 3
```

**Mixtral MoE**: `train_mixtral_moe.py`
```bash
python train_mixtral_moe.py --model_name mistralai/Mixtral-8x7B-v0.1 --batch_size 1
```

## 🎛️ Hardware-Specific Configurations

### Single GPU Systems

#### RTX 3090 (24GB VRAM)
```bash
# Conservative settings for stability
python launch_training.py --experts 32 --gpus 1 \
  --batch_size 2 --learning_rate 4e-6 --num_epochs 3
```

#### A6000 (48GB VRAM)
```bash
# Larger model with optimized settings
python launch_training.py --experts 128 --gpus 1 \
  --num_epochs 5 --disable_wandb
```

#### A100 (80GB VRAM)
```bash
# Largest base model
python launch_training.py --experts 256 --gpus 1 \
  --num_epochs 5
```

### Multi-GPU Systems (Distributed Training)

#### 2x RTX 3090 (48GB total)
```bash
# Distributed training with moderate model
python launch_training.py --experts 64 --gpus 2 \
  --num_epochs 5 --disable_wandb
```

#### 2x A6000 (96GB total)
```bash
# Large model distributed training
python launch_training.py --experts 128 --gpus 2 \
  --num_epochs 5 --disable_wandb
```

#### 4x A6000 (192GB total)
```bash
# Largest model with maximum parallelism
python launch_training.py --experts 256 --gpus 4 \
  --num_epochs 5 --disable_wandb
```

#### 4x A100 (320GB total)
```bash
# Maximum performance configuration
python launch_training.py --experts 256 --gpus 4 \
  --num_epochs 5 --batch_size 4
```

### Advanced Multi-GPU Configuration
```bash
# Manual GPU selection and custom settings
python train_switch_distributed.py --experts 128 --gpus 4 \
  --batch_size 3 --learning_rate 6e-6 --num_epochs 5 \
  --gradient_accumulation_steps 2 --disable_wandb
```

## 📊 Evaluation Methods

### 1. Simple Evaluation (Recommended First)

**Purpose**: Quick assessment of model functionality and basic metrics.

**Script**: `simple_switch_eval.py`

**Usage**:
```bash
python simple_switch_eval.py --model_path ../models/switch_128_experts
```

**What it measures**:
- Basic text generation capability
- Perplexity on test data
- Model loading verification
- Generation success rate

### 2. Comprehensive Evaluation

**Purpose**: Detailed analysis including domain-specific understanding.

**Script**: `evaluate_model.py`

**Usage**:
```bash
python evaluate_model.py \
  --model_path ../models/switch_128_experts \
  --model_type switch
```

**What it measures**:
- Perplexity with proper seq2seq handling
- Research domain concept understanding
- Text generation quality assessment
- Model parameter statistics

**Supported Model Types**:
- `switch`: Switch Transformer models
- `small_moe`: Custom small MoE models
- `mixtral`: Mixtral-based models

## 📈 Model Specifications

### Switch Transformer Models Comparison

| Model | Experts | Parameters | Memory Req | Recommended Hardware |
|-------|---------|------------|------------|---------------------|
| `switch-base-8` | 8 | ~7.4B | ~8GB | RTX 3090+ |
| `switch-base-16` | 16 | ~7.4B | ~10GB | RTX 3090+ |
| `switch-base-32` | 32 | ~7.4B | ~14GB | RTX 3090+ |
| `switch-base-64` | 64 | ~7.4B | ~20GB | A6000+ |
| `switch-base-128` | 128 | ~7.4B | ~32GB | A6000+ |
| `switch-base-256` | 256 | ~7.4B | ~48GB | A6000+ |

### Training Time Estimates (30K samples)

| Model | Hardware | Estimated Time |
|-------|----------|----------------|
| switch-base-8 | RTX 3090 | 45-90 min |
| switch-base-32 | RTX 3090 | 60-120 min |
| switch-base-128 | A6000 | 90-180 min |
| switch-base-256 | A6000 | 120-240 min |

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. PDF Processing Issues
**NLTK punkt_tab not found**:
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

**PDFs not processing**:
```bash
pip install PyPDF2 PyMuPDF pdfplumber
```

#### 2. Training Issues

**NaN Loss During Training**:
- Router weights become unstable
- **Solution**: Use unified script (auto-applies stabilization)
- **Manual fix**: Reduce learning rate, ensure FP32 training

**CUDA Out of Memory**:
```bash
# Reduce batch size
python train_switch_unified.py --experts 128 --batch_size 2

# Use smaller model
python train_switch_unified.py --experts 64
```

**Multi-GPU Issues**:
```bash
# Force single GPU
CUDA_VISIBLE_DEVICES=0 python train_switch_unified.py --experts 128
```

**Wandb Import Error**:
```bash
# Disable wandb
python train_switch_unified.py --experts 128 --disable_wandb
```

#### 3. Data Issues

**Empty Dataset**:
```bash
# Check if PDFs were processed
ls -la ../data/train/
wc -l ../data/train/train_data.json
```

**Poor Quality Data**:
```bash
# Re-process PDFs with better filtering
python process_pdfs.py --raw_pdf_dir ../data/raw_pdfs --output_dir ../data
```

## 📋 Results Interpretation

### Dataset Statistics (Current)
- **Total PDFs**: ~166 research papers
- **Training samples**: ~30,000 text segments
- **Validation samples**: ~3,000 text segments
- **Test samples**: ~3,000 text segments

### Perplexity Guidelines

| Perplexity Range | Quality | Interpretation |
|------------------|---------|----------------|
| 10-40 | Excellent | Well-trained, coherent generation |
| 40-80 | Good | Reasonable performance |
| 80-200 | Poor | Undertrained, needs more epochs |
| 200+ | Very Poor | Severe training issues |

### Generation Quality Assessment

**Good Signs**:
- Coherent research paper language
- Proper technical terminology
- Logical flow of ideas
- Minimal repetition

**Warning Signs**:
- Repetitive phrases
- Broken grammar
- Non-English output
- Random symbols

## 🔄 Complete Workflow Example

### Full Training Pipeline

1. **Process PDFs**:
```bash
python process_pdfs.py --raw_pdf_dir ../data/raw_pdfs --output_dir ../data
```

2. **Train Switch Model**:
```bash
# For A6000
CUDA_VISIBLE_DEVICES=0 python train_switch_unified.py --experts 128 --disable_wandb
```

3. **Quick Evaluation**:
```bash
python simple_switch_eval.py --model_path ../models/switch_128_experts
```

4. **Comprehensive Evaluation**:
```bash
python evaluate_model.py \
  --model_path ../models/switch_128_experts \
  --model_type switch
```

5. **Review Results**:
```bash
ls ../evaluations/
cat ../evaluations/simple_switch_eval_*.txt
```

## 🗂️ File Structure

```
switch_training/
├── scripts/
│   ├── process_pdfs.py              # PDF processing pipeline
│   ├── train_switch_unified.py      # Main training script (NEW)
│   ├── train_switch_128.py          # Switch-128 specific
│   ├── train_switch_256.py          # Switch-256 specific
│   ├── train_switch_stabilized.py   # Original stabilized training
│   ├── simple_switch_eval.py        # Quick evaluation
│   ├── evaluate_model.py            # Comprehensive evaluation
│   └── train_small_moe.py          # Alternative MoE training
├── data/
│   ├── raw_pdfs/                    # Original PDF files
│   ├── processed/                   # Individual PDF processing
│   ├── train/train_data.json        # Training dataset
│   ├── val/val_data.json           # Validation dataset
│   └── test/test_data.json         # Test dataset
├── models/                          # Saved models (gitignored)
├── evaluations/                     # Evaluation results
└── TRAINING_GUIDE.md               # This document
```

## 🛠️ Advanced Usage

### Custom Data Processing
```bash
# Process with custom settings
python process_pdfs.py \
  --raw_pdf_dir ../data/raw_pdfs \
  --output_dir ../data \
  --max_length 1024 \
  --min_length 50
```

### Multi-Model Training
```bash
# Train multiple models in sequence
for experts in 32 64 128; do
  python train_switch_unified.py --experts $experts --disable_wandb
done
```

### Batch Evaluation
```bash
# Evaluate all trained models
for model_dir in ../models/switch_*_experts; do
  python simple_switch_eval.py --model_path $model_dir
done
```

## 📚 Additional Resources

- **Switch Transformer Paper**: [Switch Transformer: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- **HuggingFace Documentation**: [Switch Transformers](https://huggingface.co/docs/transformers/model_doc/switch_transformers)
- **Available Models**: [Switch Transformer Collection](https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f)

## 🐛 Support

If you encounter issues:

1. Check the auto-generated `training_info.json` in model output directory
2. Review error logs in the `logs/` subdirectory
3. Verify hardware compatibility and memory requirements
4. Test with smaller expert counts first
5. Ensure all dependencies are correctly installed

---

*Last updated: July 14, 2025*