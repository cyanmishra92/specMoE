# 🚀 Installation Guide - Fresh Setup

## Quick Setup (Recommended)

### Option 1: Conda Environment (Recommended)
```bash
# Make executable and run
chmod +x setup_conda_env.sh
./setup_conda_env.sh

# Activate environment
conda activate specmoe

# Test installation
python scripts/check_current_status.py
```

### Option 2: Pip Installation
```bash
# Create virtual environment
python -m venv specmoe_env
source specmoe_env/bin/activate  # Linux/Mac
# specmoe_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Test installation
python scripts/check_current_status.py
```

## Fresh Directory Test

To test from scratch in a new directory:

### 1. Clone/Copy Project
```bash
# Copy entire project to new location
cp -r /path/to/specMoE /new/test/location/
cd /new/test/location/specMoE
```

### 2. Setup Environment
```bash
# Run conda setup
./setup_conda_env.sh
conda activate specmoe
```

### 3. Test Working Pipeline
```bash
# Quick test (no 128-expert model)
python scripts/pipelines/run_working_pipeline.py

# Full test (128-expert model - takes time)
python scripts/pipelines/run_working_pipeline.py --use-128-experts
```

## Dependencies

### Core Requirements
- Python 3.9+
- PyTorch 2.0+ with CUDA support
- Transformers 4.21+
- CUDA 11.8+ (for GPU acceleration)

### Hardware Requirements
- NVIDIA GPU with 16GB+ VRAM (RTX 3090 recommended)
- 32GB+ RAM recommended
- 50GB+ disk space for models and data

### Software Requirements
- Conda or Python 3.9+
- Git (for cloning)
- CUDA toolkit (for GPU support)

## Verification

After installation, verify everything works:

```bash
python -c "
import torch
import transformers
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
print('✅ Transformers:', transformers.__version__)
"
```

Expected output:
```
✅ PyTorch: 2.x.x
✅ CUDA available: True
✅ GPU: NVIDIA GeForce RTX 3090
✅ Transformers: 4.x.x
```

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Install correct PyTorch for your CUDA
# Visit: https://pytorch.org/get-started/locally/
```

### Memory Issues
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Reduce batch size in scripts if needed
```

### Import Errors
```bash
# Reinstall in environment
conda activate specmoe
pip install --upgrade transformers torch

# Check Python path
python -c "import sys; print(sys.path)"
```

## Ready to Use

After successful installation:

1. **Quick test**: `python scripts/pipelines/run_working_pipeline.py`
2. **Full pipeline**: `python scripts/pipelines/run_working_pipeline.py --use-128-experts`
3. **Status check**: `python scripts/check_current_status.py`

The setup script creates a complete environment ready for the 128-expert speculation pipeline!