#!/bin/bash
"""
Setup Conda Environment for Enhanced Pre-gated MoE
Creates a complete environment with all dependencies
"""

set -e  # Exit on any error

echo "🚀 Setting up Enhanced Pre-gated MoE Conda Environment"
echo "====================================================="

# Environment name
ENV_NAME="specmoe"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "🗑️  Removing existing ${ENV_NAME} environment..."
    conda env remove -n ${ENV_NAME} -y
fi

echo "📦 Creating new conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.9 -y

echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "📚 Installing PyTorch with CUDA support..."
# Install PyTorch with CUDA 11.8 (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "🤗 Installing Transformers and ML libraries..."
pip install transformers>=4.21.0
pip install accelerate>=0.20.0
pip install datasets>=2.0.0

echo "📊 Installing data science libraries..."
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install scikit-learn>=1.0.0

echo "🔧 Installing utility libraries..."
pip install tqdm>=4.64.0
pip install psutil>=5.8.0
pip install pathlib

echo "🧪 Installing development tools..."
pip install pytest>=7.0.0
pip install black>=22.0.0
pip install isort>=5.10.0

echo "📈 Installing optional visualization..."
pip install plotly>=5.0.0
pip install wandb  # For experiment tracking (optional)

echo "🔍 Verifying installation..."
python -c "
import torch
import transformers
import datasets
import numpy as np
import matplotlib.pyplot as plt
import sklearn
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU device:', torch.cuda.get_device_name(0))
    print('✅ GPU memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('✅ Transformers version:', transformers.__version__)
print('✅ All dependencies verified!')
"

echo ""
echo "🎉 Environment setup complete!"
echo "====================================="
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test the installation:"
echo "  conda activate ${ENV_NAME}"
echo "  cd /path/to/specMoE"
echo "  python scripts/check_current_status.py"
echo ""
echo "To run the complete pipeline:"
echo "  python scripts/pipelines/run_working_pipeline.py --use-128-experts"
echo ""
echo "Environment name: ${ENV_NAME}"
echo "Python version: $(python --version)"
echo "Conda environment path: $(conda info --envs | grep ${ENV_NAME} | awk '{print $2}')"