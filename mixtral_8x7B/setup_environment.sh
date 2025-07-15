#!/bin/bash
# Mixtral 8x7B Environment Setup Script
# Optimized for A100/A6000 GPUs

set -e

echo "🚀 Setting up Mixtral 8x7B MoE Environment"
echo "Optimized for A100/A6000 GPUs"
echo "=========================================="

# Check GPU and CUDA
echo "📊 Checking GPU and CUDA availability..."
nvidia-smi
echo ""
echo "🔧 CUDA Environment:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA Version: $(nvcc --version | grep "release" | cut -d' ' -f5-6)"
echo ""

# Check Python version
echo "🐍 Python version:"
python --version
echo ""

# Create conda environment if needed
ENV_NAME="mixtral_moe"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "📦 Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.11 -y
else
    echo "✅ Conda environment $ENV_NAME already exists"
fi

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
# Auto-detect CUDA version and install appropriate PyTorch
if [[ "$CUDA_HOME" == *"11.8"* ]]; then
    echo "Installing PyTorch for CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_HOME" == *"12.1"* ]]; then
    echo "Installing PyTorch for CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch for CUDA 11.8 (default)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install Transformers and related packages
echo "🤗 Installing Transformers and dependencies..."
pip install transformers>=4.35.0
pip install accelerate>=0.20.0
pip install datasets>=2.14.0
pip install bitsandbytes>=0.41.0

# Install additional ML packages
echo "📊 Installing additional ML packages..."
pip install numpy>=1.24.0
pip install scipy>=1.11.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0

# Install progress and logging
echo "📈 Installing progress and logging tools..."
pip install tqdm>=4.65.0
pip install tensorboard>=2.13.0
pip install wandb>=0.15.0

# Install Jupyter for analysis
echo "📓 Installing Jupyter..."
pip install jupyter>=1.0.0
pip install ipykernel>=6.25.0

# Install system monitoring
echo "🖥️ Installing system monitoring..."
pip install psutil>=5.9.0
pip install GPUtil>=1.4.0
pip install nvidia-ml-py3>=7.352.0

# Test installations
echo "🧪 Testing installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import bitsandbytes; print('BitsAndBytes: OK')"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p routing_data
mkdir -p logs
mkdir -p results

# Set up Hugging Face CLI
echo "🤗 Setting up Hugging Face CLI..."
pip install huggingface_hub>=0.17.0

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate $ENV_NAME"
echo "2. Login to Hugging Face: huggingface-cli login"
echo "3. Run trace collection: python scripts/collection/collect_mixtral_traces.py"
echo ""
echo "GPU Requirements:"
echo "- RTX 3090: 24GB VRAM (may need CPU offload)"
echo "- A6000: 48GB VRAM (recommended)"
echo "- A100: 40GB/80GB VRAM (optimal)"
echo ""
echo "🎯 Ready for Mixtral 8x7B MoE trace collection!"