#!/usr/bin/env python3
"""
Setup script for DeepSeek-MoE-16B Expert Speculation Training
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.20.0",
        "huggingface-hub>=0.17.0",
        "GPUtil>=1.4.0"
    ]
    
    print("Installing requirements...")
    for req in requirements:
        print(f"Installing {req}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])
    
    print("✅ All requirements installed!")

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ Found {gpu_count} GPU(s):")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                
                if memory_gb >= 40:
                    print(f"    ✅ Excellent for DeepSeek-MoE-16B")
                elif memory_gb >= 24:
                    print(f"    ✅ Good with 4-bit quantization")
                elif memory_gb >= 16:
                    print(f"    ⚠️ May work with aggressive quantization")
                else:
                    print(f"    ❌ Insufficient memory")
        else:
            print("❌ No CUDA GPUs found")
    except ImportError:
        print("❌ PyTorch not installed")

def create_directories():
    """Create necessary directories"""
    dirs = [
        "models",
        "routing_data",
        "logs",
        "results"
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def main():
    print("🚀 Setting up DeepSeek-MoE-16B Expert Speculation Training")
    print("=" * 60)
    
    # Install requirements
    install_requirements()
    print()
    
    # Check GPU
    check_gpu()
    print()
    
    # Create directories
    create_directories()
    print()
    
    print("🎉 Setup complete!")
    print()
    print("Next steps:")
    print("1. Login to HuggingFace: huggingface-cli login")
    print("2. Run trace collection: python scripts/collection/collect_deepseek_moe_traces.py")
    print("3. Train speculation model: python scripts/training/train_deepseek_moe_speculation.py")
    print()
    print("Note: DeepSeek-MoE-16B requires 16GB+ VRAM. Use 4-bit quantization on RTX 3090.")

if __name__ == "__main__":
    main()