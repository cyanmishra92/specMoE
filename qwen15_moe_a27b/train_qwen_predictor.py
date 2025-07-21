#!/usr/bin/env python3
"""
Simple launcher for Qwen Multi-Expert Predictor Training
"""

import subprocess
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Train Qwen Multi-Expert Predictor')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (RTX 3090 optimized)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs') 
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Build command
    script_path = Path(__file__).parent / 'scripts' / 'train_qwen_multi_expert_predictor.py'
    
    cmd = [
        sys.executable, str(script_path),
        '--batch-size', str(args.batch_size),
        '--epochs', str(args.epochs), 
        '--lr', str(args.lr)
    ]
    
    if args.resume:
        cmd.extend(['--resume', args.resume])
    
    print(f"🚀 Starting Qwen Multi-Expert Predictor Training")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 60)
        print("✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()