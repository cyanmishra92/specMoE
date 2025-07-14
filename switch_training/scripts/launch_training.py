#!/usr/bin/env python3
"""
Training Launcher Script

Easy-to-use launcher for Switch Transformer training with automatic GPU selection.
Provides both single-GPU and distributed training options.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_gpu_count():
    """Get number of available GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], 
                              capture_output=True, text=True, check=True)
        return len(result.stdout.strip().split('\n'))
    except:
        return 0

def show_gpu_status():
    """Display GPU status"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        print("\n📊 GPU Status:")
        print("=" * 80)
        print(f"{'GPU':<3} {'Name':<25} {'Memory Used':<12} {'Memory Total':<12} {'Utilization':<12}")
        print("-" * 80)
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_idx = parts[0]
                    name = parts[1][:24]  # Truncate long names
                    mem_used = f"{parts[2]}MB"
                    mem_total = f"{parts[3]}MB"
                    util = f"{parts[4]}%"
                    print(f"{gpu_idx:<3} {name:<25} {mem_used:<12} {mem_total:<12} {util:<12}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")

def check_data_availability():
    """Check if training data is available"""
    data_dir = Path("../data")
    train_file = data_dir / "train" / "train_data.json"
    val_file = data_dir / "val" / "val_data.json"
    test_file = data_dir / "test" / "test_data.json"
    
    if not train_file.exists():
        logger.error(f"Training data not found at {train_file}")
        logger.info("Please run: python process_pdfs.py --raw_pdf_dir ../data/raw_pdfs --output_dir ../data")
        return False
    
    # Check data size
    try:
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        with open(val_file, 'r') as f:
            val_data = json.load(f)
        
        logger.info(f"✅ Data found: {len(train_data):,} train samples, {len(val_data):,} val samples")
        return True
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

def launch_training(args):
    """Launch training with appropriate script"""
    
    # Check data availability
    if not check_data_availability():
        return False
    
    # Show GPU status
    show_gpu_status()
    
    # Determine which script to use
    if args.gpus == 1 or args.force_single:
        logger.info("🚀 Launching single GPU training...")
        script = "train_switch_unified.py"
        cmd = [
            "python", script,
            "--experts", str(args.experts)
        ]
        
        if args.disable_wandb:
            cmd.append("--disable_wandb")
        
        if args.num_epochs != 3:
            cmd.extend(["--num_epochs", str(args.num_epochs)])
        
        if args.batch_size:
            cmd.extend(["--batch_size", str(args.batch_size)])
        
        if args.learning_rate:
            cmd.extend(["--learning_rate", str(args.learning_rate)])
        
        if args.output_dir:
            cmd.extend(["--output_dir", args.output_dir])
        
    else:
        logger.info(f"🚀 Launching distributed training on {args.gpus} GPUs...")
        script = "train_switch_distributed.py"
        cmd = [
            "python", script,
            "--experts", str(args.experts),
            "--gpus", str(args.gpus)
        ]
        
        if args.disable_wandb:
            cmd.append("--disable_wandb")
        
        if args.num_epochs != 3:
            cmd.extend(["--num_epochs", str(args.num_epochs)])
        
        if args.batch_size:
            cmd.extend(["--batch_size", str(args.batch_size)])
        
        if args.learning_rate:
            cmd.extend(["--learning_rate", str(args.learning_rate)])
        
        if args.output_dir:
            cmd.extend(["--output_dir", args.output_dir])
    
    # Show command
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Ask for confirmation
    if not args.yes:
        response = input("\nProceed with training? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Training cancelled.")
            return False
    
    # Launch training
    try:
        logger.info("Starting training...")
        result = subprocess.run(cmd, check=True)
        logger.info("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Switch Transformer Training Launcher")
    parser.add_argument("--experts", type=int, choices=[8, 16, 32, 64, 128, 256], 
                        help="Number of experts (8, 16, 32, 64, 128, 256)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--force-single", action="store_true", help="Force single GPU training")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size (auto-optimized if not provided)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (auto-optimized if not provided)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--output_dir", type=str, help="Output directory (auto-generated if not provided)")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm training start")
    parser.add_argument("--status", action="store_true", help="Show GPU status and exit")
    
    args = parser.parse_args()
    
    # Show GPU status if requested
    if args.status:
        show_gpu_status()
        return
    
    # Check if experts is provided (required for training)
    if args.experts is None:
        logger.error("--experts is required for training")
        return
    
    # Check GPU availability
    total_gpus = get_gpu_count()
    if total_gpus == 0:
        logger.error("No GPUs found!")
        return
    
    logger.info(f"Found {total_gpus} GPUs")
    
    # Validate GPU count
    if args.gpus > total_gpus:
        logger.error(f"Requested {args.gpus} GPUs but only {total_gpus} available")
        return
    
    # Show configuration
    print("\n🎯 Training Configuration:")
    print("=" * 50)
    print(f"Model: Switch Transformer with {args.experts} experts")
    print(f"GPUs: {args.gpus} {'(single)' if args.gpus == 1 else '(distributed)'}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {'Auto-optimized' if not args.batch_size else args.batch_size}")
    print(f"Learning rate: {'Auto-optimized' if not args.learning_rate else args.learning_rate}")
    print(f"Wandb: {'Disabled' if args.disable_wandb else 'Enabled'}")
    print("=" * 50)
    
    # Launch training
    success = launch_training(args)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("Next steps:")
        print("1. Check training logs in the output directory")
        print("2. Run evaluation: python simple_switch_eval.py --model_path <output_dir>")
        print("3. Run comprehensive evaluation: python evaluate_model.py --model_path <output_dir> --model_type switch")
    else:
        print("\n❌ Training failed or was cancelled.")

if __name__ == "__main__":
    main()