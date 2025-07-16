#!/usr/bin/env python3
"""
Example of memory-efficient training on RTX 3090 using sharded data
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and log the output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    else:
        print("❌ Error!")
        if result.stderr:
            print("Error:")
            print(result.stderr)
        return False
    
    return True

def main():
    """Example workflow for RTX 3090 training"""
    
    print("🔥 RTX 3090 Memory-Efficient Training Example")
    print("=" * 60)
    
    # 1. Collect traces with sharding
    print("\n1. Collecting traces with automatic sharding...")
    
    collect_cmd = [
        "python", "../collection/collect_qwen15_moe_traces_medium.py",
        "--target_traces", "1000",  # Smaller number for example
        "--output_suffix", "rtx3090",
        "--shard_data",  # Enable sharding
        "--shard_size_mb", "400"  # 400MB shards for RTX 3090
    ]
    
    if not run_command(collect_cmd, "Collecting and sharding traces"):
        return
    
    # 2. Train with sharded data
    print("\n2. Training with sharded data...")
    
    train_cmd = [
        "python", "../train_multi_expert_predictor.py",
        "--shard_dir", "routing_data/qwen15_moe_a27b_traces_medium_rtx3090_shards",
        "--batch_size", "4",  # Smaller batch size for RTX 3090
        "--epochs", "10",
        "--lr", "1e-4",
        "--device", "cuda",
        "--save_dir", "checkpoints/rtx3090_experiment"
    ]
    
    if not run_command(train_cmd, "Training with sharded data"):
        return
    
    # 3. Show memory usage tips
    print("\n3. RTX 3090 Memory Optimization Tips:")
    print("=" * 60)
    print("""
    ✅ Tips for successful training on RTX 3090 (24GB):
    
    1. **Sharding**: Always use --shard_data flag
       - Target shard size: 400-500MB
       - Keeps memory usage predictable
    
    2. **Batch Size**: Use smaller batch sizes
       - Recommended: 4-8 for medium models
       - Use gradient accumulation for effective larger batches
    
    3. **Model Configuration**:
       - Use mixed precision training
       - Enable gradient checkpointing
       - Clear GPU cache periodically
    
    4. **Monitoring**: Watch GPU memory usage
       - nvidia-smi to monitor usage
       - torch.cuda.empty_cache() periodically
    
    5. **Data Loading**: Use fewer workers
       - num_workers=1 or 2 max
       - Reduces CPU memory pressure
    """)
    
    # 4. Show file sizes
    print("\n4. Checking generated files:")
    print("=" * 60)
    
    # List sharded files
    shard_dir = Path("routing_data/qwen15_moe_a27b_traces_medium_rtx3090_shards")
    if shard_dir.exists():
        print(f"📁 Sharded files in {shard_dir}:")
        for file in sorted(shard_dir.glob("*.pkl")):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.1f} MB")
    
    # List checkpoints
    checkpoint_dir = Path("checkpoints/rtx3090_experiment")
    if checkpoint_dir.exists():
        print(f"🏆 Checkpoints in {checkpoint_dir}:")
        for file in sorted(checkpoint_dir.glob("*.pth")):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.1f} MB")
    
    print("\n🎉 RTX 3090 training example completed!")
    print("📊 Ready for expert speculation training on your GPU!")

if __name__ == "__main__":
    main()