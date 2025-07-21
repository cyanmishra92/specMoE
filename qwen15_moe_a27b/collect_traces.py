#!/usr/bin/env python3
"""
Simplified Qwen Trace Collection Launcher
Choose collection size and automatic RTX 3090 optimization
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Collect Qwen traces with RTX 3090 optimization')
    parser.add_argument('size', choices=['small', 'medium', 'large'], 
                       help='Collection size: small(10), medium(1000), large(20000) traces')
    parser.add_argument('--traces', type=int, help='Custom number of traces (overrides size)')
    parser.add_argument('--shard', action='store_true', help='Create training shards for RTX 3090')
    parser.add_argument('--shard-size', type=int, default=500, help='Traces per shard')
    
    args = parser.parse_args()
    
    # Determine script and default traces
    if args.size == 'small':
        script = 'scripts/collection/collect_qwen15_moe_traces_small.py'
        traces = 10
        print(f"🔬 Small collection: {traces} traces (~30 seconds)")
    elif args.size == 'medium':
        script = 'scripts/collection/collect_qwen15_moe_traces_medium.py' 
        traces = args.traces or 1000
        print(f"📊 Medium collection: {traces} traces (~5-10 minutes)")
    else:  # large
        script = 'scripts/collection/collect_qwen15_moe_traces.py'
        traces = 20000
        print(f"🚀 Large collection: {traces} traces (~20-30 minutes)")
    
    # Build command
    cmd = [sys.executable, script]
    
    if args.size == 'medium':
        cmd.extend(['--target_traces', str(traces)])
        if args.shard:
            cmd.append('--shard_data')
            cmd.extend(['--shard_size_mb', '400'])  # RTX 3090 optimized
            print(f"🧩 Creating shards for RTX 3090 training")
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 50)
    
    # Run collection
    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 50)
        print("✅ Collection completed successfully!")
        
        if args.shard and args.size == 'medium':
            print("📁 Training shards created in routing_data/")
            print("🎯 Ready for RTX 3090 training with:")
            print("   python scripts/train_multi_expert_predictor.py --use_shards routing_data/shards")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Collection failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()