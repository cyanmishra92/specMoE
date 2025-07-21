#!/usr/bin/env python3
"""
Create training shards from existing streaming files (avoid OOM)
"""

import pickle
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_shards_from_existing():
    """Create training shards from existing files without loading everything into memory"""
    
    # Input files
    streaming_file = Path("routing_data/streaming/batch_000.pkl")
    checkpoint_file = Path("routing_data/checkpoints/checkpoint_1000_traces.pkl")
    
    # Output directory
    shard_dir = Path("routing_data/shards")
    shard_dir.mkdir(exist_ok=True)
    
    shard_size = 500
    shard_id = 0
    
    logger.info("🧩 Creating training shards from existing files...")
    
    # Process streaming file first
    if streaming_file.exists():
        logger.info(f"Loading {streaming_file}...")
        with open(streaming_file, 'rb') as f:
            streaming_traces = pickle.load(f)
        
        logger.info(f"Found {len(streaming_traces)} traces in streaming file")
        
        # Create shards from streaming traces
        for i in range(0, len(streaming_traces), shard_size):
            shard_traces = streaming_traces[i:i+shard_size]
            
            # Save shard
            shard_file = shard_dir / f"shard_{shard_id:03d}_{len(shard_traces)}_traces.pkl"
            with open(shard_file, 'wb') as f:
                pickle.dump(shard_traces, f)
            
            file_size_mb = shard_file.stat().st_size / (1024 * 1024)
            
            # Save metadata
            metadata = {
                'shard_id': shard_id,
                'num_traces': len(shard_traces),
                'file_size_mb': file_size_mb,
                'rtx3090_optimized': True,
                'source': 'streaming_batch_000'
            }
            
            metadata_file = shard_dir / f"shard_{shard_id:03d}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"  Shard {shard_id}: {len(shard_traces)} traces, {file_size_mb:.1f}MB")
            shard_id += 1
        
        del streaming_traces  # Free memory
    
    # Process checkpoint file
    if checkpoint_file.exists():
        logger.info(f"Loading {checkpoint_file}...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_traces = pickle.load(f)
        
        logger.info(f"Found {len(checkpoint_traces)} traces in checkpoint file")
        
        # Create shards from checkpoint traces
        for i in range(0, len(checkpoint_traces), shard_size):
            shard_traces = checkpoint_traces[i:i+shard_size]
            
            # Save shard
            shard_file = shard_dir / f"shard_{shard_id:03d}_{len(shard_traces)}_traces.pkl"
            with open(shard_file, 'wb') as f:
                pickle.dump(shard_traces, f)
            
            file_size_mb = shard_file.stat().st_size / (1024 * 1024)
            
            # Save metadata
            metadata = {
                'shard_id': shard_id,
                'num_traces': len(shard_traces),
                'file_size_mb': file_size_mb,
                'rtx3090_optimized': True,
                'source': 'checkpoint_1000'
            }
            
            metadata_file = shard_dir / f"shard_{shard_id:03d}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"  Shard {shard_id}: {len(shard_traces)} traces, {file_size_mb:.1f}MB")
            shard_id += 1
    
    # Create training config
    total_shards = shard_id
    config = {
        'total_traces': 3000,
        'shard_size': shard_size,
        'num_shards': total_shards,
        'rtx3090_settings': {
            'batch_size': 16,
            'max_seq_length': 256,
            'fp16': True,
            'gradient_accumulation': 4
        }
    }
    
    config_file = shard_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Created {total_shards} shards for RTX 3090 training")
    logger.info(f"📁 Shards saved in {shard_dir}")
    logger.info(f"🎯 Ready for training with 3000 traces!")

if __name__ == "__main__":
    create_shards_from_existing()