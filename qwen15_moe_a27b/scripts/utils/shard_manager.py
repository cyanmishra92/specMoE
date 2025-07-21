#!/usr/bin/env python3
"""
Data Shard Manager for RTX 3090 Memory-Efficient Training
Handles trace data sharding, loading, and memory management
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Iterator
import torch
import gc
import logging

logger = logging.getLogger(__name__)

class ShardManager:
    """Manages data shards for memory-efficient training on RTX 3090"""
    
    def __init__(self, shard_dir: str = "routing_data/shards", max_memory_gb: float = 20.0):
        self.shard_dir = Path(shard_dir)
        self.max_memory_gb = max_memory_gb
        self.shard_files = []
        self.shard_metadata = {}
        self._scan_shards()
    
    def _scan_shards(self):
        """Scan for available shard files"""
        if not self.shard_dir.exists():
            logger.warning(f"Shard directory {self.shard_dir} does not exist")
            return
        
        # Find all shard files
        shard_files = sorted(self.shard_dir.glob("shard_*.pkl"))
        metadata_files = sorted(self.shard_dir.glob("shard_*_metadata.json"))
        
        for shard_file in shard_files:
            shard_id = self._extract_shard_id(shard_file.name)
            self.shard_files.append(shard_file)
            
            # Load metadata if available
            metadata_file = self.shard_dir / f"shard_{shard_id:03d}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.shard_metadata[shard_id] = json.load(f)
        
        logger.info(f"Found {len(self.shard_files)} shards in {self.shard_dir}")
    
    def _extract_shard_id(self, filename: str) -> int:
        """Extract shard ID from filename"""
        parts = filename.split('_')
        return int(parts[1])
    
    def get_shard_info(self) -> List[Dict]:
        """Get information about all available shards"""
        info = []
        for shard_file in self.shard_files:
            shard_id = self._extract_shard_id(shard_file.name)
            file_size_mb = shard_file.stat().st_size / (1024 * 1024)
            
            shard_info = {
                'shard_id': shard_id,
                'file_path': str(shard_file),
                'file_size_mb': file_size_mb,
                'metadata': self.shard_metadata.get(shard_id, {})
            }
            info.append(shard_info)
        
        return info
    
    def load_shard(self, shard_id: int):
        """Load a specific shard"""
        shard_file = self.shard_dir / f"shard_{shard_id:03d}_{self.shard_metadata.get(shard_id, {}).get('num_traces', 'unknown')}_traces.pkl"
        
        if not shard_file.exists():
            # Try alternate naming pattern
            shard_files = list(self.shard_dir.glob(f"shard_{shard_id:03d}_*.pkl"))
            if shard_files:
                shard_file = shard_files[0]
            else:
                raise FileNotFoundError(f"Shard {shard_id} not found")
        
        logger.info(f"Loading shard {shard_id} from {shard_file}")
        with open(shard_file, 'rb') as f:
            traces = pickle.load(f)
        
        return traces
    
    def shard_iterator(self, batch_size: int = 1) -> Iterator:
        """Iterator over shards for memory-efficient training"""
        for shard_file in self.shard_files:
            shard_id = self._extract_shard_id(shard_file.name)
            
            logger.info(f"Loading shard {shard_id} for training...")
            traces = self.load_shard(shard_id)
            
            # Yield traces in batches
            for i in range(0, len(traces), batch_size):
                batch = traces[i:i+batch_size]
                yield batch
            
            # Clean memory after processing shard
            del traces
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Completed shard {shard_id}, memory cleaned")
    
    def create_training_config(self) -> Dict:
        """Create training configuration optimized for RTX 3090"""
        total_traces = sum(metadata.get('num_traces', 0) for metadata in self.shard_metadata.values())
        total_size_mb = sum(info['file_size_mb'] for info in self.get_shard_info())
        
        # RTX 3090 optimized settings
        config = {
            'total_traces': total_traces,
            'total_size_mb': total_size_mb,
            'num_shards': len(self.shard_files),
            'rtx3090_optimized': True,
            'recommended_settings': {
                'batch_size': 16,  # Conservative for RTX 3090
                'max_sequence_length': 256,  # Shorter sequences
                'gradient_accumulation_steps': 4,
                'fp16': True,  # Use mixed precision
                'shard_based_training': True
            },
            'memory_management': {
                'clear_cache_every_shard': True,
                'max_memory_usage_gb': self.max_memory_gb,
                'load_one_shard_at_time': True
            }
        }
        
        return config
    
    def save_training_config(self, output_path: str = "routing_data/rtx3090_training_config.json"):
        """Save RTX 3090 optimized training configuration"""
        config = self.create_training_config()
        
        config_path = Path(output_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"RTX 3090 training config saved to {config_path}")
        return config

def main():
    """Demo/test the shard manager"""
    import argparse
    parser = argparse.ArgumentParser(description='Manage data shards for RTX 3090 training')
    parser.add_argument('--scan', action='store_true', help='Scan and show shard information')
    parser.add_argument('--config', action='store_true', help='Create RTX 3090 training config')
    parser.add_argument('--shard-dir', default='routing_data/shards', help='Shard directory path')
    args = parser.parse_args()
    
    manager = ShardManager(args.shard_dir)
    
    if args.scan:
        print("Available shards:")
        for info in manager.get_shard_info():
            print(f"  Shard {info['shard_id']}: {info['file_size_mb']:.1f}MB, "
                  f"{info['metadata'].get('num_traces', 'unknown')} traces")
    
    if args.config:
        config = manager.save_training_config()
        print(f"Training config created with {config['num_shards']} shards, "
              f"{config['total_traces']} total traces")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()