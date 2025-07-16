#!/usr/bin/env python3
"""
Data sharding utilities for memory-efficient training on RTX 3090 (24GB)
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Tuple
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class QwenMoEGatingDataPoint:
    """Qwen MoE gating data point"""
    layer_id: int
    hidden_states: torch.Tensor
    input_embeddings: torch.Tensor
    target_routing: torch.Tensor
    target_top_k: torch.Tensor
    prev_layer_gates: List[torch.Tensor]
    sequence_length: int
    token_ids: Optional[torch.Tensor]
    dataset_name: str
    sample_id: str

class TraceDataSharder:
    """
    Shard large trace files into smaller chunks for memory-efficient training
    """
    
    def __init__(self, shard_size_mb: int = 500):
        """
        Args:
            shard_size_mb: Target size for each shard in MB
        """
        self.shard_size_mb = shard_size_mb
        self.shard_size_bytes = shard_size_mb * 1024 * 1024
    
    def estimate_trace_size(self, trace) -> int:
        """Estimate memory size of a single trace in bytes"""
        size = 0
        
        # Tensor sizes
        if hasattr(trace, 'hidden_states') and trace.hidden_states is not None:
            size += trace.hidden_states.numel() * trace.hidden_states.element_size()
        
        if hasattr(trace, 'input_embeddings') and trace.input_embeddings is not None:
            size += trace.input_embeddings.numel() * trace.input_embeddings.element_size()
        
        if hasattr(trace, 'target_routing') and trace.target_routing is not None:
            size += trace.target_routing.numel() * trace.target_routing.element_size()
        
        if hasattr(trace, 'target_top_k') and trace.target_top_k is not None:
            size += trace.target_top_k.numel() * trace.target_top_k.element_size()
        
        # Rough estimate for other fields
        size += 1000  # Metadata, strings, etc.
        
        return size
    
    def shard_traces(self, input_file: str, output_dir: str) -> List[str]:
        """
        Shard a large trace file into smaller chunks
        
        Args:
            input_file: Path to input pickle file
            output_dir: Directory to save shards
            
        Returns:
            List of shard file paths
        """
        logger.info(f"Loading traces from {input_file}")
        
        # Load all traces
        with open(input_file, 'rb') as f:
            all_traces = pickle.load(f)
        
        logger.info(f"Loaded {len(all_traces)} traces")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group traces by layer for better locality
        layer_groups = {}
        for trace in all_traces:
            layer_id = trace.layer_id
            if layer_id not in layer_groups:
                layer_groups[layer_id] = []
            layer_groups[layer_id].append(trace)
        
        logger.info(f"Found {len(layer_groups)} layers")
        
        # Create shards
        shard_files = []
        current_shard = []
        current_size = 0
        shard_idx = 0
        
        for layer_id in sorted(layer_groups.keys()):
            layer_traces = layer_groups[layer_id]
            logger.info(f"Processing layer {layer_id} with {len(layer_traces)} traces")
            
            for trace in layer_traces:
                trace_size = self.estimate_trace_size(trace)
                
                # Check if adding this trace would exceed shard size
                if current_size + trace_size > self.shard_size_bytes and current_shard:
                    # Save current shard
                    shard_file = output_path / f"shard_{shard_idx:03d}.pkl"
                    self._save_shard(current_shard, shard_file)
                    shard_files.append(str(shard_file))
                    
                    # Start new shard
                    current_shard = []
                    current_size = 0
                    shard_idx += 1
                
                current_shard.append(trace)
                current_size += trace_size
        
        # Save final shard
        if current_shard:
            shard_file = output_path / f"shard_{shard_idx:03d}.pkl"
            self._save_shard(current_shard, shard_file)
            shard_files.append(str(shard_file))
        
        # Save metadata
        metadata = {
            'total_traces': len(all_traces),
            'num_shards': len(shard_files),
            'shard_size_mb': self.shard_size_mb,
            'layer_groups': {layer_id: len(traces) for layer_id, traces in layer_groups.items()},
            'shard_files': shard_files
        }
        
        metadata_file = output_path / "sharding_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Created {len(shard_files)} shards in {output_dir}")
        logger.info(f"Average shard size: {self.shard_size_mb} MB")
        
        return shard_files
    
    def _save_shard(self, traces: List, shard_file: Path):
        """Save a shard to file"""
        with open(shard_file, 'wb') as f:
            pickle.dump(traces, f)
        
        # Log shard info
        file_size_mb = shard_file.stat().st_size / (1024 * 1024)
        logger.info(f"Saved shard {shard_file.name}: {len(traces)} traces, {file_size_mb:.1f} MB")

class ShardedDataLoader:
    """
    Memory-efficient data loader for sharded trace data
    """
    
    def __init__(self, shard_dir: str, batch_size: int = 8, shuffle: bool = True):
        """
        Args:
            shard_dir: Directory containing sharded data
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
        """
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load metadata
        metadata_file = self.shard_dir / "sharding_metadata.pkl"
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.shard_files = self.metadata['shard_files']
        self.total_traces = self.metadata['total_traces']
        
        logger.info(f"Loaded sharded dataset: {len(self.shard_files)} shards, {self.total_traces} traces")
    
    def __len__(self) -> int:
        """Get total number of traces"""
        return self.total_traces
    
    def get_num_batches(self) -> int:
        """Get number of batches"""
        return math.ceil(self.total_traces / self.batch_size)
    
    def iterate_shards(self) -> Iterator[List]:
        """Iterate over shards"""
        shard_order = list(range(len(self.shard_files)))
        if self.shuffle:
            np.random.shuffle(shard_order)
        
        for shard_idx in shard_order:
            shard_file = self.shard_files[shard_idx]
            logger.debug(f"Loading shard {shard_idx}: {shard_file}")
            
            with open(shard_file, 'rb') as f:
                traces = pickle.load(f)
            
            if self.shuffle:
                np.random.shuffle(traces)
            
            yield traces
    
    def iterate_batches(self) -> Iterator[List]:
        """Iterate over batches across all shards"""
        current_batch = []
        
        for shard_traces in self.iterate_shards():
            for trace in shard_traces:
                current_batch.append(trace)
                
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []
        
        # Yield final batch if not empty
        if current_batch:
            yield current_batch
    
    def get_sample_batch(self, num_samples: int = 10) -> List:
        """Get a small sample batch for testing"""
        for shard_traces in self.iterate_shards():
            return shard_traces[:num_samples]
        return []

class MemoryEfficientTrainer:
    """
    Memory-efficient trainer for RTX 3090 (24GB)
    """
    
    def __init__(self, model, optimizer, criterion, device, gradient_accumulation_steps: int = 4):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss criterion
            device: Training device
            gradient_accumulation_steps: Steps to accumulate gradients (for effective larger batch size)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
    
    def train_epoch(self, data_loader: ShardedDataLoader) -> Dict[str, float]:
        """Train for one epoch with memory efficiency"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        accumulation_step = 0
        
        # Clear gradients at start
        self.optimizer.zero_grad()
        
        for batch in data_loader.iterate_batches():
            # Process batch (this would be dataset-specific)
            # For now, just a placeholder
            batch_loss = self._process_batch(batch)
            
            # Scale loss for gradient accumulation
            scaled_loss = batch_loss / self.gradient_accumulation_steps
            scaled_loss.backward()
            
            accumulation_step += 1
            
            # Update weights after accumulation steps
            if accumulation_step % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            # Clear GPU cache periodically
            if num_batches % 10 == 0:
                torch.cuda.empty_cache()
        
        # Final gradient update if needed
        if accumulation_step % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {'loss': total_loss / num_batches}
    
    def _process_batch(self, batch: List) -> torch.Tensor:
        """Process a batch - placeholder for actual implementation"""
        # This would be implemented based on the specific dataset format
        # For now, return a dummy loss
        return torch.tensor(0.0, device=self.device, requires_grad=True)

def shard_trace_file(input_file: str, output_dir: str, shard_size_mb: int = 500) -> List[str]:
    """
    Convenience function to shard a trace file
    
    Args:
        input_file: Path to input pickle file
        output_dir: Directory to save shards
        shard_size_mb: Target size for each shard in MB
        
    Returns:
        List of shard file paths
    """
    sharder = TraceDataSharder(shard_size_mb=shard_size_mb)
    return sharder.shard_traces(input_file, output_dir)

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Shard trace data for memory-efficient training')
    parser.add_argument('input_file', type=str, help='Input pickle file')
    parser.add_argument('output_dir', type=str, help='Output directory for shards')
    parser.add_argument('--shard_size_mb', type=int, default=500, help='Target shard size in MB')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Shard the data
    shard_files = shard_trace_file(args.input_file, args.output_dir, args.shard_size_mb)
    
    print(f"Created {len(shard_files)} shards in {args.output_dir}")
    
    # Test the sharded data loader
    loader = ShardedDataLoader(args.output_dir, batch_size=4)
    sample_batch = loader.get_sample_batch(5)
    print(f"Sample batch size: {len(sample_batch)}")

if __name__ == "__main__":
    main()