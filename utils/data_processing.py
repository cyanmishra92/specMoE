#!/usr/bin/env python3
"""
Data Processing Utilities
Clean data loading and processing for MoE speculation training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class SpeculationDataset(Dataset):
    """Clean dataset for MoE speculation training"""
    
    def __init__(self, traces, max_seq_len=32, split='train', use_top_k=True):
        self.traces = []
        self.max_seq_len = max_seq_len
        self.split = split
        self.use_top_k = use_top_k
        
        # Group by sample_id to avoid data leakage
        sample_groups = {}
        for trace_dict in traces:
            sample_id = trace_dict['sample_id']
            if sample_id not in sample_groups:
                sample_groups[sample_id] = []
            sample_groups[sample_id].append(trace_dict)
        
        # Split by sample groups
        sample_ids = list(sample_groups.keys())
        train_ids, test_ids = train_test_split(sample_ids, test_size=0.3, random_state=42)
        train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=42)
        
        if split == 'train':
            selected_ids = train_ids
        elif split == 'val':
            selected_ids = val_ids
        elif split == 'test':
            selected_ids = test_ids
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Process traces
        for sample_id in selected_ids:
            for trace_dict in sample_groups[sample_id]:
                processed = self._process_trace(trace_dict)
                if processed is not None:
                    self.traces.append(processed)
        
        logger.info(f"Created {split} dataset: {len(self.traces)} samples from {len(selected_ids)} sequences")
    
    def _process_trace(self, trace_dict):
        """Clean trace processing"""
        try:
            # Convert to tensors
            hidden_states = torch.from_numpy(trace_dict['hidden_states']).float()
            target_routing = torch.from_numpy(trace_dict['target_routing']).float()
            
            # Handle dimensions
            if hidden_states.dim() > 2:
                hidden_states = hidden_states.squeeze(0)
            if target_routing.dim() > 2:
                target_routing = target_routing.squeeze(0)
            
            seq_len = min(hidden_states.size(0), self.max_seq_len)
            hidden_states = hidden_states[:seq_len]
            target_routing = target_routing[:seq_len]
            
            # Extract expert targets
            if self.use_top_k:
                # Use top-2 for MoE (primary expert)
                top_k_values, top_k_indices = torch.topk(target_routing, k=2, dim=-1)
                expert_targets = top_k_indices[:, 0]  # Primary expert
            else:
                # Use argmax
                expert_targets = torch.argmax(target_routing, dim=-1)
            
            # Create previous layer context
            num_experts = target_routing.size(-1)
            if 'prev_layer_gates' in trace_dict and trace_dict['prev_layer_gates']:
                prev_gates = trace_dict['prev_layer_gates'][-1]  # Last layer
                prev_gate = torch.from_numpy(prev_gates).float()
                if prev_gate.dim() > 2:
                    prev_gate = prev_gate.squeeze(0)
                prev_gate = prev_gate[:seq_len]
            else:
                # Create meaningful dummy context
                prev_gate = torch.zeros(seq_len, num_experts)
                layer_id = trace_dict.get('layer_id', 1)
                prev_gate[:, layer_id % num_experts] = 0.5
                prev_gate = torch.softmax(prev_gate + torch.randn_like(prev_gate) * 0.1, dim=-1)
            
            return {
                'hidden_states': hidden_states,
                'prev_gate': prev_gate,
                'expert_targets': expert_targets,
                'target_routing': target_routing,
                'seq_len': seq_len,
                'layer_id': trace_dict.get('layer_id', 1),
                'sample_id': trace_dict['sample_id']
            }
            
        except Exception as e:
            logger.warning(f"Error processing trace: {e}")
            return None
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        
        # Pad to max length
        hidden_states = torch.zeros(self.max_seq_len, trace['hidden_states'].size(-1))
        prev_gate = torch.zeros(self.max_seq_len, trace['prev_gate'].size(-1))
        expert_targets = torch.full((self.max_seq_len,), -100, dtype=torch.long)  # Ignore padding
        
        seq_len = trace['seq_len']
        hidden_states[:seq_len] = trace['hidden_states']
        prev_gate[:seq_len] = trace['prev_gate']
        expert_targets[:seq_len] = trace['expert_targets']
        
        # Create attention mask
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:seq_len] = True
        
        return {
            'hidden_states': hidden_states,
            'prev_gate': prev_gate,
            'expert_targets': expert_targets,
            'mask': mask,
            'layer_id': trace['layer_id'],
            'sample_id': trace['sample_id']
        }

def load_traces(trace_file="routing_data/robust_traces.pkl"):
    """Load traces safely without reading full content"""
    logger.info(f"Loading traces from {trace_file}")
    
    if not Path(trace_file).exists():
        logger.error(f"Trace file not found: {trace_file}")
        return None
    
    # Get file size for info
    file_size = Path(trace_file).stat().st_size / (1024**2)  # MB
    logger.info(f"Trace file size: {file_size:.1f} MB")
    
    with open(trace_file, 'rb') as f:
        traces = pickle.load(f)
    
    logger.info(f"Loaded {len(traces)} traces")
    
    # Log basic info without reading full data
    if len(traces) > 0:
        sample_trace = traces[0]
        logger.info(f"Sample trace keys: {list(sample_trace.keys())}")
        logger.info(f"Hidden states shape: {sample_trace['hidden_states'].shape}")
        logger.info(f"Target routing shape: {sample_trace['target_routing'].shape}")
        logger.info(f"Number of experts: {sample_trace['target_routing'].shape[-1]}")
    
    return traces

def get_data_info(traces):
    """Get dataset information without processing all data"""
    if not traces:
        return {}
    
    # Sample a few traces for info
    sample_traces = traces[:min(10, len(traces))]
    
    hidden_sizes = []
    num_experts_list = []
    seq_lengths = []
    
    for trace in sample_traces:
        hidden_sizes.append(trace['hidden_states'].shape[-1])
        num_experts_list.append(trace['target_routing'].shape[-1])
        seq_lengths.append(trace['hidden_states'].shape[0])
    
    info = {
        'total_traces': len(traces),
        'hidden_size': max(set(hidden_sizes), key=hidden_sizes.count),  # Most common
        'num_experts': max(set(num_experts_list), key=num_experts_list.count),
        'avg_seq_length': np.mean(seq_lengths),
        'max_seq_length': max(seq_lengths),
        'min_seq_length': min(seq_lengths)
    }
    
    return info

def create_datasets(traces, max_seq_len=32, use_top_k=True):
    """Create train/val/test datasets"""
    
    train_dataset = SpeculationDataset(traces, max_seq_len, 'train', use_top_k)
    val_dataset = SpeculationDataset(traces, max_seq_len, 'val', use_top_k)
    test_dataset = SpeculationDataset(traces, max_seq_len, 'test', use_top_k)
    
    return train_dataset, val_dataset, test_dataset

def get_dataset_stats(dataset):
    """Get statistics about a dataset"""
    if len(dataset) == 0:
        return {}
    
    # Sample a few items
    sample_items = [dataset[i] for i in range(min(10, len(dataset)))]
    
    hidden_size = sample_items[0]['hidden_states'].size(-1)
    num_experts = sample_items[0]['prev_gate'].size(-1)
    max_seq_len = sample_items[0]['hidden_states'].size(0)
    
    # Count valid tokens
    total_valid_tokens = 0
    total_tokens = 0
    
    for item in sample_items:
        mask = item['mask']
        total_valid_tokens += mask.sum().item()
        total_tokens += mask.size(0)
    
    avg_valid_ratio = total_valid_tokens / total_tokens if total_tokens > 0 else 0
    
    stats = {
        'dataset_size': len(dataset),
        'hidden_size': hidden_size,
        'num_experts': num_experts,
        'max_seq_len': max_seq_len,
        'avg_valid_token_ratio': avg_valid_ratio
    }
    
    return stats