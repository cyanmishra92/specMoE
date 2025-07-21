#!/usr/bin/env python3
"""
Train Qwen Multi-Expert Predictor using Shard-Based Data Loading
Optimized for RTX 3090 with 24GB VRAM
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import time
import gc
from typing import List, Dict, Tuple

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from qwen_multi_expert_predictor import create_qwen_predictor

# Import datapoint class for pickle loading
sys.path.append(os.path.join(os.path.dirname(__file__), 'collection'))
from collect_qwen15_moe_traces_streaming import QwenMoEGatingDataPoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenTraceDataset(Dataset):
    """Dataset for Qwen MoE traces with shard-based loading"""
    
    def __init__(self, shard_files: List[Path], max_seq_len: int = 256):
        self.shard_files = shard_files
        self.max_seq_len = max_seq_len
        self.current_shard_idx = 0
        self.current_shard_data = []
        self.total_traces = 0
        
        # Load metadata to get total trace count
        for shard_file in shard_files:
            metadata_file = shard_file.parent / f"{shard_file.stem}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.total_traces += metadata.get('num_traces', 0)
        
        logger.info(f"Dataset: {len(shard_files)} shards, {self.total_traces} total traces")
        
        # Load first shard
        self._load_shard(0)
    
    def _load_shard(self, shard_idx: int):
        """Load a specific shard into memory"""
        if shard_idx >= len(self.shard_files):
            return False
            
        shard_file = self.shard_files[shard_idx]
        logger.info(f"Loading shard {shard_idx}: {shard_file}")
        
        try:
            with open(shard_file, 'rb') as f:
                self.current_shard_data = pickle.load(f)
            
            self.current_shard_idx = shard_idx
            logger.info(f"Loaded {len(self.current_shard_data)} traces from shard {shard_idx}")
            
            # Clear GPU cache after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load shard {shard_idx}: {e}")
            return False
    
    def __len__(self):
        return self.total_traces
    
    def __getitem__(self, idx):
        """Get trace by global index with automatic shard switching"""
        # Calculate which shard this index belongs to
        traces_per_shard = len(self.current_shard_data) if self.current_shard_data else 500
        target_shard = min(idx // traces_per_shard, len(self.shard_files) - 1)
        
        # Load target shard if not current
        if target_shard != self.current_shard_idx:
            if not self._load_shard(target_shard):
                # Fallback to current shard
                target_shard = self.current_shard_idx
        
        # Get local index within shard
        local_idx = idx % len(self.current_shard_data)
        if local_idx >= len(self.current_shard_data):
            local_idx = len(self.current_shard_data) - 1
        
        trace = self.current_shard_data[local_idx]
        
        # Extract features and targets - handle different tensor shapes
        hidden_states = trace.hidden_states
        target_routing = trace.target_routing
        target_top_k = trace.target_top_k
        
        # Handle different tensor shapes properly
        if hidden_states.dim() > 2:
            # Reshape [batch, seq, hidden] -> [seq, hidden] by taking first sequence
            hidden_states = hidden_states[0]  # Take first batch element
            target_routing = target_routing[0]
            target_top_k = target_top_k[0]
        
        # Truncate sequences if too long
        if hidden_states.shape[0] > self.max_seq_len:
            hidden_states = hidden_states[:self.max_seq_len]
            target_routing = target_routing[:self.max_seq_len]
            target_top_k = target_top_k[:self.max_seq_len]
        
        # Pad sequences if too short
        seq_len = hidden_states.shape[0]
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            
            # Pad hidden states
            hidden_dim = hidden_states.shape[1]
            hidden_padding = torch.zeros(pad_len, hidden_dim)
            hidden_states = torch.cat([hidden_states, hidden_padding], dim=0)
            
            # Pad routing targets
            num_experts = target_routing.shape[1]
            routing_padding = torch.zeros(pad_len, num_experts)
            target_routing = torch.cat([target_routing, routing_padding], dim=0)
            
            # Pad top-k targets
            k = target_top_k.shape[1]
            topk_padding = torch.full((pad_len, k), num_experts, dtype=target_top_k.dtype)  # Invalid expert index
            target_top_k = torch.cat([target_top_k, topk_padding], dim=0)
        
        # Extract expert weights from routing probabilities
        expert_weights = torch.zeros_like(target_top_k, dtype=torch.float)
        for seq_idx in range(min(seq_len, self.max_seq_len)):
            for k_idx in range(target_top_k.shape[1]):
                expert_idx = target_top_k[seq_idx, k_idx]
                if expert_idx < target_routing.shape[1]:  # Valid expert
                    expert_weights[seq_idx, k_idx] = target_routing[seq_idx, expert_idx]
        
        # Create attention mask
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:seq_len] = True
        
        return {
            'hidden_states': hidden_states,
            'expert_indices': target_top_k,
            'expert_weights': expert_weights,
            'attention_mask': attention_mask,
            'original_seq_len': seq_len
        }

class ShardDataLoader:
    """Custom data loader that processes multiple shards concurrently for better efficiency"""
    
    def __init__(self, shard_files: List[Path], batch_size: int = 16, shuffle: bool = True, max_seq_len: int = 256, shards_per_group: int = 2):
        self.shard_files = shard_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.shards_per_group = shards_per_group
        
    def __iter__(self):
        """Iterate through shards in groups for better memory efficiency"""
        shard_order = list(range(len(self.shard_files)))
        if self.shuffle:
            np.random.shuffle(shard_order)
        
        # Process shards in groups
        for group_start in range(0, len(shard_order), self.shards_per_group):
            group_end = min(group_start + self.shards_per_group, len(shard_order))
            shard_group = shard_order[group_start:group_end]
            
            logger.info(f"Processing shard group: {[shard_order[i] for i in range(group_start, group_end)]}")
            
            # Load all shards in the group
            all_traces = []
            for shard_idx in shard_group:
                shard_file = self.shard_files[shard_idx]
                logger.info(f"Loading shard {shard_idx}: {shard_file.name}")
                
                try:
                    with open(shard_file, 'rb') as f:
                        shard_traces = pickle.load(f)
                    logger.info(f"Loaded {len(shard_traces)} traces from shard {shard_idx}")
                    all_traces.extend(shard_traces)
                    
                except Exception as e:
                    logger.error(f"Error loading shard {shard_idx}: {e}")
                    continue
            
            if not all_traces:
                logger.warning(f"No traces loaded from shard group {shard_group}")
                continue
                
            logger.info(f"Total traces in group: {len(all_traces)}")
            
            # Shuffle combined traces
            if self.shuffle:
                np.random.shuffle(all_traces)
            
            # Process combined traces in batches
            batch_count = 0
            total_batches = (len(all_traces) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(all_traces), self.batch_size):
                batch_traces = all_traces[i:i + self.batch_size]
                batch_count += 1
                
                if batch_count % 10 == 0 or batch_count == total_batches:
                    logger.info(f"Processing batch {batch_count}/{total_batches} from shard group")
                
                batch = self._collate_traces(batch_traces)
                yield batch
            
            logger.info(f"Processed {batch_count} batches from shard group {shard_group}")
            
            # Clean memory after each group
            del all_traces
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _collate_traces(self, traces):
        """Collate a batch of traces"""
        batch_size = len(traces)
        logger.info(f"Collating batch of {batch_size} traces")
        
        # Initialize batch tensors
        hidden_states = torch.zeros(batch_size, self.max_seq_len, 2048)  # Qwen hidden dim
        expert_indices = torch.full((batch_size, self.max_seq_len, 4), 60, dtype=torch.long)  # Invalid expert initially
        expert_weights = torch.zeros(batch_size, self.max_seq_len, 4)
        attention_masks = torch.zeros(batch_size, self.max_seq_len, dtype=torch.bool)
        
        for i, trace in enumerate(traces):
            # Extract trace data - handle different tensor shapes
            trace_hidden = trace.hidden_states
            trace_routing = trace.target_routing
            trace_topk = trace.target_top_k
            
            # Handle different tensor shapes - flatten batch dimensions properly
            if trace_hidden.dim() > 2:
                # Reshape [batch, seq, hidden] -> [batch*seq, hidden], then take first seq_len
                batch_size_trace, seq_len_trace = trace_hidden.shape[:2]
                trace_hidden = trace_hidden.view(-1, trace_hidden.shape[-1])[:seq_len_trace]
                trace_routing = trace_routing.view(-1, trace_routing.shape[-1])[:seq_len_trace]  
                trace_topk = trace_topk.view(-1, trace_topk.shape[-1])[:seq_len_trace]
            
            seq_len = min(trace_hidden.shape[0], self.max_seq_len)
            # Reduce seq_len for faster processing during debugging
            seq_len = min(seq_len, 64)  # Temporary limit for faster testing
            
            # Copy data
            hidden_states[i, :seq_len] = trace_hidden[:seq_len]
            expert_indices[i, :seq_len] = trace_topk[:seq_len]
            attention_masks[i, :seq_len] = True
            
            # Extract weights - much faster vectorized version
            try:
                # Clamp expert indices to valid range
                valid_topk = torch.clamp(trace_topk[:seq_len], 0, trace_routing.shape[1] - 1)
                
                # Use advanced indexing to extract weights efficiently
                for seq_idx in range(seq_len):
                    for k_idx in range(min(4, valid_topk.shape[1])):
                        expert_idx = valid_topk[seq_idx, k_idx]
                        expert_weights[i, seq_idx, k_idx] = trace_routing[seq_idx, expert_idx]
                        
            except Exception as e:
                logger.warning(f"Error extracting weights for trace {i}: {e}")
                # Skip this trace if there's an issue
        
        return {
            'hidden_states': hidden_states,
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'attention_mask': attention_masks
        }

def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Move to device
        hidden_states = batch['hidden_states'].to(device)
        expert_indices = batch['expert_indices'].to(device)
        expert_weights = batch['expert_weights'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # Model prediction
            predictions = model(hidden_states, attention_mask)
            
            # Compute loss
            targets = {
                'expert_indices': expert_indices,
                'expert_weights': expert_weights
            }
            
            loss_dict = model.compute_loss(predictions, targets)
            loss = loss_dict['total_loss']
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/num_batches:.4f}",
            'expert_loss': f"{loss_dict['expert_loss'].item():.4f}",
            'topk_loss': f"{loss_dict['top_k_loss'].item():.4f}"
        })
        
        # Memory cleanup
        del hidden_states, expert_indices, expert_weights, attention_mask, predictions, loss_dict
        torch.cuda.empty_cache()
    
    if num_batches == 0:
        logger.warning("No batches processed in this epoch!")
        return 0.0
    
    return total_loss / num_batches

def validate_model(model, dataloader, device):
    """Validate model performance with comprehensive top-k metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Track comprehensive metrics
    k_values = [1, 3, 5, 10, 20]
    top_k_metrics = {k: [] for k in k_values}
    exact_match_scores = []
    partial_match_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            num_batches += 1
            hidden_states = batch['hidden_states'].to(device)
            expert_indices = batch['expert_indices'].to(device)
            expert_weights = batch['expert_weights'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                predictions = model(hidden_states, attention_mask)
                
                targets = {
                    'expert_indices': expert_indices,
                    'expert_weights': expert_weights
                }
                
                loss_dict = model.compute_loss(predictions, targets)
                total_loss += loss_dict['total_loss'].item()
                
                # Get predictions for top-k evaluation
                expert_logits = predictions['expert_logits']  # [batch, seq, 60]
                predicted_top4 = predictions['top_k_indices']  # [batch, seq, 4]
                
                batch_size, seq_len = expert_indices.shape[:2]
                
                # Calculate comprehensive metrics for each sequence position
                for b in range(batch_size):
                    for s in range(seq_len):
                        if not attention_mask[b, s]:
                            continue
                        
                        target = expert_indices[b, s]  # [4] - target top-4 experts
                        pred_logits = expert_logits[b, s]  # [60] - predicted logits
                        
                        # Filter valid targets (< 60)
                        valid_targets = target[target < 60]
                        if len(valid_targets) == 0:
                            continue
                        
                        # Top-k accuracy for different k values
                        for k in k_values:
                            if k <= 60:  # Don't exceed number of experts
                                _, top_k_indices = torch.topk(pred_logits, k=k, dim=-1)
                                
                                # Check how many target experts are in predicted top-k
                                matches = sum(1 for expert in valid_targets if expert in top_k_indices)
                                accuracy = matches / len(valid_targets)
                                top_k_metrics[k].append(accuracy)
                        
                        # Exact match (all 4 experts predicted correctly in top-4)
                        pred_top4 = predicted_top4[b, s]  # [4]
                        exact_match = all(expert in pred_top4 for expert in valid_targets)
                        exact_match_scores.append(float(exact_match))
                        
                        # Partial match (at least 1 expert predicted correctly)
                        partial_match = any(expert in pred_top4 for expert in valid_targets)
                        partial_match_scores.append(float(partial_match))
            
            # Memory cleanup
            del hidden_states, expert_indices, expert_weights, attention_mask, predictions
            torch.cuda.empty_cache()
    
    # Compute final metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    results = {}
    for k in k_values:
        if top_k_metrics[k]:
            results[f'top_{k}'] = np.mean(top_k_metrics[k])
    
    results['exact_match'] = np.mean(exact_match_scores) if exact_match_scores else 0
    results['partial_match'] = np.mean(partial_match_scores) if partial_match_scores else 0
    
    return avg_loss, results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Qwen Multi-Expert Predictor')
    parser.add_argument('--shard-dir', default='../routing_data/shards', help='Directory containing shards')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (RTX 3090 optimized)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-dir', default='../models/checkpoints', help='Save directory')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--shards-per-group', type=int, default=4, help='Number of shards to process together (default: 4, ~16GB). Use 1-6 based on your GPU memory.')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Convert paths to absolute paths
    script_dir = Path(__file__).parent.absolute()
    shard_dir = (script_dir / args.shard_dir).resolve()
    save_dir = (script_dir / args.save_dir).resolve()
    
    # Create directories
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load shard files
    shard_files = sorted(shard_dir.glob("shard_*_traces.pkl"))
    logger.info(f"Found {len(shard_files)} shard files")
    
    if not shard_files:
        logger.error(f"No shard files found in {shard_dir}")
        return
    
    # Split shards for train/validation
    train_shards = shard_files[:int(0.8 * len(shard_files))]
    val_shards = shard_files[int(0.8 * len(shard_files)):]
    
    logger.info(f"Training shards: {len(train_shards)}, Validation shards: {len(val_shards)}")
    
    # Create data loaders - configurable shard group size
    val_shards_per_group = min(args.shards_per_group, len(val_shards)) if val_shards else 1
    
    logger.info(f"Using {args.shards_per_group} shards per group for training (~{args.shards_per_group * 4}GB memory)")
    logger.info(f"Using {val_shards_per_group} shards per group for validation")
    
    train_loader = ShardDataLoader(train_shards, batch_size=args.batch_size, shuffle=True, shards_per_group=args.shards_per_group)
    val_loader = ShardDataLoader(val_shards, batch_size=args.batch_size, shuffle=False, shards_per_group=val_shards_per_group) if val_shards else None
    
    # Create model
    config_file = shard_dir / "training_config.json"
    model = create_qwen_predictor(config_file if config_file.exists() else None)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch+1)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        if val_loader:
            val_loss, val_metrics = validate_model(model, val_loader, device)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Log comprehensive metrics
            metrics_str = []
            for k in [1, 3, 5, 10, 20]:
                if f'top_{k}' in val_metrics:
                    metrics_str.append(f"Top-{k}: {val_metrics[f'top_{k}']:.4f}")
            
            logger.info(f"Top-k Accuracies: {' | '.join(metrics_str)}")
            logger.info(f"Exact Match: {val_metrics['exact_match']:.4f}, Partial Match: {val_metrics['partial_match']:.4f}")
            
            # Use top-1 accuracy as main validation metric
            val_accuracy = val_metrics.get('top_1', 0.0)
        else:
            val_loss = train_loss
            val_metrics = {}
            val_accuracy = 0.0
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'val_accuracy': val_accuracy
        }
        
        # Save latest
        torch.save(checkpoint, save_dir / 'latest_checkpoint.pth')
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, save_dir / 'best_checkpoint.pth')
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()