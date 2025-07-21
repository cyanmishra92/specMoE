#!/usr/bin/env python3
"""
Simplified training script for stable expert prediction
Focus on getting basic functionality working
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
import logging
import pickle
from tqdm import tqdm
import gc

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from simple_qwen_predictor import create_simple_qwen_predictor

# Import datapoint class for pickle loading
sys.path.append(os.path.join(os.path.dirname(__file__), 'collection'))
from collect_qwen15_moe_traces_streaming import QwenMoEGatingDataPoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDataLoader:
    """Simple data loader with configurable shard grouping"""
    
    def __init__(self, shard_files, batch_size=8, shuffle=True, shards_per_group=2):
        self.shard_files = shard_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shards_per_group = shards_per_group
        self.max_seq_len = 128  # Shorter sequences for stability
    
    def __iter__(self):
        """Process shards in groups for better efficiency"""
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
                
                if batch_count % 20 == 0 or batch_count == total_batches:
                    logger.info(f"Processing batch {batch_count}/{total_batches} from shard group")
                
                batch = self._collate_traces(batch_traces)
                if batch is not None:
                    yield batch
            
            logger.info(f"Processed {batch_count} batches from shard group {shard_group}")
            
            # Clean memory after each group
            del all_traces
            gc.collect()
            torch.cuda.empty_cache()
    
    def _collate_traces(self, traces):
        """Collate traces into batch"""
        try:
            batch_size = len(traces)
            hidden_states = torch.zeros(batch_size, self.max_seq_len, 2048)
            expert_indices = torch.full((batch_size, self.max_seq_len, 4), 60, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, self.max_seq_len, dtype=torch.bool)
            
            for i, trace in enumerate(traces):
                # Handle tensor shapes
                trace_hidden = trace.hidden_states
                trace_topk = trace.target_top_k
                
                # Take first sequence if batch dimension exists
                if trace_hidden.dim() > 2:
                    trace_hidden = trace_hidden[0]
                    trace_topk = trace_topk[0]
                
                seq_len = min(trace_hidden.shape[0], self.max_seq_len)
                
                hidden_states[i, :seq_len] = trace_hidden[:seq_len]
                expert_indices[i, :seq_len] = trace_topk[:seq_len]
                attention_mask[i, :seq_len] = True
            
            return {
                'hidden_states': hidden_states,
                'expert_indices': expert_indices,
                'attention_mask': attention_mask
            }
        except Exception as e:
            logger.warning(f"Error collating batch: {e}")
            return None


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        try:
            # Move to device
            hidden_states = batch['hidden_states'].to(device)
            expert_indices = batch['expert_indices'].to(device) 
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(hidden_states, attention_mask)
            
            targets = {'expert_indices': expert_indices}
            loss_dict = model.compute_loss(predictions, targets)
            
            loss = loss_dict['total_loss']
            
            # Check for NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss.item()}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
            
        except Exception as e:
            logger.warning(f"Error in batch: {e}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_model(model, dataloader, device):
    """Simple validation"""
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_predictions = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                hidden_states = batch['hidden_states'].to(device)
                expert_indices = batch['expert_indices'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                predictions = model(hidden_states, attention_mask)
                
                targets = {'expert_indices': expert_indices}
                loss_dict = model.compute_loss(predictions, targets)
                
                if not torch.isnan(loss_dict['total_loss']):
                    total_loss += loss_dict['total_loss'].item()
                    num_batches += 1
                
                # Calculate top-k accuracy
                expert_logits = predictions['expert_logits']
                
                for b in range(expert_indices.shape[0]):
                    for s in range(expert_indices.shape[1]):
                        if not attention_mask[b, s]:
                            continue
                        
                        targets_seq = expert_indices[b, s]
                        valid_targets = targets_seq[targets_seq < 60]
                        
                        if len(valid_targets) == 0:
                            continue
                        
                        logits_seq = expert_logits[b, s]
                        
                        # Top-1 accuracy
                        _, top1_pred = torch.topk(logits_seq, k=1)
                        if top1_pred[0] in valid_targets:
                            correct_top1 += 1
                        
                        # Top-5 accuracy  
                        _, top5_pred = torch.topk(logits_seq, k=5)
                        if any(pred in valid_targets for pred in top5_pred):
                            correct_top5 += 1
                        
                        total_predictions += 1
                        
            except Exception as e:
                logger.warning(f"Error in validation batch: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    top1_acc = correct_top1 / total_predictions if total_predictions > 0 else 0.0
    top5_acc = correct_top5 / total_predictions if total_predictions > 0 else 0.0
    
    return avg_loss, top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description='Train Simple Qwen Predictor')
    parser.add_argument('--shard-dir', default='../routing_data/shards', help='Shard directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (conservative)')
    parser.add_argument('--save-dir', default='../models/simple_checkpoints', help='Save directory')
    parser.add_argument('--shards-per-group', type=int, default=2, help='Number of shards to process together (default: 2, ~8GB)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load shards
    shard_dir = Path(args.shard_dir)
    shard_files = sorted(shard_dir.glob("*.pkl"))
    
    if not shard_files:
        raise ValueError(f"No shard files found in {shard_dir}")
    
    logger.info(f"Found {len(shard_files)} shard files")
    
    # Split train/val
    train_shards = shard_files[:int(0.8 * len(shard_files))]
    val_shards = shard_files[int(0.8 * len(shard_files)):]
    
    logger.info(f"Training shards: {len(train_shards)}, Validation shards: {len(val_shards)}")
    
    # Create data loaders with configurable shard grouping
    logger.info(f"Using {args.shards_per_group} shards per group (~{args.shards_per_group * 4}GB memory)")
    
    train_loader = SimpleDataLoader(train_shards, batch_size=args.batch_size, shuffle=True, shards_per_group=args.shards_per_group)
    val_loader = SimpleDataLoader(val_shards, batch_size=args.batch_size, shuffle=False, shards_per_group=min(args.shards_per_group, len(val_shards)))
    
    # Create model
    model = create_simple_qwen_predictor()
    model = model.to(device)
    
    # Optimizer with conservative settings
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"\\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch+1)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_top1, val_top5 = validate_model(model, val_loader, device)
        logger.info(f"Val loss: {val_loss:.4f}, Top-1: {val_top1:.4f}, Top-5: {val_top5:.4f}")
        
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_top1': val_top1,
            'val_top5': val_top5
        }
        
        torch.save(checkpoint, save_dir / 'latest_checkpoint.pth')
        
        # Save best model
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(checkpoint, save_dir / 'best_checkpoint.pth')
            logger.info(f"New best model saved with Top-1: {val_top1:.4f}")
    
    logger.info(f"Training completed! Best Top-1 accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()