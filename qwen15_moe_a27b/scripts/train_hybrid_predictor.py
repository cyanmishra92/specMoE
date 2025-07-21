#!/usr/bin/env python3
"""
Hybrid Training Script: Classification + Temporal Prediction
Combines ExpertFlow-style immediate classification with temporal speculation
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
from hybrid_expert_predictor import create_hybrid_predictor

# Import datapoint class for pickle loading
sys.path.append(os.path.join(os.path.dirname(__file__), 'collection'))
from collect_qwen15_moe_traces_streaming import QwenMoEGatingDataPoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridDataLoader:
    """Data loader for hybrid training with temporal targets"""
    
    def __init__(self, shard_files, batch_size=12, shuffle=True, shards_per_group=2, lookahead_steps=4):
        self.shard_files = shard_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shards_per_group = shards_per_group
        self.lookahead_steps = lookahead_steps
        self.max_seq_len = 256  # Standard sequence length
    
    def __iter__(self):
        """Process shards in groups"""
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
                
                if batch_count % 15 == 0 or batch_count == total_batches:
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
        """Collate traces into batch with temporal targets"""
        try:
            batch_size = len(traces)
            hidden_states = torch.zeros(batch_size, self.max_seq_len, 2048)
            expert_indices = torch.full((batch_size, self.max_seq_len, 4), 60, dtype=torch.long)
            
            # Future expert indices for temporal prediction
            future_expert_indices = torch.full(
                (batch_size, self.max_seq_len, self.lookahead_steps, 4), 
                60, dtype=torch.long
            )
            
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
                
                # Current data
                hidden_states[i, :seq_len] = trace_hidden[:seq_len]
                expert_indices[i, :seq_len] = trace_topk[:seq_len]
                attention_mask[i, :seq_len] = True
                
                # Future targets (for temporal prediction)
                for s in range(seq_len):
                    for step in range(self.lookahead_steps):
                        future_pos = s + step + 1
                        if future_pos < seq_len:
                            future_expert_indices[i, s, step] = trace_topk[future_pos]
                        else:
                            # If we go beyond sequence, use padding (60)
                            future_expert_indices[i, s, step] = torch.full((4,), 60)
            
            return {
                'hidden_states': hidden_states,
                'expert_indices': expert_indices,
                'future_expert_indices': future_expert_indices,
                'attention_mask': attention_mask
            }
        except Exception as e:
            logger.warning(f"Error collating batch: {e}")
            return None


def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """Train for one epoch with hybrid loss"""
    model.train()
    total_loss = 0.0
    classification_loss_sum = 0.0
    temporal_loss_sum = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        try:
            # Move to device
            hidden_states = batch['hidden_states'].to(device)
            expert_indices = batch['expert_indices'].to(device)
            future_expert_indices = batch['future_expert_indices'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                predictions = model(hidden_states, attention_mask)
                
                targets = {
                    'expert_indices': expert_indices,
                    'future_expert_indices': future_expert_indices
                }
                
                loss_dict = model.compute_loss(predictions, targets)
                loss = loss_dict['total_loss']
            
            # Check for NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss.item()}")
                continue
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            classification_loss_sum += loss_dict['classification_loss'].item()
            temporal_loss_sum += loss_dict['temporal_loss'].item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'cls_loss': f"{classification_loss_sum / num_batches:.4f}",
                'temp_loss': f"{temporal_loss_sum / num_batches:.4f}"
            })
            
            # Memory cleanup
            del hidden_states, expert_indices, future_expert_indices, attention_mask, predictions
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Error in batch: {e}")
            continue
    
    if num_batches == 0:
        logger.warning("No batches processed in this epoch!")
        return 0.0, 0.0, 0.0
    
    return (total_loss / num_batches, 
            classification_loss_sum / num_batches,
            temporal_loss_sum / num_batches)


def validate_model(model, dataloader, device):
    """Validation with comprehensive metrics"""
    model.eval()
    total_loss = 0.0
    classification_loss_sum = 0.0
    temporal_loss_sum = 0.0
    num_batches = 0
    
    # Metrics tracking
    k_values = [1, 3, 5, 10, 20]
    current_top_k = {k: [] for k in k_values}
    temporal_top_k = {step: {k: [] for k in k_values} for step in range(4)}
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                hidden_states = batch['hidden_states'].to(device)
                expert_indices = batch['expert_indices'].to(device)
                future_expert_indices = batch['future_expert_indices'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    predictions = model(hidden_states, attention_mask)
                    
                    targets = {
                        'expert_indices': expert_indices,
                        'future_expert_indices': future_expert_indices
                    }
                    
                    loss_dict = model.compute_loss(predictions, targets)
                    
                    if not any(torch.isnan(v) or torch.isinf(v) for v in loss_dict.values()):
                        total_loss += loss_dict['total_loss'].item()
                        classification_loss_sum += loss_dict['classification_loss'].item()
                        temporal_loss_sum += loss_dict['temporal_loss'].item()
                        num_batches += 1
                
                # Evaluate current predictions
                current_probs = predictions['current_expert_probs']
                batch_size, seq_len = expert_indices.shape[:2]
                
                for b in range(batch_size):
                    for s in range(seq_len):
                        if not attention_mask[b, s]:
                            continue
                        
                        # Current targets
                        current_targets = expert_indices[b, s]
                        valid_current = current_targets[current_targets < 60]
                        
                        if len(valid_current) > 0:
                            current_logits = current_probs[b, s]
                            
                            for k in k_values:
                                if k <= 60:
                                    _, top_k_indices = torch.topk(current_logits, k=k)
                                    matches = sum(1 for expert in valid_current if expert in top_k_indices)
                                    accuracy = matches / len(valid_current)
                                    current_top_k[k].append(accuracy)
                        
                        # Temporal targets evaluation
                        future_probs = predictions['future_expert_probs']
                        for step in range(4):
                            if s + step + 1 < seq_len:
                                future_targets = future_expert_indices[b, s, step]
                                valid_future = future_targets[future_targets < 60]
                                
                                if len(valid_future) > 0:
                                    step_logits = future_probs[b, s, step]
                                    
                                    for k in k_values:
                                        if k <= 60:
                                            _, top_k_indices = torch.topk(step_logits, k=k)
                                            matches = sum(1 for expert in valid_future if expert in top_k_indices)
                                            accuracy = matches / len(valid_future)
                                            temporal_top_k[step][k].append(accuracy)
                        
            except Exception as e:
                logger.warning(f"Error in validation batch: {e}")
                continue
            
            # Memory cleanup
            del hidden_states, expert_indices, future_expert_indices, attention_mask, predictions
            torch.cuda.empty_cache()
    
    # Compute final metrics
    if num_batches == 0:
        return 0.0, 0.0, 0.0, {}, {}
    
    avg_loss = total_loss / num_batches
    avg_cls_loss = classification_loss_sum / num_batches
    avg_temp_loss = temporal_loss_sum / num_batches
    
    # Current metrics
    current_metrics = {}
    for k in k_values:
        if current_top_k[k]:
            current_metrics[f'current_top_{k}'] = np.mean(current_top_k[k])
    
    # Temporal metrics (average across steps)
    temporal_metrics = {}
    for k in k_values:
        step_accuracies = []
        for step in range(4):
            if temporal_top_k[step][k]:
                step_accuracies.append(np.mean(temporal_top_k[step][k]))
        if step_accuracies:
            temporal_metrics[f'temporal_top_{k}'] = np.mean(step_accuracies)
    
    return avg_loss, avg_cls_loss, avg_temp_loss, current_metrics, temporal_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Expert Predictor')
    parser.add_argument('--shard-dir', default='../routing_data/shards', help='Shard directory')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--save-dir', default='../models/hybrid_checkpoints', help='Save directory')
    parser.add_argument('--shards-per-group', type=int, default=2, help='Number of shards to process together')
    parser.add_argument('--lookahead-steps', type=int, default=4, help='Number of future steps to predict')
    
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
    
    # Create data loaders
    logger.info(f"Using {args.shards_per_group} shards per group (~{args.shards_per_group * 4}GB memory)")
    
    train_loader = HybridDataLoader(
        train_shards, 
        batch_size=args.batch_size, 
        shuffle=True, 
        shards_per_group=args.shards_per_group,
        lookahead_steps=args.lookahead_steps
    )
    
    val_loader = HybridDataLoader(
        val_shards, 
        batch_size=args.batch_size, 
        shuffle=False, 
        shards_per_group=min(args.shards_per_group, len(val_shards)),
        lookahead_steps=args.lookahead_steps
    )
    
    # Create model
    config = {
        'lookahead_steps': args.lookahead_steps
    }
    model = create_hybrid_predictor(config)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    best_current_top1 = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_cls_loss, train_temp_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch+1
        )
        logger.info(f"Train - Total: {train_loss:.4f}, Classification: {train_cls_loss:.4f}, Temporal: {train_temp_loss:.4f}")
        
        # Validate
        val_loss, val_cls_loss, val_temp_loss, current_metrics, temporal_metrics = validate_model(
            model, val_loader, device
        )
        logger.info(f"Val - Total: {val_loss:.4f}, Classification: {val_cls_loss:.4f}, Temporal: {val_temp_loss:.4f}")
        
        # Log current metrics
        if current_metrics:
            current_str = " | ".join([f"Top-{k.split('_')[-1]}: {v:.4f}" for k, v in current_metrics.items()])
            logger.info(f"Current Expert Accuracy: {current_str}")
        
        # Log temporal metrics
        if temporal_metrics:
            temporal_str = " | ".join([f"Top-{k.split('_')[-1]}: {v:.4f}" for k, v in temporal_metrics.items()])
            logger.info(f"Temporal Expert Accuracy: {temporal_str}")
        
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'current_metrics': current_metrics,
            'temporal_metrics': temporal_metrics
        }
        
        torch.save(checkpoint, save_dir / 'latest_checkpoint.pth')
        
        # Save best model based on current top-1 accuracy
        current_top1 = current_metrics.get('current_top_1', 0.0)
        if current_top1 > best_current_top1:
            best_current_top1 = current_top1
            torch.save(checkpoint, save_dir / 'best_checkpoint.pth')
            logger.info(f"New best model saved with Current Top-1: {current_top1:.4f}")
    
    logger.info(f"Training completed! Best Current Top-1 accuracy: {best_current_top1:.4f}")


if __name__ == "__main__":
    main()