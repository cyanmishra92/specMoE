#!/usr/bin/env python3
"""
Train Multi-Expert Predictor for Qwen1.5-MoE-A2.7B
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass

# Add models to path
sys.path.append('../models')
from multi_expert_predictor import MultiExpertPredictor, MultiExpertLoss, expert_pair_to_index

# Set up logging
logging.basicConfig(level=logging.INFO)
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

class QwenMoEDataset(Dataset):
    """Dataset for Qwen MoE expert prediction"""
    
    def __init__(
        self,
        traces: List[QwenMoEGatingDataPoint],
        prediction_window: int = 3,
        num_experts: int = 60
    ):
        self.traces = traces
        self.prediction_window = prediction_window
        self.num_experts = num_experts
        self.data_points = []
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare training data from traces"""
        logger.info(f"Preparing training data from {len(self.traces)} traces...")
        
        # Group traces by layer and sample
        layer_traces = {}
        for trace in self.traces:
            layer_id = trace.layer_id
            sample_id = trace.sample_id
            
            if layer_id not in layer_traces:
                layer_traces[layer_id] = {}
            layer_traces[layer_id][sample_id] = trace
        
        # Create training sequences
        for layer_id in sorted(layer_traces.keys()):
            if layer_id < self.prediction_window:
                continue
                
            current_layer_traces = layer_traces[layer_id]
            
            for sample_id, trace in current_layer_traces.items():
                # Get previous layers' hidden states
                prev_hidden_states = []
                prev_layer_ids = []
                
                valid_sequence = True
                for prev_layer_id in range(layer_id - self.prediction_window, layer_id):
                    if (prev_layer_id in layer_traces and 
                        sample_id in layer_traces[prev_layer_id]):
                        prev_trace = layer_traces[prev_layer_id][sample_id]
                        prev_hidden_states.append(prev_trace.hidden_states)
                        prev_layer_ids.append(prev_layer_id)
                    else:
                        valid_sequence = False
                        break
                
                if valid_sequence and len(prev_hidden_states) == self.prediction_window:
                    # Stack hidden states: [seq_len, window_size, hidden_size]
                    input_states = torch.stack(prev_hidden_states, dim=1)
                    input_layer_ids = torch.tensor(prev_layer_ids)
                    
                    # Target: expert pair indices and individual experts
                    target_experts = trace.target_top_k  # [seq_len, 2]
                    
                    # Convert to pair indices
                    seq_len = target_experts.shape[0]
                    target_pair_indices = torch.zeros(seq_len, dtype=torch.long)
                    
                    for i in range(seq_len):
                        expert1, expert2 = target_experts[i, 0].item(), target_experts[i, 1].item()
                        if expert1 < self.num_experts and expert2 < self.num_experts:
                            pair_idx = expert_pair_to_index(expert1, expert2, self.num_experts)
                            target_pair_indices[i] = pair_idx
                    
                    self.data_points.append({
                        'input_states': input_states,
                        'input_layer_ids': input_layer_ids,
                        'target_pair_indices': target_pair_indices,
                        'target_experts': target_experts,
                        'layer_id': layer_id,
                        'sample_id': sample_id
                    })
        
        logger.info(f"Created {len(self.data_points)} training sequences")
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        return self.data_points[idx]

def collate_fn(batch):
    """Custom collate function for batching"""
    input_states = torch.stack([item['input_states'] for item in batch])
    input_layer_ids = torch.stack([item['input_layer_ids'] for item in batch])
    target_pair_indices = torch.stack([item['target_pair_indices'] for item in batch])
    target_experts = torch.stack([item['target_experts'] for item in batch])
    
    return {
        'input_states': input_states,
        'input_layer_ids': input_layer_ids,
        'target_pair_indices': target_pair_indices,
        'target_experts': target_experts
    }

def train_epoch(
    model: MultiExpertPredictor,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: MultiExpertLoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_pair_loss = 0.0
    total_expert_loss = 0.0
    total_confidence_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_states = batch['input_states'].to(device)
        input_layer_ids = batch['input_layer_ids'].to(device)
        target_pair_indices = batch['target_pair_indices'].to(device)
        target_experts = batch['target_experts'].to(device)
        
        # Forward pass
        predictions = model(input_states, input_layer_ids)
        
        # Compute loss
        loss_dict = criterion(predictions, target_pair_indices, target_experts)
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss_dict['total_loss'].item()
        total_pair_loss += loss_dict['pair_loss'].item()
        total_expert_loss += loss_dict['expert_loss'].item()
        total_confidence_loss += loss_dict['confidence_loss'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total_loss'].item():.4f}",
            'pair': f"{loss_dict['pair_loss'].item():.4f}",
            'expert': f"{loss_dict['expert_loss'].item():.4f}"
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'pair_loss': total_pair_loss / num_batches,
        'expert_loss': total_expert_loss / num_batches,
        'confidence_loss': total_confidence_loss / num_batches
    }

def train_epoch_sharded(
    model: MultiExpertPredictor,
    sharded_loader,
    optimizer: optim.Optimizer,
    criterion: MultiExpertLoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch using sharded data"""
    model.train()
    total_loss = 0.0
    total_expert_loss = 0.0
    total_ranking_loss = 0.0
    total_confidence_loss = 0.0
    num_batches = 0
    
    # Estimate total batches for progress bar
    estimated_batches = sharded_loader.get_num_batches()
    progress_bar = tqdm(total=estimated_batches, desc=f'Epoch {epoch} (Sharded)')
    
    for batch_traces in sharded_loader.iterate_batches():
        # Convert traces to dataset format
        dataset = QwenMoEDataset(
            traces=batch_traces,
            prediction_window=3,  # Fixed for now
            num_experts=60
        )
        
        if len(dataset) == 0:
            continue
        
        # Create batch
        batch_data = []
        for i in range(len(dataset)):
            batch_data.append(dataset[i])
        
        if not batch_data:
            continue
        
        # Collate batch
        batch = collate_fn(batch_data)
        
        # Move to device
        input_states = batch['input_states'].to(device)
        input_layer_ids = batch['input_layer_ids'].to(device)
        target_experts = batch['target_experts'].to(device)
        
        # Forward pass
        predictions = model(input_states, input_layer_ids)
        
        # Compute loss - handle different loss function signatures
        try:
            # Try the pair-based loss first
            target_pair_indices = torch.zeros(target_experts.shape[0], target_experts.shape[1], dtype=torch.long, device=device)
            for b in range(target_experts.shape[0]):
                for s in range(target_experts.shape[1]):
                    expert1, expert2 = target_experts[b, s, 0].item(), target_experts[b, s, 1].item()
                    if expert1 < 60 and expert2 < 60:  # Valid expert indices
                        pair_idx = expert_pair_to_index(expert1, expert2, 60)
                        target_pair_indices[b, s] = pair_idx
            
            loss_dict = criterion(predictions, target_pair_indices, target_experts)
        except Exception as e:
            # Fallback to expert-only loss
            logger.warning(f"Using fallback loss computation: {e}")
            loss_dict = criterion(predictions, target_experts)
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss_dict['total_loss'].item()
        total_expert_loss += loss_dict['expert_loss'].item()
        total_ranking_loss += loss_dict['ranking_loss'].item()
        total_confidence_loss += loss_dict['confidence_loss'].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total_loss'].item():.4f}",
            'expert': f"{loss_dict['expert_loss'].item():.4f}",
            'ranking': f"{loss_dict['ranking_loss'].item():.4f}"
        })
        progress_bar.update(1)
        
        # Clear cache periodically
        if num_batches % 10 == 0:
            torch.cuda.empty_cache()
    
    progress_bar.close()
    
    return {
        'total_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'expert_loss': total_expert_loss / num_batches if num_batches > 0 else 0.0,
        'ranking_loss': total_ranking_loss / num_batches if num_batches > 0 else 0.0,
        'confidence_loss': total_confidence_loss / num_batches if num_batches > 0 else 0.0
    }

def evaluate_model(
    model: MultiExpertPredictor,
    dataloader: DataLoader,
    criterion: MultiExpertLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_pair_accuracy = 0.0
    total_expert_accuracy = 0.0
    num_batches = len(dataloader)
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move to device
            input_states = batch['input_states'].to(device)
            input_layer_ids = batch['input_layer_ids'].to(device)
            target_pair_indices = batch['target_pair_indices'].to(device)
            target_experts = batch['target_experts'].to(device)
            
            # Forward pass
            predictions = model(input_states, input_layer_ids)
            
            # Compute loss
            loss_dict = criterion(predictions, target_pair_indices, target_experts)
            total_loss += loss_dict['total_loss'].item()
            
            # Compute accuracy
            pair_logits = predictions['expert_pair_logits']
            expert_logits = predictions['expert_logits']
            
            # Pair accuracy
            pair_preds = torch.argmax(pair_logits, dim=-1)
            pair_correct = (pair_preds == target_pair_indices).float()
            total_pair_accuracy += pair_correct.mean().item()
            
            # Expert accuracy (individual)
            expert_preds = torch.argmax(expert_logits, dim=-1)
            expert1_correct = (expert_preds == target_experts[:, :, 0]).float()
            expert2_correct = (expert_preds == target_experts[:, :, 1]).float()
            expert_accuracy = (expert1_correct + expert2_correct) / 2
            total_expert_accuracy += expert_accuracy.mean().item()
            
            num_samples += input_states.size(0)
    
    return {
        'total_loss': total_loss / num_batches,
        'pair_accuracy': total_pair_accuracy / num_batches,
        'expert_accuracy': total_expert_accuracy / num_batches
    }

def main():
    parser = argparse.ArgumentParser(description='Train Multi-Expert Predictor')
    parser.add_argument('--trace_file', type=str, default='routing_data/qwen15_moe_a27b_traces_small.pkl',
                        help='Path to trace file')
    parser.add_argument('--shard_dir', type=str, default='',
                        help='Path to sharded data directory (alternative to trace_file)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--prediction_window', type=int, default=3, help='Prediction window size')
    parser.add_argument('--hidden_size', type=int, default=2048, help='Hidden size')
    parser.add_argument('--intermediate_size', type=int, default=1024, help='Intermediate size')
    parser.add_argument('--num_experts', type=int, default=60, help='Number of experts')
    parser.add_argument('--num_layers', type=int, default=24, help='Number of layers')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load traces or use sharded data
    if args.shard_dir:
        logger.info(f"Using sharded data from {args.shard_dir}")
        
        # Add utils to path
        import sys
        utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
        if utils_path not in sys.path:
            sys.path.append(utils_path)
        from data_sharding import ShardedDataLoader
        
        # Create sharded data loader
        sharded_loader = ShardedDataLoader(
            args.shard_dir,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # Get a sample to determine dataset structure
        sample_traces = sharded_loader.get_sample_batch(100)
        
        # Create a small dataset for validation
        val_dataset = QwenMoEDataset(
            traces=sample_traces[:20],  # Use small sample for validation
            prediction_window=args.prediction_window,
            num_experts=args.num_experts
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )
        
        train_loader = None  # Will use sharded loader directly
        
    else:
        # Load traces from single file
        logger.info(f"Loading traces from {args.trace_file}")
        with open(args.trace_file, 'rb') as f:
            traces = pickle.load(f)
        
        logger.info(f"Loaded {len(traces)} traces")
        
        # Create dataset
        dataset = QwenMoEDataset(
            traces=traces,
            prediction_window=args.prediction_window,
            num_experts=args.num_experts
        )
        
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )
        
        sharded_loader = None
    
    # Create model
    model = MultiExpertPredictor(
        num_experts=args.num_experts,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        prediction_window=args.prediction_window
    ).to(device)
    
    # Create criterion and optimizer
    criterion = MultiExpertLoss(num_experts=args.num_experts)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_history = {'train_loss': [], 'val_loss': [], 'pair_accuracy': [], 'expert_accuracy': []}
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        # Train
        if sharded_loader:
            train_metrics = train_epoch_sharded(model, sharded_loader, optimizer, criterion, device, epoch)
        else:
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['total_loss'])
        
        # Log metrics
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        logger.info(f"  Pair Accuracy: {val_metrics['pair_accuracy']:.4f}")
        logger.info(f"  Expert Accuracy: {val_metrics['expert_accuracy']:.4f}")
        
        # Save history
        train_history['train_loss'].append(train_metrics['total_loss'])
        train_history['val_loss'].append(val_metrics['total_loss'])
        train_history['pair_accuracy'].append(val_metrics['pair_accuracy'])
        train_history['expert_accuracy'].append(val_metrics['expert_accuracy'])
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'pair_accuracy': val_metrics['pair_accuracy'],
                'expert_accuracy': val_metrics['expert_accuracy'],
                'args': args
            }, save_dir / 'best_model.pth')
            logger.info(f"  New best model saved!")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'args': args
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save final model and history
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['total_loss'],
        'args': args
    }, save_dir / 'final_model.pth')
    
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to: {save_dir}")

if __name__ == "__main__":
    main()