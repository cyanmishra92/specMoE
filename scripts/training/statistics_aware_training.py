#!/usr/bin/env python3
"""
Statistics-Aware Expert Speculation Training
Training script for the statistics-aware model with layer-specific priors
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.statistics_aware_model import StatisticsAwareSpeculationModel
from utils.data_processing import load_traces

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatisticsAwareDataset(torch.utils.data.Dataset):
    """Dataset for statistics-aware training"""
    
    def __init__(self, traces, context_length=3, prediction_horizon=2):
        self.traces = []
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
        logger.info(f"Processing traces for statistics-aware training...")
        logger.info(f"Context length: {context_length}, Prediction horizon: {prediction_horizon}")
        
        # Group traces by sample_id
        sample_groups = {}
        for trace in traces:
            sample_id = trace['sample_id']
            layer_id = trace.get('layer_id', 1)
            
            if sample_id not in sample_groups:
                sample_groups[sample_id] = {}
            
            sample_groups[sample_id][layer_id] = trace
        
        # Create training sequences
        for sample_id, layer_traces in sample_groups.items():
            sorted_layers = sorted(layer_traces.keys())
            
            if len(sorted_layers) < context_length + prediction_horizon:
                continue
            
            # Extract expert selections
            layer_experts = {}
            for layer_id in sorted_layers:
                trace = layer_traces[layer_id]
                target_routing = torch.from_numpy(trace['target_routing']).float()
                if target_routing.dim() > 2:
                    target_routing = target_routing.squeeze(0)
                
                expert_ids = torch.argmax(target_routing, dim=-1)
                layer_experts[layer_id] = expert_ids
            
            # Create sequences with sliding window
            for start_layer in range(len(sorted_layers) - context_length - prediction_horizon + 1):
                context_layers = sorted_layers[start_layer:start_layer + context_length]
                target_layers = sorted_layers[start_layer + context_length:start_layer + context_length + prediction_horizon]
                
                seq_lengths = [layer_experts[layer_id].size(0) for layer_id in context_layers]
                max_seq_len = min(seq_lengths)
                
                if max_seq_len == 0:
                    continue
                
                context_expert_seq = torch.stack([
                    layer_experts[layer_id][:max_seq_len] for layer_id in context_layers
                ], dim=1)
                
                target_expert_seq = torch.stack([
                    layer_experts[layer_id][:max_seq_len] for layer_id in target_layers
                ], dim=1)
                
                self.traces.append({
                    'context_experts': context_expert_seq,
                    'target_experts': target_expert_seq[:, 0],  # First target layer
                    'target_layer_id': torch.tensor(target_layers[0]),
                    'context_layers': torch.tensor(context_layers),
                    'sample_id': sample_id
                })
        
        logger.info(f"Created {len(self.traces)} training sequences")
        
        # Calculate layer distribution
        layer_counts = {}
        for trace in self.traces:
            layer_id = trace['target_layer_id'].item()
            layer_counts[layer_id] = layer_counts.get(layer_id, 0) + 1
        
        logger.info(f"Target layer distribution: {layer_counts}")
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return self.traces[idx]

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    
    # Find max sequence length in batch
    max_seq_len = max(item['context_experts'].size(0) for item in batch)
    
    # Pad sequences to max length
    padded_context = []
    padded_targets = []
    target_layer_ids = []
    context_layers = []
    
    for item in batch:
        context = item['context_experts']
        target = item['target_experts']
        seq_len = context.size(0)
        
        if seq_len < max_seq_len:
            # Pad with zeros (will be masked out)
            pad_size = max_seq_len - seq_len
            context = torch.nn.functional.pad(context, (0, 0, 0, pad_size), value=0)
            target = torch.nn.functional.pad(target, (0, pad_size), value=0)
        
        padded_context.append(context)
        padded_targets.append(target)
        target_layer_ids.append(item['target_layer_id'])
        context_layers.append(item['context_layers'])
    
    return {
        'context_experts': torch.stack(padded_context),
        'target_experts': torch.stack(padded_targets),
        'target_layer_id': torch.stack(target_layer_ids),
        'context_layers': torch.stack(context_layers),
        'seq_lengths': torch.tensor([item['context_experts'].size(0) for item in batch])
    }

def calculate_accuracies(predictions, targets, confidences=None):
    """Calculate various accuracy metrics"""
    
    # Convert to numpy for sklearn
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Mask out padding (assuming padded with 0s)
    valid_mask = targets_np != 0
    
    if not valid_mask.any():
        return {'top_1_accuracy': 0.0, 'top_3_accuracy': 0.0, 'top_5_accuracy': 0.0}
    
    valid_predictions = predictions_np[valid_mask]
    valid_targets = targets_np[valid_mask]
    
    # Get top-k predictions
    top_k_predictions = np.argsort(valid_predictions, axis=1)[:, -5:]  # Top 5
    
    # Calculate accuracies
    total_samples = len(valid_targets)
    top_k_correct = {}
    
    for k in [1, 3, 5]:
        correct = 0
        for i, target in enumerate(valid_targets):
            if target in top_k_predictions[i, -k:]:
                correct += 1
        top_k_correct[k] = correct
    
    accuracies = {}
    for k in [1, 3, 5]:
        accuracies[f'top_{k}_accuracy'] = top_k_correct[k] / total_samples * 100
    
    if confidences is not None:
        valid_confidences = confidences.cpu().numpy()[valid_mask]
        accuracies['avg_confidence'] = np.mean(valid_confidences)
    
    return accuracies

def train_statistics_aware_model():
    """Train statistics-aware speculation model"""
    
    # Configuration optimized for statistics-aware training
    config = {
        'num_experts': 128,
        'hidden_size': 512,
        'model_dim': 384,
        'num_heads': 12,
        'ff_dim': 1536,
        'num_attention_layers': 6,
        'dropout': 0.1,
        'context_length': 3,
        'prediction_horizon': 2,
        'batch_size': 32,
        'learning_rate': 1e-4,  # Slightly higher for statistics-aware features
        'num_epochs': 150,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'label_smoothing': 0.05,
        'patience': 15,
        'min_lr': 1e-6,
        'prior_weight': 0.1,  # Weight for statistical priors
        'confidence_weight': 0.1  # Weight for confidence loss
    }
    
    logger.info("Starting Statistics-Aware Speculation Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    traces = load_traces("routing_data/maximum_real_traces.pkl")
    if traces is None:
        raise ValueError("Could not load traces")
    
    # Create dataset
    dataset = StatisticsAwareDataset(
        traces,
        context_length=config['context_length'],
        prediction_horizon=config['prediction_horizon']
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = StatisticsAwareSpeculationModel(
        num_experts=config['num_experts'],
        hidden_size=config['hidden_size'],
        model_dim=config['model_dim'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        num_attention_layers=config['num_attention_layers'],
        context_length=config['context_length'],
        prediction_horizon=config['prediction_horizon']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss functions
    main_criterion = nn.CrossEntropyLoss(
        label_smoothing=config['label_smoothing'],
        ignore_index=0
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=1,
        eta_min=config['min_lr']
    )
    
    # Training loop
    best_val_accuracy = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    logger.info("Starting training...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
        
        for batch_idx, batch in enumerate(train_pbar):
            context_experts = batch['context_experts'].to(device)
            target_experts = batch['target_experts'].to(device)
            target_layer_id = batch['target_layer_id'].to(device)
            context_layers = batch['context_layers'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(context_experts, target_layer_id, context_layers, seq_lengths)
            expert_logits = outputs['expert_logits']
            confidence = outputs['confidence']
            stat_priors = outputs['statistical_priors']
            
            # Reshape for loss calculation
            batch_size, seq_len, num_experts = expert_logits.shape
            expert_logits = expert_logits.view(-1, num_experts)
            target_experts_flat = target_experts.view(-1)
            
            # Main prediction loss
            main_loss = main_criterion(expert_logits, target_experts_flat)
            
            # Statistical prior regularization
            prior_loss = main_criterion(stat_priors, target_experts[:, 0])  # First token
            
            # Confidence regularization (encourage confident predictions)
            confidence_loss = torch.mean(torch.abs(confidence - 0.8))
            
            # Total loss
            total_loss = (main_loss + 
                         config['prior_weight'] * prior_loss + 
                         config['confidence_weight'] * confidence_loss)
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            optimizer.step()
            
            total_train_loss += total_loss.item()
            
            # Collect predictions for accuracy calculation
            _, predicted = torch.max(expert_logits, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(target_experts_flat.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Main': f'{main_loss.item():.4f}',
                'Prior': f'{prior_loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Calculate training accuracy
        train_acc = accuracy_score(train_targets, train_predictions)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_predictions = []
        val_targets = []
        val_confidences = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Validation')
            
            for batch in val_pbar:
                context_experts = batch['context_experts'].to(device)
                target_experts = batch['target_experts'].to(device)
                target_layer_id = batch['target_layer_id'].to(device)
                context_layers = batch['context_layers'].to(device)
                seq_lengths = batch['seq_lengths'].to(device)
                
                outputs = model(context_experts, target_layer_id, context_layers, seq_lengths)
                expert_logits = outputs['expert_logits']
                confidence = outputs['confidence']
                stat_priors = outputs['statistical_priors']
                
                # Reshape for loss calculation
                batch_size, seq_len, num_experts = expert_logits.shape
                expert_logits = expert_logits.view(-1, num_experts)
                target_experts_flat = target_experts.view(-1)
                
                # Calculate losses
                main_loss = main_criterion(expert_logits, target_experts_flat)
                prior_loss = main_criterion(stat_priors, target_experts[:, 0])
                confidence_loss = torch.mean(torch.abs(confidence - 0.8))
                
                total_loss = (main_loss + 
                             config['prior_weight'] * prior_loss + 
                             config['confidence_weight'] * confidence_loss)
                
                total_val_loss += total_loss.item()
                
                # Collect predictions
                val_predictions.append(expert_logits.cpu())
                val_targets.append(target_experts.cpu())
                val_confidences.append(confidence.cpu())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate validation accuracies
        val_predictions_tensor = torch.cat(val_predictions, dim=0)
        val_targets_tensor = torch.cat(val_targets, dim=0).view(-1)
        val_confidences_tensor = torch.cat(val_confidences, dim=0)
        
        val_acc_metrics = calculate_accuracies(val_predictions_tensor, val_targets_tensor, val_confidences_tensor)
        val_accuracy = val_acc_metrics['top_1_accuracy']
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Logging
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        logger.info(f"  Top-3 Acc: {val_acc_metrics['top_3_accuracy']:.4f}, Top-5 Acc: {val_acc_metrics['top_5_accuracy']:.4f}")
        logger.info(f"  Avg Confidence: {val_acc_metrics['avg_confidence']:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            best_model_path = f"models/statistics_aware_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'config': config,
                'val_metrics': val_acc_metrics
            }, best_model_path)
            
            logger.info(f"New best model saved: {val_accuracy:.4f} -> {best_model_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy,
        'config': config
    }
    
    history_path = f"models/statistics_aware_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history saved to: {history_path}")
    
    return best_model_path

if __name__ == "__main__":
    train_statistics_aware_model()