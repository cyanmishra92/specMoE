#!/usr/bin/env python3
"""
Improved Speculation Training
Simple improvements to the proven baseline approach

Improvements:
1. Better hyperparameters (learning rate, batch size)
2. More epochs with early stopping
3. Better data loading
4. Simplified but effective approach
"""

import os
import sys
import json
import logging
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.interlayer_model import InterLayerSpeculationModel
from utils.data_processing import load_traces

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedSpeculativeDataset(torch.utils.data.Dataset):
    """Improved dataset with better processing"""
    
    def __init__(self, traces, max_layers=12, context_length=3, prediction_horizon=2):
        self.traces = []
        self.max_layers = max_layers
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
        logger.info(f"Processing traces for improved training...")
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
            
            # Create sequences
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
                    'target_experts': target_expert_seq[:, 0],
                    'target_layer_id': torch.tensor(target_layers[0]),
                    'sample_id': sample_id,
                    'seq_len': max_seq_len
                })
        
        logger.info(f"Created {len(self.traces)} training sequences from {len(sample_groups)} samples")
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return self.traces[idx]

def collate_fn(batch):
    """Collate function for improved training"""
    max_seq_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    context_length = batch[0]['context_experts'].size(1)
    
    context_experts = torch.zeros(batch_size, max_seq_len, context_length, dtype=torch.long)
    target_experts = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    layer_ids = torch.zeros(batch_size, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        context_experts[i, :seq_len] = item['context_experts']
        target_experts[i, :seq_len] = item['target_experts']
        layer_ids[i] = item['target_layer_id']
        attention_mask[i, :seq_len] = True
    
    return {
        'context_experts': context_experts,
        'target_experts': target_experts,
        'layer_ids': layer_ids,
        'attention_mask': attention_mask
    }

def evaluate_model(model, dataloader, device):
    """Evaluate improved model"""
    model.eval()
    total_samples = 0
    top_k_correct = {1: 0, 3: 0, 5: 0, 10: 0}
    total_confidence = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            context_experts = batch['context_experts'].to(device)
            target_experts = batch['target_experts'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            expert_logits, confidence, _ = model(context_experts, layer_ids, attention_mask)
            
            valid_mask = (target_experts != -100) & attention_mask
            valid_logits = expert_logits[valid_mask]
            valid_targets = target_experts[valid_mask]
            valid_confidence = confidence[valid_mask]
            
            if valid_logits.size(0) == 0:
                continue
            
            for k in [1, 3, 5, 10]:
                _, top_k_pred = torch.topk(valid_logits, k, dim=-1)
                top_k_hits = (top_k_pred == valid_targets.unsqueeze(1)).any(dim=1)
                top_k_correct[k] += top_k_hits.sum().item()
            
            total_samples += valid_logits.size(0)
            total_confidence += valid_confidence.sum().item()
    
    accuracies = {}
    for k in [1, 3, 5, 10]:
        accuracies[f'top_{k}_accuracy'] = top_k_correct[k] / total_samples * 100
    
    accuracies['avg_confidence'] = total_confidence / total_samples
    
    return accuracies

def train_improved_speculation():
    """Train improved speculation model"""
    
    config = {
        'num_experts': 128,
        'hidden_size': 512,
        'num_layers': 12,
        'model_dim': 320,           # Slightly larger than baseline
        'num_heads': 10,            # Slightly more heads
        'ff_dim': 1280,             # Larger FF
        'num_attention_layers': 5,  # One more layer
        'dropout': 0.12,            # Slightly more dropout
        'context_length': 3,
        'prediction_horizon': 2,
        'batch_size': 28,           # Better batch size
        'learning_rate': 6e-5,      # Lower learning rate
        'num_epochs': 120,          # Extended epochs for better convergence
        'warmup_steps': 800,
        'weight_decay': 0.012,      # Slightly more regularization
        'gradient_clip': 0.8,
        'label_smoothing': 0.06
    }
    
    logger.info("Starting Improved Speculation Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    traces = load_traces("routing_data/robust_traces.pkl")
    if traces is None:
        raise ValueError("Could not load traces")
    
    # Create dataset
    dataset = ImprovedSpeculativeDataset(
        traces,
        max_layers=config['num_layers'],
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
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = InterLayerSpeculationModel(
        num_experts=config['num_experts'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        model_dim=config['model_dim'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        num_attention_layers=config['num_attention_layers'],
        dropout=config['dropout'],
        context_length=config['context_length'],
        prediction_horizon=config['prediction_horizon']
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.7)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config['label_smoothing'])
    
    # Training loop
    best_accuracy = 0.0
    training_results = []
    patience = 25  # More patience for extended training
    no_improve = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            context_experts = batch['context_experts'].to(device)
            target_experts = batch['target_experts'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Warmup learning rate
            if epoch * len(train_loader) + batch_idx < config['warmup_steps']:
                warmup_lr = config['learning_rate'] * (epoch * len(train_loader) + batch_idx) / config['warmup_steps']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            optimizer.zero_grad()
            
            expert_logits, confidence, _ = model(context_experts, layer_ids, attention_mask)
            
            valid_mask = (target_experts != -100) & attention_mask
            valid_logits = expert_logits[valid_mask]
            valid_targets = target_experts[valid_mask]
            
            if valid_logits.size(0) > 0:
                loss = criterion(valid_logits, valid_targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Validation
        logger.info("Evaluating on validation set...")
        val_metrics = evaluate_model(model, val_loader, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        val_metrics.update({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'learning_rate': current_lr
        })
        
        logger.info(f"Results: {json.dumps(val_metrics, indent=2)}")
        training_results.append(val_metrics)
        
        # Update scheduler
        scheduler.step(val_metrics['top_1_accuracy'])
        
        # Save best model
        if val_metrics['top_1_accuracy'] > best_accuracy:
            best_accuracy = val_metrics['top_1_accuracy']
            torch.save(model.state_dict(), 'improved_speculation_best.pt')
            logger.info(f"New best model saved! Top-1 accuracy: {best_accuracy:.2f}%")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    logger.info("\\nFinal evaluation on validation set:")
    final_metrics = evaluate_model(model, val_loader, device)
    logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
    logger.info("Improved training completed!")
    logger.info(f"Best Top-1 Accuracy: {best_accuracy:.2f}%")
    
    # Save results
    results = {
        'config': config,
        'training_history': training_results,
        'final_metrics': final_metrics,
        'best_accuracy': best_accuracy
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"improved_speculation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    model, results = train_improved_speculation()