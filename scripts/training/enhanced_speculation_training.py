#!/usr/bin/env python3
"""
Enhanced Speculative Expert Routing Training
Implements optimizations to improve beyond 33.75% accuracy

Optimizations:
1. Larger model capacity (512 dim, 16 heads, 2048 ff)
2. Extended training (100 epochs) with cosine scheduling
3. Advanced regularization techniques
4. Better initialization and warmup
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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

class SpeculativeExpertDataset(torch.utils.data.Dataset):
    """Enhanced dataset with better sequence processing"""
    
    def __init__(self, traces, max_layers=12, context_length=3, prediction_horizon=2):
        self.traces = []
        self.max_layers = max_layers
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
        logger.info(f"Processing traces for enhanced speculation...")
        logger.info(f"Context length: {context_length}, Prediction horizon: {prediction_horizon}")
        
        # Group traces by sample_id
        sample_groups = {}
        for trace in traces:
            sample_id = trace['sample_id']
            layer_id = trace.get('layer_id', 1)
            
            if sample_id not in sample_groups:
                sample_groups[sample_id] = {}
            
            sample_groups[sample_id][layer_id] = trace
        
        # Create training sequences with enhanced processing
        valid_samples = 0
        for sample_id, layer_traces in sample_groups.items():
            sorted_layers = sorted(layer_traces.keys())
            
            if len(sorted_layers) < context_length + prediction_horizon:
                continue
            
            # Extract expert selections and hidden states
            layer_experts = {}
            layer_hidden_states = {}
            
            for layer_id in sorted_layers:
                trace = layer_traces[layer_id]
                
                # Get expert selections (top-1)
                target_routing = torch.from_numpy(trace['target_routing']).float()
                if target_routing.dim() > 2:
                    target_routing = target_routing.squeeze(0)
                
                expert_ids = torch.argmax(target_routing, dim=-1)
                layer_experts[layer_id] = expert_ids
                
                # Get hidden states
                hidden_states = torch.from_numpy(trace['hidden_states']).float()
                if hidden_states.dim() > 2:
                    hidden_states = hidden_states.squeeze(0)
                layer_hidden_states[layer_id] = hidden_states
            
            # Create sequences with sliding window
            for start_layer in range(len(sorted_layers) - context_length - prediction_horizon + 1):
                context_layers = sorted_layers[start_layer:start_layer + context_length]
                target_layers = sorted_layers[start_layer + context_length:start_layer + context_length + prediction_horizon]
                
                # Get minimum sequence length
                seq_lengths = [layer_experts[layer_id].size(0) for layer_id in context_layers]
                max_seq_len = min(seq_lengths)
                
                if max_seq_len == 0:
                    continue
                
                # Build context expert sequence
                context_expert_seq = torch.stack([
                    layer_experts[layer_id][:max_seq_len] for layer_id in context_layers
                ], dim=1)
                
                # Build target expert sequence
                target_expert_seq = torch.stack([
                    layer_experts[layer_id][:max_seq_len] for layer_id in target_layers
                ], dim=1)
                
                # Create layer ID tensor
                target_layer_ids = torch.tensor(target_layers[:1])
                
                self.traces.append({
                    'context_experts': context_expert_seq,
                    'target_experts': target_expert_seq[:, 0],
                    'target_layer_id': target_layer_ids[0],
                    'sample_id': sample_id,
                    'seq_len': max_seq_len
                })
                
                valid_samples += 1
        
        logger.info(f"Created {len(self.traces)} training sequences from {len(sample_groups)} samples")
        logger.info(f"Average sequences per sample: {valid_samples / len(sample_groups):.1f}")
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return self.traces[idx]

def collate_fn(batch):
    """Enhanced collate function with better padding"""
    max_seq_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    context_length = batch[0]['context_experts'].size(1)
    
    # Initialize tensors
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
    """Enhanced evaluation with more metrics"""
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
            
            # Apply mask for valid positions
            valid_mask = (target_experts != -100) & attention_mask
            valid_logits = expert_logits[valid_mask]
            valid_targets = target_experts[valid_mask]
            valid_confidence = confidence[valid_mask]
            
            if valid_logits.size(0) == 0:
                continue
            
            # Calculate top-k accuracy
            for k in [1, 3, 5, 10]:
                _, top_k_pred = torch.topk(valid_logits, k, dim=-1)
                top_k_hits = (top_k_pred == valid_targets.unsqueeze(1)).any(dim=1)
                top_k_correct[k] += top_k_hits.sum().item()
            
            total_samples += valid_logits.size(0)
            total_confidence += valid_confidence.sum().item()
    
    # Calculate accuracy metrics
    accuracies = {}
    for k in [1, 3, 5, 10]:
        accuracies[f'top_{k}_accuracy'] = top_k_correct[k] / total_samples * 100
    
    accuracies['avg_confidence'] = total_confidence / total_samples
    
    return accuracies

def train_enhanced_speculation_model():
    """Train enhanced inter-layer speculation model"""
    
    # Enhanced configuration
    config = {
        'num_experts': 128,
        'hidden_size': 512,
        'num_layers': 12,
        'model_dim': 512,         # Increased from 256
        'num_heads': 16,          # Increased from 8
        'ff_dim': 2048,           # Increased from 1024
        'num_attention_layers': 6, # Increased from 4
        'dropout': 0.15,          # Slightly increased
        'context_length': 3,
        'prediction_horizon': 2,
        'batch_size': 16,         # Reduced for larger model
        'learning_rate': 5e-5,    # Lower for stability
        'num_epochs': 100,        # Extended training
        'warmup_steps': 2000,     # Longer warmup
        'weight_decay': 0.02,     # Increased regularization
        'gradient_clip': 0.5,     # Tighter clipping
        'label_smoothing': 0.1    # Add label smoothing
    }
    
    logger.info("Starting Enhanced Speculative Expert Routing Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading routing traces...")
    traces = load_traces("routing_data/robust_traces.pkl")
    if traces is None:
        raise ValueError("Could not load traces")
    
    # Create dataset
    dataset = SpeculativeExpertDataset(
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
    
    # Initialize enhanced model
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
    
    # Enhanced optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)  # Better for transformers
    )
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20,  # Restart every 20 epochs
        T_mult=2,  # Double period after each restart
        eta_min=1e-7
    )
    
    # Enhanced loss function with label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config['label_smoothing'])
    
    # Training loop
    best_accuracy = 0.0
    training_results = []
    patience = 15
    no_improve = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in progress_bar:
            context_experts = batch['context_experts'].to(device)
            target_experts = batch['target_experts'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Warmup learning rate
            if epoch * len(train_loader) + num_batches < config['warmup_steps']:
                warmup_lr = config['learning_rate'] * (epoch * len(train_loader) + num_batches) / config['warmup_steps']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            optimizer.zero_grad()
            
            expert_logits, confidence, pattern_weights = model(context_experts, layer_ids, attention_mask)
            
            # Calculate loss only for valid positions
            valid_mask = (target_experts != -100) & attention_mask
            valid_logits = expert_logits[valid_mask]
            valid_targets = target_experts[valid_mask]
            
            if valid_logits.size(0) > 0:
                loss = criterion(valid_logits, valid_targets)
                
                # Add confidence regularization
                conf_loss = 0.1 * torch.mean((confidence[valid_mask] - 0.5) ** 2)
                total_loss = loss + conf_loss
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
        
        # Update scheduler (after warmup)
        if epoch * len(train_loader) >= config['warmup_steps']:
            scheduler.step()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Validation phase
        logger.info("Evaluating on validation set...")
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Log results
        current_lr = optimizer.param_groups[0]['lr']
        val_metrics.update({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'learning_rate': current_lr
        })
        
        logger.info(f"Results: {json.dumps(val_metrics, indent=2)}")
        training_results.append(val_metrics)
        
        # Save best model
        if val_metrics['top_1_accuracy'] > best_accuracy:
            best_accuracy = val_metrics['top_1_accuracy']
            torch.save(model.state_dict(), 'enhanced_speculation_best.pt')
            logger.info(f"New best model saved! Top-1 accuracy: {best_accuracy:.2f}%")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    logger.info("\nFinal evaluation on validation set:")
    final_metrics = evaluate_model(model, val_loader, device)
    logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
    logger.info("Enhanced training completed successfully!")
    logger.info(f"Best Top-1 Accuracy: {best_accuracy:.2f}%")
    
    # Save results
    results = {
        'config': config,
        'training_history': training_results,
        'final_metrics': final_metrics,
        'best_accuracy': best_accuracy
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_speculation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    model, results = train_enhanced_speculation_model()