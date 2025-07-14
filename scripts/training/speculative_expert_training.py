#!/usr/bin/env python3
"""
Speculative Expert Routing Training
Train models to predict future expert selections for prefetching optimization.

Similar to speculative decoding (EAGLE, Spec-Infer, Medusa) but for expert routing:
- Predict which experts will be used in future layers
- Enable prefetching to reduce memory bandwidth
- Pipeline expert computation across transformer layers
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import json
from pathlib import Path
import time
from tqdm import tqdm

import sys
sys.path.append('/data/research/specMoE/specMoE')

from models.interlayer_model import InterLayerSpeculationModel, InterlayerLoss
from utils.data_processing import load_traces

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeculativeExpertDataset(torch.utils.data.Dataset):
    """
    Dataset for speculative expert routing training.
    Creates sequences of expert selections across layers for future prediction.
    """
    
    def __init__(self, traces, max_layers=12, context_length=8, prediction_horizon=3):
        self.traces = []
        self.max_layers = max_layers
        self.context_length = context_length  # How many past layers to use
        self.prediction_horizon = prediction_horizon  # How many future layers to predict
        
        logger.info(f"Processing traces for speculative expert routing...")
        logger.info(f"Context length: {context_length}, Prediction horizon: {prediction_horizon}")
        
        # Group traces by sample_id and sequence position
        sample_groups = {}
        for trace in traces:
            sample_id = trace['sample_id']
            layer_id = trace.get('layer_id', 1)
            
            if sample_id not in sample_groups:
                sample_groups[sample_id] = {}
            
            sample_groups[sample_id][layer_id] = trace
        
        # Create training sequences
        valid_samples = 0
        for sample_id, layer_traces in sample_groups.items():
            # Sort layers by ID
            sorted_layers = sorted(layer_traces.keys())
            
            if len(sorted_layers) < context_length + prediction_horizon:
                continue  # Need enough layers for context + prediction
            
            # Extract expert selections for each layer
            layer_experts = {}
            layer_hidden_states = {}
            
            for layer_id in sorted_layers:
                trace = layer_traces[layer_id]
                
                # Get expert selections (top-1 for each token)
                target_routing = torch.from_numpy(trace['target_routing']).float()
                if target_routing.dim() > 2:
                    target_routing = target_routing.squeeze(0)
                
                expert_ids = torch.argmax(target_routing, dim=-1)  # [seq_len]
                layer_experts[layer_id] = expert_ids
                
                # Get hidden states
                hidden_states = torch.from_numpy(trace['hidden_states']).float()
                if hidden_states.dim() > 2:
                    hidden_states = hidden_states.squeeze(0)
                layer_hidden_states[layer_id] = hidden_states
            
            # Create sliding window sequences
            for start_layer in range(len(sorted_layers) - context_length - prediction_horizon + 1):
                context_layers = sorted_layers[start_layer:start_layer + context_length]
                target_layers = sorted_layers[start_layer + context_length:start_layer + context_length + prediction_horizon]
                
                # Get sequence length (use minimum across layers)
                seq_lengths = [layer_experts[layer_id].size(0) for layer_id in context_layers]
                min_seq_len = min(seq_lengths)
                max_seq_len = min(min_seq_len, 32)  # Limit sequence length
                
                if max_seq_len < 4:  # Need minimum sequence length
                    continue
                
                # Build context expert sequence [seq_len, context_length]
                context_expert_seq = torch.stack([
                    layer_experts[layer_id][:max_seq_len] for layer_id in context_layers
                ], dim=1)
                
                # Build target expert sequence [seq_len, prediction_horizon]
                target_expert_seq = torch.stack([
                    layer_experts[layer_id][:max_seq_len] for layer_id in target_layers
                ], dim=1)
                
                # Create layer ID tensor
                target_layer_ids = torch.tensor(target_layers[:1])  # Predict first future layer
                
                self.traces.append({
                    'context_experts': context_expert_seq,  # [seq_len, context_length]
                    'target_experts': target_expert_seq[:, 0],  # [seq_len] - first future layer
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
    """Custom collate function for batching sequences"""
    max_seq_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    context_length = batch[0]['context_experts'].size(1)
    
    # Pad sequences
    context_experts = torch.zeros(batch_size, max_seq_len, context_length, dtype=torch.long)
    target_experts = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)  # -100 for padding
    layer_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        context_experts[i, :seq_len] = item['context_experts']
        target_experts[i, :seq_len] = item['target_experts']
        layer_ids[i, :seq_len] = item['target_layer_id']
        attention_mask[i, :seq_len] = True
    
    return {
        'context_experts': context_experts,
        'target_experts': target_experts,
        'layer_ids': layer_ids,
        'attention_mask': attention_mask
    }

def evaluate_speculative_accuracy(model, dataloader, device, top_k_values=[1, 3, 5, 10]):
    """
    Evaluate speculative expert prediction accuracy.
    Measures how well we can predict future expert selections.
    """
    model.eval()
    total_samples = 0
    top_k_correct = {k: 0 for k in top_k_values}
    total_confidence = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            context_experts = batch['context_experts'].to(device)
            target_experts = batch['target_experts'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Model prediction
            expert_logits, confidence, _ = model(context_experts, layer_ids, attention_mask)
            
            # Get valid positions (not padding)
            valid_mask = (target_experts != -100) & attention_mask
            valid_logits = expert_logits[valid_mask]
            valid_targets = target_experts[valid_mask]
            valid_confidence = confidence[valid_mask]
            
            if valid_logits.size(0) == 0:
                continue
            
            # Calculate top-k accuracy
            for k in top_k_values:
                _, top_k_pred = torch.topk(valid_logits, k, dim=-1)
                top_k_hits = (top_k_pred == valid_targets.unsqueeze(1)).any(dim=1)
                top_k_correct[k] += top_k_hits.sum().item()
            
            total_samples += valid_logits.size(0)
            total_confidence += valid_confidence.sum().item()
    
    # Calculate accuracy metrics
    accuracies = {}
    for k in top_k_values:
        accuracies[f'top_{k}_accuracy'] = top_k_correct[k] / total_samples * 100
    
    accuracies['avg_confidence'] = total_confidence / total_samples
    
    return accuracies

def train_speculative_expert_model():
    """Train the speculative expert routing model"""
    
    # Configuration
    config = {
        'num_experts': 128,
        'hidden_size': 512,
        'num_layers': 12,
        'model_dim': 256,
        'num_heads': 8,
        'ff_dim': 1024,
        'dropout': 0.1,
        'context_length': 3,
        'prediction_horizon': 2,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'gradient_clip': 1.0
    }
    
    logger.info("Starting Speculative Expert Routing Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading routing traces...")
    traces = load_traces("routing_data/maximum_real_traces.pkl")
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
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
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
        dropout=config['dropout']
    ).to(device)
    
    criterion = InterlayerLoss(num_experts=config['num_experts'])
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['num_epochs'],
        pct_start=0.1
    )
    
    # Training loop
    best_accuracy = 0
    results = []
    
    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        model.train()
        train_loss = 0
        train_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in progress_bar:
            context_experts = batch['context_experts'].to(device)
            target_experts = batch['target_experts'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            expert_logits, confidence, pattern_weights = model(context_experts, layer_ids, attention_mask)
            
            # Calculate loss
            valid_mask = (target_experts != -100) & attention_mask
            loss_dict = criterion(expert_logits, confidence, pattern_weights, target_experts, valid_mask)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            batch_samples = valid_mask.sum().item()
            train_loss += loss.item() * batch_samples
            train_samples += batch_samples
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_train_loss = train_loss / train_samples
        
        # Validation
        logger.info("Evaluating on validation set...")
        val_metrics = evaluate_speculative_accuracy(model, val_loader, device)
        
        # Log results
        result = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'learning_rate': scheduler.get_last_lr()[0],
            **val_metrics
        }
        results.append(result)
        
        logger.info(f"Results: {json.dumps(result, indent=2)}")
        
        # Save best model
        if val_metrics['top_1_accuracy'] > best_accuracy:
            best_accuracy = val_metrics['top_1_accuracy']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch + 1,
                'accuracy': best_accuracy
            }, 'best_speculative_expert_model.pth')
            logger.info(f"New best model saved! Top-1 accuracy: {best_accuracy:.2f}%")
    
    # Final evaluation
    logger.info("\nFinal evaluation on validation set:")
    final_metrics = evaluate_speculative_accuracy(model, val_loader, device)
    logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
    # Save training results
    with open('speculative_expert_training_results.json', 'w') as f:
        json.dump({
            'config': config,
            'results': results,
            'final_metrics': final_metrics
        }, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Best Top-1 Accuracy: {best_accuracy:.2f}%")
    
    return model, results

if __name__ == "__main__":
    model, results = train_speculative_expert_model()