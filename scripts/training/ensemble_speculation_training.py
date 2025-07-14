#!/usr/bin/env python3
"""
Ensemble Speculation Training
Train multiple diverse models and combine their predictions for better accuracy

Approaches:
1. Multiple model architectures with different configurations
2. Weighted ensemble based on validation performance
3. Temperature scaling for calibrated confidence
4. Advanced ensemble techniques (voting, stacking)
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
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
    """Dataset for ensemble training"""
    
    def __init__(self, traces, max_layers=12, context_length=3, prediction_horizon=2):
        self.traces = []
        self.max_layers = max_layers
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
        logger.info(f"Processing traces for ensemble training...")
        
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
    """Collate function for ensemble training"""
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

class EnsembleModel(nn.Module):
    """Ensemble of multiple speculation models"""
    
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        self.weights = nn.Parameter(torch.tensor(self.weights), requires_grad=False)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, context_experts, layer_ids, attention_mask):
        """Forward pass through ensemble"""
        all_logits = []
        all_confidence = []
        
        for model in self.models:
            logits, confidence, _ = model(context_experts, layer_ids, attention_mask)
            all_logits.append(logits)
            all_confidence.append(confidence)
        
        # Weighted average of logits
        ensemble_logits = torch.zeros_like(all_logits[0])
        for i, (logits, weight) in enumerate(zip(all_logits, self.weights)):
            ensemble_logits += weight * logits
        
        # Apply temperature scaling
        ensemble_logits = ensemble_logits / self.temperature
        
        # Average confidence
        ensemble_confidence = torch.mean(torch.stack(all_confidence), dim=0)
        
        return ensemble_logits, ensemble_confidence, None

def evaluate_model(model, dataloader, device):
    """Evaluate ensemble model"""
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
            
            # Calculate top-k accuracy
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

def train_individual_model(config, train_loader, val_loader, device, model_name):
    """Train individual model for ensemble"""
    logger.info(f"Training {model_name}...")
    
    # Create model with specific config
    model = InterLayerSpeculationModel(**config['model_params']).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    if config.get('use_cosine_schedule', False):
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config.get('label_smoothing', 0.0))
    
    best_accuracy = 0.0
    patience = config.get('patience', 10)
    no_improve = 0
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            context_experts = batch['context_experts'].to(device)
            target_experts = batch['target_experts'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            expert_logits, confidence, _ = model(context_experts, layer_ids, attention_mask)
            
            valid_mask = (target_experts != -100) & attention_mask
            valid_logits = expert_logits[valid_mask]
            valid_targets = target_experts[valid_mask]
            
            if valid_logits.size(0) > 0:
                loss = criterion(valid_logits, valid_targets)
                loss.backward()
                
                if config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        current_accuracy = val_metrics['top_1_accuracy']
        
        # Scheduler step
        if config.get('use_cosine_schedule', False):
            scheduler.step()
        else:
            scheduler.step(current_accuracy)
        
        # Early stopping
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), f'{model_name}_best.pt')
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            logger.info(f"{model_name} early stopping at epoch {epoch+1}")
            break
        
        if epoch % 10 == 0:
            logger.info(f"{model_name} Epoch {epoch+1}: Acc={current_accuracy:.2f}%, Best={best_accuracy:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load(f'{model_name}_best.pt'))
    logger.info(f"{model_name} training completed. Best accuracy: {best_accuracy:.2f}%")
    
    return model, best_accuracy

def train_ensemble_speculation():
    """Train ensemble of speculation models"""
    
    logger.info("Starting Ensemble Speculation Training")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    traces = load_traces("routing_data/maximum_real_traces.pkl")
    if traces is None:
        raise ValueError("Could not load traces")
    
    # Create dataset
    dataset = SpeculativeExpertDataset(traces)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Define ensemble configurations
    ensemble_configs = [
        {
            'name': 'Large_Model',
            'model_params': {
                'num_experts': 128,
                'hidden_size': 512,
                'num_layers': 12,
                'model_dim': 512,
                'num_heads': 16,
                'ff_dim': 2048,
                'num_attention_layers': 6,
                'dropout': 0.1
            },
            'learning_rate': 5e-5,
            'weight_decay': 0.02,
            'num_epochs': 50,
            'use_cosine_schedule': True,
            'gradient_clip': 1.0,
            'label_smoothing': 0.1
        },
        {
            'name': 'Medium_Model',
            'model_params': {
                'num_experts': 128,
                'hidden_size': 512,
                'num_layers': 12,
                'model_dim': 384,
                'num_heads': 12,
                'ff_dim': 1536,
                'num_attention_layers': 4,
                'dropout': 0.15
            },
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'num_epochs': 60,
            'use_cosine_schedule': False,
            'gradient_clip': 0.5,
            'label_smoothing': 0.05
        },
        {
            'name': 'Compact_Model',
            'model_params': {
                'num_experts': 128,
                'hidden_size': 512,
                'num_layers': 12,
                'model_dim': 256,
                'num_heads': 8,
                'ff_dim': 1024,
                'num_attention_layers': 4,
                'dropout': 0.2
            },
            'learning_rate': 2e-4,
            'weight_decay': 0.005,
            'num_epochs': 70,
            'use_cosine_schedule': False,
            'gradient_clip': 0.3,
            'label_smoothing': 0.0
        }
    ]
    
    # Train individual models
    models = []
    individual_accuracies = []
    
    for config in ensemble_configs:
        model, best_acc = train_individual_model(config, train_loader, val_loader, device, config['name'])
        models.append(model)
        individual_accuracies.append(best_acc)
    
    # Create ensemble with performance-based weights
    total_acc = sum(individual_accuracies)
    weights = [acc / total_acc for acc in individual_accuracies]
    
    logger.info("Individual model performance:")
    for i, (config, acc, weight) in enumerate(zip(ensemble_configs, individual_accuracies, weights)):
        logger.info(f"  {config['name']}: {acc:.2f}% (weight: {weight:.3f})")
    
    # Create ensemble model
    ensemble = EnsembleModel(models, weights).to(device)
    
    # Evaluate ensemble
    logger.info("Evaluating ensemble...")
    ensemble_metrics = evaluate_model(ensemble, val_loader, device)
    
    logger.info(f"Ensemble Results: {json.dumps(ensemble_metrics, indent=2)}")
    
    # Save ensemble
    torch.save({
        'models': [model.state_dict() for model in models],
        'weights': weights,
        'configs': ensemble_configs,
        'ensemble_metrics': ensemble_metrics,
        'individual_accuracies': individual_accuracies
    }, 'ensemble_speculation_model.pt')
    
    logger.info("Ensemble training completed!")
    logger.info(f"Ensemble Top-1 Accuracy: {ensemble_metrics['top_1_accuracy']:.2f}%")
    
    return ensemble, ensemble_metrics

if __name__ == "__main__":
    ensemble, results = train_ensemble_speculation()