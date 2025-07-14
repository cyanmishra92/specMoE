#!/usr/bin/env python3
"""
Enhanced Ensemble Speculation Training
Multi-predictor ensemble with entropy-aware weighting
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.ensemble_speculation_model import EnsembleSpeculationModel
from utils.data_processing import load_traces

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedEnsembleDataset(torch.utils.data.Dataset):
    """Enhanced dataset for ensemble training"""
    
    def __init__(self, traces, context_length=32, prediction_horizon=1):
        self.traces = []
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
        logger.info(f"Processing traces for ensemble training...")
        logger.info(f"Context length: {context_length}, Prediction horizon: {prediction_horizon}")
        
        # Group traces by sample_id and layer
        from collections import defaultdict
        sample_groups = defaultdict(lambda: defaultdict(list))
        
        for trace in traces:
            sample_id = trace['sample_id']
            layer_id = trace['layer_id']
            sample_groups[sample_id][layer_id].append(trace)
        
        # Process each sample
        for sample_id, layer_traces in sample_groups.items():
            for layer_id, traces_list in layer_traces.items():
                for trace in traces_list:
                    target_routing = trace['target_routing']
                    expert_sequence = np.argmax(target_routing, axis=1)
                    
                    # Create training sequences
                    for i in range(len(expert_sequence) - prediction_horizon):
                        if i >= context_length:
                            context = expert_sequence[i-context_length:i]
                            target = expert_sequence[i:i+prediction_horizon]
                            
                            self.traces.append({
                                'context': context,
                                'target': target[0],  # Single step prediction
                                'layer_id': layer_id,
                                'position': i,
                                'sample_id': sample_id,
                                'sequence_length': len(expert_sequence)
                            })
        
        logger.info(f"Created {len(self.traces)} training sequences")
        
        # Calculate layer distribution
        layer_counts = {}
        for trace in self.traces:
            layer_id = trace['layer_id']
            layer_counts[layer_id] = layer_counts.get(layer_id, 0) + 1
        
        logger.info(f"Layer distribution: {layer_counts}")
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        
        return {
            'context': torch.tensor(trace['context'], dtype=torch.long),
            'target': torch.tensor(trace['target'], dtype=torch.long),
            'layer_id': torch.tensor(trace['layer_id'], dtype=torch.long),
            'position': torch.tensor(trace['position'], dtype=torch.long),
            'sample_id': trace['sample_id']
        }

class EnsembleTrainer:
    """Enhanced trainer for ensemble speculation model"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with different learning rates for different components
        self.optimizer = optim.AdamW([
            {'params': self.model.branch_predictor.parameters(), 'lr': config['lr'] * 0.5},
            {'params': self.model.pattern_predictor.parameters(), 'lr': config['lr'] * 0.7},
            {'params': self.model.transformer_predictor.parameters(), 'lr': config['lr']},
            {'params': self.model.meta_network.parameters(), 'lr': config['lr'] * 1.5},
            {'params': [self.model.entropy_weights, self.model.predictor_weights], 'lr': config['lr'] * 0.3}
        ], weight_decay=config['weight_decay'])
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'], eta_min=1e-6)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Layer-specific metrics
        layer_metrics = {layer: {'correct': 0, 'total': 0} for layer in [1, 3, 5, 7, 9, 11]}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            context = batch['context'].to(self.device)
            target = batch['target'].to(self.device)
            layer_id = batch['layer_id'].to(self.device)
            position = batch['position'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(context, layer_id, position)
            prediction = output['prediction']
            confidence = output['confidence']
            
            # Main prediction loss
            main_loss = self.criterion(prediction, target)
            
            # Confidence regularization
            confidence_loss = torch.mean(torch.abs(confidence - 0.5))  # Encourage diverse confidence
            
            # Ensemble diversity loss
            branch_pred = output['branch_pred']
            pattern_pred = output['pattern_pred']
            transformer_pred = output['transformer_pred']
            
            # Encourage diversity between predictors
            diversity_loss = (
                torch.mean(torch.cosine_similarity(branch_pred, pattern_pred, dim=1)) +
                torch.mean(torch.cosine_similarity(pattern_pred, transformer_pred, dim=1)) +
                torch.mean(torch.cosine_similarity(branch_pred, transformer_pred, dim=1))
            ) / 3
            
            # Total loss
            total_loss_batch = (
                main_loss + 
                0.1 * confidence_loss + 
                0.05 * diversity_loss
            )
            
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Collect predictions and targets
            _, predicted = torch.max(prediction, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Layer-specific accuracy
            for i, layer in enumerate(layer_id):
                layer_key = layer.item()
                if layer_key in layer_metrics:
                    layer_metrics[layer_key]['correct'] += (predicted[i] == target[i]).item()
                    layer_metrics[layer_key]['total'] += 1
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{total_loss_batch.item():.4f}',
                    'Acc': f'{accuracy_score(all_targets[-100:], all_predictions[-100:]):.3f}'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Log layer-specific accuracies
        layer_accs = {}
        for layer, metrics in layer_metrics.items():
            if metrics['total'] > 0:
                layer_accs[layer] = metrics['correct'] / metrics['total']
        
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")
        logger.info(f"Layer accuracies: {layer_accs}")
        
        return avg_loss, accuracy, layer_accs
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Layer-specific metrics
        layer_metrics = {layer: {'correct': 0, 'total': 0} for layer in [1, 3, 5, 7, 9, 11]}
        
        # Predictor contribution analysis
        predictor_contributions = {'branch': [], 'pattern': [], 'transformer': []}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                context = batch['context'].to(self.device)
                target = batch['target'].to(self.device)
                layer_id = batch['layer_id'].to(self.device)
                position = batch['position'].to(self.device)
                
                output = self.model(context, layer_id, position)
                prediction = output['prediction']
                
                loss = self.criterion(prediction, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(prediction, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Layer-specific accuracy
                for i, layer in enumerate(layer_id):
                    layer_key = layer.item()
                    if layer_key in layer_metrics:
                        layer_metrics[layer_key]['correct'] += (predicted[i] == target[i]).item()
                        layer_metrics[layer_key]['total'] += 1
                
                # Predictor contributions
                weights = output['weights']
                predictor_contributions['branch'].append(weights[:, 0].mean().item())
                predictor_contributions['pattern'].append(weights[:, 1].mean().item())
                predictor_contributions['transformer'].append(weights[:, 2].mean().item())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Layer-specific accuracies
        layer_accs = {}
        for layer, metrics in layer_metrics.items():
            if metrics['total'] > 0:
                layer_accs[layer] = metrics['correct'] / metrics['total']
        
        # Average predictor contributions
        avg_contributions = {
            'branch': np.mean(predictor_contributions['branch']),
            'pattern': np.mean(predictor_contributions['pattern']),
            'transformer': np.mean(predictor_contributions['transformer'])
        }
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        logger.info(f"Layer accuracies: {layer_accs}")
        logger.info(f"Predictor contributions: {avg_contributions}")
        
        return avg_loss, accuracy, layer_accs, avg_contributions
    
    def train(self):
        """Full training loop"""
        logger.info("Starting ensemble training...")
        
        best_model_path = f"models/ensemble_speculation_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_loss, train_acc, train_layer_accs = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc, val_layer_accs, predictor_contributions = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'train_accuracy': train_acc,
                    'layer_accuracies': val_layer_accs,
                    'predictor_contributions': predictor_contributions
                }, best_model_path)
                
                logger.info(f"New best model saved: {val_acc:.4f}")
            
            # Early stopping
            if epoch > 10 and val_acc < max(self.val_accuracies[-10:]) - 0.05:
                logger.info("Early stopping triggered")
                break
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_accuracy:.4f}")
        return best_model_path

def main():
    """Main training function"""
    # Configuration
    config = {
        'batch_size': 64,
        'context_length': 32,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 50,
        'num_experts': 128,
        'hidden_dim': 512
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    traces = load_traces("routing_data/maximum_real_traces.pkl")
    if traces is None:
        raise ValueError("Could not load traces")
    
    # Create datasets
    logger.info("Creating datasets...")
    full_dataset = EnhancedEnsembleDataset(
        traces, 
        context_length=config['context_length']
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = EnsembleSpeculationModel(
        num_experts=config['num_experts'],
        context_length=config['context_length'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = EnsembleTrainer(model, train_loader, val_loader, device, config)
    
    # Train
    best_model_path = trainer.train()
    
    # Save training history
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_accuracies': trainer.val_accuracies,
        'config': config,
        'best_val_accuracy': trainer.best_val_accuracy
    }
    
    history_path = f"models/ensemble_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history saved to: {history_path}")
    logger.info(f"Best model saved to: {best_model_path}")
    
    return best_model_path

if __name__ == "__main__":
    main()