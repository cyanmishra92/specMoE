#!/usr/bin/env python3
"""
Efficient Ensemble Training
Fast, regularized training with better convergence
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.efficient_ensemble_model import EfficientEnsembleModel, EfficientEnsembleDataset
from utils.data_processing import load_traces

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EfficientTrainer:
    """Fast, regularized trainer"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Simple loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Single optimizer with proper weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['lr'],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
    def train_epoch(self, epoch):
        """Fast training epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Layer-specific metrics
        layer_metrics = {layer: {'correct': 0, 'total': 0} for layer in [1, 3, 5, 7, 9, 11]}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            context = batch['context'].to(self.device, non_blocking=True)
            target = batch['target'].to(self.device, non_blocking=True)
            layer_id = batch['layer_id'].to(self.device, non_blocking=True)
            position = batch['position'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(context, layer_id, position)
            prediction = output['prediction']
            
            # Simple loss
            loss = self.criterion(prediction, target)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
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
            if batch_idx % 50 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{current_lr:.2e}',
                    'Acc': f'{accuracy_score(all_targets[-1000:], all_predictions[-1000:]):.3f}'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Layer-specific accuracies
        layer_accs = {}
        for layer, metrics in layer_metrics.items():
            if metrics['total'] > 0:
                layer_accs[layer] = metrics['correct'] / metrics['total']
        
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")
        logger.info(f"Layer accuracies: {layer_accs}")
        
        return avg_loss, accuracy, layer_accs
    
    def validate(self):
        """Fast validation"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Layer-specific metrics
        layer_metrics = {layer: {'correct': 0, 'total': 0} for layer in [1, 3, 5, 7, 9, 11]}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                context = batch['context'].to(self.device, non_blocking=True)
                target = batch['target'].to(self.device, non_blocking=True)
                layer_id = batch['layer_id'].to(self.device, non_blocking=True)
                position = batch['position'].to(self.device, non_blocking=True)
                
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
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Layer-specific accuracies
        layer_accs = {}
        for layer, metrics in layer_metrics.items():
            if metrics['total'] > 0:
                layer_accs[layer] = metrics['correct'] / metrics['total']
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        logger.info(f"Layer accuracies: {layer_accs}")
        
        return avg_loss, accuracy, layer_accs
    
    def train(self):
        """Fast training loop with early stopping"""
        logger.info("Starting efficient ensemble training...")
        
        best_model_path = f"models/efficient_ensemble_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_loss, train_acc, train_layer_accs = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate every epoch
            val_loss, val_acc, val_layer_accs = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Check for overfitting
            overfitting_gap = train_acc - val_acc
            if overfitting_gap > 0.1:  # 10% gap indicates overfitting
                logger.warning(f"Overfitting detected! Gap: {overfitting_gap:.3f}")
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'train_accuracy': train_acc,
                    'layer_accuracies': val_layer_accs,
                    'overfitting_gap': overfitting_gap
                }, best_model_path)
                
                logger.info(f"New best model saved: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                logger.info(f"Early stopping after {epoch} epochs")
                break
            
            # Stop if severely overfitting
            if overfitting_gap > 0.15:
                logger.warning("Severe overfitting detected, stopping training")
                break
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_accuracy:.4f}")
        return best_model_path

def main():
    """Main training function"""
    # Efficient configuration
    config = {
        'batch_size': 256,  # Increased for better GPU utilization
        'context_length': 32,
        'lr': 3e-4,  # Single learning rate
        'weight_decay': 0.01,  # Increased weight decay
        'epochs': 30,  # Reduced epochs with early stopping
        'patience': 5,  # Early stopping patience
        'num_experts': 128,
        'hidden_dim': 256,
        'max_samples_per_layer': 15000  # Limit samples per layer
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    traces = load_traces("routing_data/maximum_real_traces.pkl")
    if traces is None:
        raise ValueError("Could not load traces")
    
    # Create efficient dataset
    logger.info("Creating efficient dataset...")
    full_dataset = EfficientEnsembleDataset(
        traces,
        context_length=config['context_length'],
        max_samples_per_layer=config['max_samples_per_layer']
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create efficient data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,  # Increased workers
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create efficient model
    model = EfficientEnsembleModel(
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
    trainer = EfficientTrainer(model, train_loader, val_loader, device, config)
    
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
    
    history_path = f"models/efficient_ensemble_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history saved to: {history_path}")
    logger.info(f"Best model saved to: {best_model_path}")
    
    return best_model_path

if __name__ == "__main__":
    main()