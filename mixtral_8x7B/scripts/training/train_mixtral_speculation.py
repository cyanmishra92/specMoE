#!/usr/bin/env python3
"""
Train Mixtral 8x7B Expert Speculation Models
Optimized for RTX 3090 with memory-efficient training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MixtralDataset(Dataset):
    """Dataset for Mixtral 8x7B expert speculation training"""
    
    def __init__(self, traces, context_length=3, prediction_horizon=2):
        self.traces = traces
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        """Create training sequences from traces"""
        sequences = []
        
        # Group traces by layer and sample
        layer_groups = {}
        for trace in self.traces:
            key = f"{trace.dataset_name}_{trace.sample_id}_{trace.layer_id}"
            if key not in layer_groups:
                layer_groups[key] = []
            layer_groups[key].append(trace)
        
        # Create sequences
        for key, traces in layer_groups.items():
            if len(traces) < self.context_length + self.prediction_horizon:
                continue
                
            # Sort by sequence position
            traces.sort(key=lambda x: int(x.sample_id.split('_')[-1]) if '_' in x.sample_id else 0)
            
            for i in range(len(traces) - self.context_length - self.prediction_horizon + 1):
                context_traces = traces[i:i+self.context_length]
                target_traces = traces[i+self.context_length:i+self.context_length+self.prediction_horizon]
                
                # Extract features
                context_experts = []
                context_layers = []
                
                for trace in context_traces:
                    # Get top-2 experts (Mixtral uses top-2 routing)
                    if trace.target_top_k.numel() >= 2:
                        top_experts = trace.target_top_k.flatten()[:2]
                        context_experts.append(top_experts)
                        context_layers.append(trace.layer_id)
                
                # Target expert for next token
                if target_traces and target_traces[0].target_top_k.numel() >= 1:
                    target_expert = target_traces[0].target_top_k.flatten()[0]
                    target_layer = target_traces[0].layer_id
                    
                    sequences.append({
                        'context_experts': torch.stack(context_experts) if context_experts else torch.zeros(self.context_length, 2),
                        'context_layers': torch.tensor(context_layers, dtype=torch.long),
                        'target_expert': target_expert,
                        'target_layer': target_layer,
                        'hidden_states': context_traces[-1].hidden_states if context_traces else None
                    })
        
        logger.info(f"Created {len(sequences)} training sequences")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

class MixtralSpeculationModel(nn.Module):
    """Mixtral 8x7B Expert Speculation Model"""
    
    def __init__(self, num_experts=8, hidden_size=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
        # Expert embedding for top-2 routing
        self.expert_embedding = nn.Embedding(num_experts, hidden_size)
        self.layer_embedding = nn.Embedding(32, hidden_size)  # Mixtral has 32 layers
        
        # Context processing
        self.context_encoder = nn.LSTM(
            input_size=hidden_size * 2,  # expert + layer embeddings
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, context_experts, context_layers):
        batch_size, seq_len, top_k = context_experts.shape
        
        # Handle top-2 routing: average the embeddings
        expert_embeds = self.expert_embedding(context_experts)  # [batch, seq_len, top_k, hidden]
        expert_embeds = expert_embeds.mean(dim=2)  # Average top-2 experts
        
        layer_embeds = self.layer_embedding(context_layers)  # [batch, seq_len, hidden]
        
        # Combine embeddings
        combined_embeds = torch.cat([expert_embeds, layer_embeds], dim=-1)
        
        # Process context
        lstm_out, _ = self.context_encoder(combined_embeds)
        
        # Use last hidden state for prediction
        final_hidden = lstm_out[:, -1, :]
        
        # Predictions
        expert_logits = self.prediction_head(final_hidden)
        confidence = self.confidence_head(final_hidden)
        
        return expert_logits, confidence

def calculate_accuracies(predictions, targets):
    """Calculate top-k accuracies for Mixtral"""
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Get top-k predictions
    top_k_predictions = np.argsort(predictions_np, axis=1)[:, -5:]  # Top 5 for 8 experts
    
    # Calculate accuracies
    total_samples = len(targets_np)
    accuracies = {}
    
    for k in [1, 2, 3, 5]:
        if k > predictions_np.shape[1]:
            continue
            
        correct = 0
        for i, target in enumerate(targets_np):
            if target in top_k_predictions[i, -k:]:
                correct += 1
        
        accuracies[f'top_{k}_accuracy'] = correct / total_samples * 100
    
    return accuracies

def train_mixtral_speculation():
    """Train Mixtral expert speculation model"""
    
    # Configuration for RTX 3090
    config = {
        'batch_size': 16,  # Reduced for RTX 3090
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'patience': 10,
        'context_length': 3,
        'prediction_horizon': 2
    }
    
    logger.info("🚀 Starting Mixtral 8x7B Expert Speculation Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load traces
    traces_path = Path("routing_data/mixtral_8x7b_traces.pkl")
    if not traces_path.exists():
        logger.error(f"Traces not found at {traces_path}")
        logger.error("Please run collect_mixtral_traces.py first")
        return
    
    logger.info(f"Loading traces from {traces_path}")
    with open(traces_path, 'rb') as f:
        traces = pickle.load(f)
    
    logger.info(f"Loaded {len(traces)} traces")
    
    # Create dataset
    dataset = MixtralDataset(
        traces, 
        context_length=config['context_length'],
        prediction_horizon=config['prediction_horizon']
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = MixtralSpeculationModel(
        num_experts=8,  # Mixtral has 8 experts
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training loop
    best_val_accuracy = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    logger.info("Starting training...")
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            # Move to device
            context_experts = batch['context_experts'].to(device)
            context_layers = batch['context_layers'].to(device)
            target_expert = batch['target_expert'].to(device)
            
            # Forward pass
            expert_logits, confidence = model(context_experts, context_layers)
            
            # Calculate loss
            loss = criterion(expert_logits, target_expert)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
                context_experts = batch['context_experts'].to(device)
                context_layers = batch['context_layers'].to(device)
                target_expert = batch['target_expert'].to(device)
                
                expert_logits, confidence = model(context_experts, context_layers)
                loss = criterion(expert_logits, target_expert)
                
                total_val_loss += loss.item()
                val_predictions.append(expert_logits)
                val_targets.append(target_expert)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate accuracies
        val_predictions_tensor = torch.cat(val_predictions, dim=0)
        val_targets_tensor = torch.cat(val_targets, dim=0)
        val_acc_metrics = calculate_accuracies(val_predictions_tensor, val_targets_tensor)
        val_accuracy = val_acc_metrics['top_1_accuracy']
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Logging
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Val Accuracy: {val_accuracy:.2f}%")
        for k in [1, 2, 3, 5]:
            if f'top_{k}_accuracy' in val_acc_metrics:
                logger.info(f"  Top-{k} Accuracy: {val_acc_metrics[f'top_{k}_accuracy']:.2f}%")
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'config': config
            }, 'models/mixtral_best_model.pth')
            
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"🎉 Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, 'models/mixtral_final_model.pth')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/mixtral_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Training curves saved to models/mixtral_training_curves.png")

if __name__ == "__main__":
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    train_mixtral_speculation()