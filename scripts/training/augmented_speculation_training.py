#!/usr/bin/env python3
"""
Data Augmentation for Speculation Training
Implements various augmentation techniques to improve generalization

Augmentation Techniques:
1. Expert permutation - Randomly shuffle expert indices while preserving patterns
2. Layer dropout - Randomly mask some context layers during training
3. Sequence subsampling - Use random subsequences for training
4. Noise injection - Add small noise to expert embeddings
5. Mixup - Blend expert sequences from different samples
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
import random
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.interlayer_model import InterLayerSpeculationModel
from utils.data_processing import load_traces

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AugmentedSpeculativeDataset(torch.utils.data.Dataset):
    """Dataset with augmentation techniques"""
    
    def __init__(self, traces, max_layers=12, context_length=3, prediction_horizon=2, 
                 augment_prob=0.5, expert_permute_prob=0.3, layer_dropout_prob=0.2):
        self.traces = []
        self.max_layers = max_layers
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.augment_prob = augment_prob
        self.expert_permute_prob = expert_permute_prob
        self.layer_dropout_prob = layer_dropout_prob
        
        logger.info(f"Processing traces with augmentation...")
        logger.info(f"Augmentation probability: {augment_prob}")
        logger.info(f"Expert permutation probability: {expert_permute_prob}")
        logger.info(f"Layer dropout probability: {layer_dropout_prob}")
        
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
                    'seq_len': max_seq_len,
                    'original_context': context_expert_seq.clone(),
                    'original_target': target_expert_seq[:, 0].clone()
                })
        
        logger.info(f"Created {len(self.traces)} training sequences from {len(sample_groups)} samples")
    
    def expert_permutation_augment(self, expert_seq, target_seq):
        """Randomly permute expert indices while preserving patterns"""
        num_experts = 128
        
        # Create random permutation
        perm = torch.randperm(num_experts)
        inverse_perm = torch.zeros_like(perm)
        inverse_perm[perm] = torch.arange(num_experts)
        
        # Apply permutation
        augmented_context = inverse_perm[expert_seq]
        augmented_target = inverse_perm[target_seq]
        
        return augmented_context, augmented_target
    
    def layer_dropout_augment(self, expert_seq):
        """Randomly mask some context layers"""
        seq_len, num_layers = expert_seq.shape
        
        # Randomly select layers to drop
        drop_mask = torch.rand(num_layers) < self.layer_dropout_prob
        
        # Ensure at least one layer remains
        if drop_mask.all():
            drop_mask[torch.randint(num_layers, (1,))] = False
        
        # Replace dropped layers with special token (127 for masked)
        augmented_seq = expert_seq.clone()
        augmented_seq[:, drop_mask] = 127  # Special mask token
        
        return augmented_seq
    
    def sequence_subsampling_augment(self, expert_seq, target_seq, min_length=8):
        """Use random subsequences for training"""
        seq_len = expert_seq.size(0)
        
        if seq_len <= min_length:
            return expert_seq, target_seq
        
        # Random subsequence length
        subseq_len = random.randint(min_length, seq_len)
        start_idx = random.randint(0, seq_len - subseq_len)
        
        return expert_seq[start_idx:start_idx + subseq_len], target_seq[start_idx:start_idx + subseq_len]
    
    def mixup_augment(self, expert_seq1, target_seq1, expert_seq2, target_seq2, alpha=0.2):
        """Mixup augmentation for expert sequences"""
        if expert_seq1.shape != expert_seq2.shape:
            return expert_seq1, target_seq1  # Skip if shapes don't match
        
        # Generate mixup coefficient
        lam = np.random.beta(alpha, alpha)
        
        # For discrete expert IDs, use probabilistic mixing
        # Choose first sequence with probability lam
        mix_mask = torch.rand_like(expert_seq1.float()) < lam
        mixed_context = torch.where(mix_mask, expert_seq1, expert_seq2)
        
        mix_mask_target = torch.rand_like(target_seq1.float()) < lam
        mixed_target = torch.where(mix_mask_target, target_seq1, target_seq2)
        
        return mixed_context, mixed_target
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        item = self.traces[idx].copy()
        
        # Apply augmentations during training
        if random.random() < self.augment_prob:
            context_experts = item['context_experts']
            target_experts = item['target_experts']
            
            # Expert permutation
            if random.random() < self.expert_permute_prob:
                context_experts, target_experts = self.expert_permutation_augment(
                    context_experts, target_experts)
            
            # Layer dropout
            if random.random() < self.layer_dropout_prob:
                context_experts = self.layer_dropout_augment(context_experts)
            
            # Sequence subsampling
            if random.random() < 0.3:
                context_experts, target_experts = self.sequence_subsampling_augment(
                    context_experts, target_experts)
                item['seq_len'] = context_experts.size(0)
            
            # Mixup (with another random sample)
            if random.random() < 0.2 and len(self.traces) > 1:
                other_idx = random.randint(0, len(self.traces) - 1)
                if other_idx == idx:
                    other_idx = (idx + 1) % len(self.traces)
                
                other_item = self.traces[other_idx]
                context_experts, target_experts = self.mixup_augment(
                    context_experts, target_experts,
                    other_item['context_experts'], other_item['target_experts'])
            
            item['context_experts'] = context_experts
            item['target_experts'] = target_experts
        
        return item

def collate_fn_augmented(batch):
    """Enhanced collate function for augmented data"""
    max_seq_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    context_length = batch[0]['context_experts'].size(1) if len(batch[0]['context_experts'].shape) > 1 else 3
    
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

class AugmentedInterLayerModel(InterLayerSpeculationModel):
    """Enhanced model with noise injection capability"""
    
    def __init__(self, *args, noise_std=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_std = noise_std
        
        # Add noise injection layer
        self.noise_injection = nn.Dropout(p=0.1)
    
    def forward(self, context_experts, layer_ids, attention_mask):
        # Get embeddings
        batch_size, seq_len, num_input_layers = context_experts.shape
        
        # Expert embeddings for context
        expert_embeds = self.expert_embedding(context_experts)
        
        # Add noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(expert_embeds) * self.noise_std
            expert_embeds = expert_embeds + noise
        
        # Continue with normal forward pass
        layer_pos = self.layer_pos_encoding(torch.arange(num_input_layers, device=context_experts.device))
        expert_embeds = expert_embeds + layer_pos.unsqueeze(0).unsqueeze(0)
        
        flat_embeds = expert_embeds.view(batch_size, seq_len * num_input_layers, self.model_dim)
        
        token_pos = self.token_pos_encoding[:seq_len * num_input_layers, :self.model_dim]
        flat_embeds = flat_embeds + token_pos.unsqueeze(0)
        
        if attention_mask is not None:
            flat_mask = attention_mask.unsqueeze(-1).expand(-1, -1, num_input_layers)
            flat_mask = flat_mask.reshape(batch_size, seq_len * num_input_layers)
        else:
            flat_mask = None
        
        # Apply attention layers with noise injection
        hidden = flat_embeds
        for attention_layer in self.attention_layers:
            if flat_mask is not None:
                attn_mask = ~flat_mask.bool()
            else:
                attn_mask = None
            
            # Apply noise injection
            hidden = self.noise_injection(hidden)
            hidden = attention_layer(hidden, src_key_padding_mask=attn_mask)
        
        # Continue with rest of forward pass
        query_embeds = expert_embeds[:, :, -1, :]
        
        attended_context, pattern_weights = self.cross_layer_attention(
            query_embeds.transpose(0, 1),
            hidden.transpose(0, 1),
            hidden.transpose(0, 1),
            key_padding_mask=flat_mask
        )
        
        attended_context = attended_context.transpose(0, 1)
        
        expert_logits = self.prediction_head(attended_context)
        confidence = torch.sigmoid(self.confidence_head(attended_context)).squeeze(-1)
        
        return expert_logits, confidence, pattern_weights

def evaluate_model(model, dataloader, device):
    """Evaluate augmented model"""
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

def train_augmented_speculation():
    """Train speculation model with data augmentation"""
    
    config = {
        'num_experts': 128,
        'hidden_size': 512,
        'num_layers': 12,
        'model_dim': 384,
        'num_heads': 12,
        'ff_dim': 1536,
        'num_attention_layers': 5,
        'dropout': 0.1,
        'noise_std': 0.05,
        'context_length': 3,
        'prediction_horizon': 2,
        'batch_size': 24,
        'learning_rate': 8e-5,
        'num_epochs': 80,
        'warmup_steps': 1500,
        'weight_decay': 0.015,
        'gradient_clip': 0.8,
        'label_smoothing': 0.08,
        'augment_prob': 0.6,
        'expert_permute_prob': 0.3,
        'layer_dropout_prob': 0.2
    }
    
    logger.info("Starting Augmented Speculation Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    traces = load_traces("routing_data/maximum_real_traces.pkl")
    if traces is None:
        raise ValueError("Could not load traces")
    
    # Create augmented dataset
    dataset = AugmentedSpeculativeDataset(
        traces,
        max_layers=config['num_layers'],
        context_length=config['context_length'],
        prediction_horizon=config['prediction_horizon'],
        augment_prob=config['augment_prob'],
        expert_permute_prob=config['expert_permute_prob'],
        layer_dropout_prob=config['layer_dropout_prob']
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
        collate_fn=collate_fn_augmented,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_augmented,
        num_workers=2
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize augmented model
    model = AugmentedInterLayerModel(
        num_experts=config['num_experts'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        model_dim=config['model_dim'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        num_attention_layers=config['num_attention_layers'],
        dropout=config['dropout'],
        noise_std=config['noise_std'],
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
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-7)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config['label_smoothing'])
    
    # Training loop
    best_accuracy = 0.0
    training_results = []
    patience = 20
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
                
                # Add confidence regularization
                conf_loss = 0.05 * torch.mean((confidence[valid_mask] - 0.5) ** 2)
                total_loss = loss + conf_loss
                
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
        
        # Update scheduler
        if epoch * len(train_loader) >= config['warmup_steps']:
            scheduler.step()
        
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
        
        # Save best model
        if val_metrics['top_1_accuracy'] > best_accuracy:
            best_accuracy = val_metrics['top_1_accuracy']
            torch.save(model.state_dict(), 'augmented_speculation_best.pt')
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
    
    logger.info("Augmented training completed!")
    logger.info(f"Best Top-1 Accuracy: {best_accuracy:.2f}%")
    
    # Save results
    results = {
        'config': config,
        'training_history': training_results,
        'final_metrics': final_metrics,
        'best_accuracy': best_accuracy
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"augmented_speculation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    model, results = train_augmented_speculation()