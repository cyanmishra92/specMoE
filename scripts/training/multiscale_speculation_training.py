#!/usr/bin/env python3
"""
Multi-Scale Context Window Speculation Training
Uses multiple context windows simultaneously to capture patterns at different scales

Key Innovation:
- Short-term patterns (2 layers): Immediate dependencies
- Medium-term patterns (3 layers): Local context 
- Long-term patterns (4+ layers): Global context
- Hierarchical fusion of multi-scale features
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

class MultiScaleSpeculationModel(nn.Module):
    """Multi-scale speculation model with different context windows"""
    
    def __init__(self, num_experts=128, hidden_size=512, num_layers=12, 
                 model_dim=384, num_heads=12, ff_dim=1536, dropout=0.1):
        super().__init__()
        
        self.num_experts = num_experts
        self.model_dim = model_dim
        
        # Expert embedding shared across scales
        self.expert_embedding = nn.Embedding(num_experts, model_dim)
        
        # Multi-scale context processors
        self.short_context_model = self._build_context_model(
            context_length=2, num_heads=8, num_layers=3, name="short"
        )
        self.medium_context_model = self._build_context_model(
            context_length=3, num_heads=num_heads, num_layers=4, name="medium"
        )
        self.long_context_model = self._build_context_model(
            context_length=4, num_heads=16, num_layers=5, name="long"
        )
        
        # Hierarchical fusion layers
        self.scale_fusion_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Scale-specific adapters
        self.short_adapter = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, model_dim),
            nn.Dropout(dropout)
        )
        self.medium_adapter = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )
        self.long_adapter = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
            nn.Dropout(dropout)
        )
        
        # Learned scale weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Final prediction layers
        self.fusion_norm = nn.LayerNorm(model_dim)
        self.prediction_head = nn.Linear(model_dim, num_experts)
        self.confidence_head = nn.Linear(model_dim, 1)
        
        # Positional encodings
        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(1024, model_dim))
        
        # Initialize weights properly
        self._init_weights()
        
    def _build_context_model(self, context_length, num_heads, num_layers, name):
        """Build context-specific processing model"""
        return nn.ModuleDict({
            'layer_pos_encoding': nn.Embedding(context_length, self.model_dim),
            'attention_layers': nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.model_dim,
                    nhead=num_heads,
                    dim_feedforward=self.model_dim * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                ) for _ in range(num_layers)
            ]),
            'cross_attention': nn.MultiheadAttention(
                embed_dim=self.model_dim, num_heads=num_heads, dropout=0.1, batch_first=True
            ),
            'norm': nn.LayerNorm(self.model_dim)
        })
    
    def _create_sinusoidal_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Ensure div_term matches the dimensions
        if d_model % 2 == 1:
            div_term = div_term[:-1]  # Remove last element if d_model is odd
            
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def _init_weights(self):
        """Initialize weights to prevent NaN"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.1)  # Small gain to prevent explosion
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize scale weights to be more stable
        nn.init.constant_(self.scale_weights, 1/3)
    
    def process_context_scale(self, context_experts, scale_model, context_length, attention_mask):
        """Process context at specific scale"""
        batch_size, seq_len, num_input_layers = context_experts.shape
        
        # Use only the relevant layers for this scale
        if num_input_layers >= context_length:
            # Use last `context_length` layers
            context_subset = context_experts[:, :, -context_length:]
        else:
            # Pad if needed
            padding = torch.zeros(batch_size, seq_len, context_length - num_input_layers, 
                                dtype=context_experts.dtype, device=context_experts.device)
            context_subset = torch.cat([padding, context_experts], dim=2)
        
        # Expert embeddings
        expert_embeds = self.expert_embedding(context_subset)  # [batch, seq, context_length, model_dim]
        
        # Add layer positional encoding
        layer_pos = scale_model['layer_pos_encoding'](
            torch.arange(context_length, device=context_experts.device)
        )
        expert_embeds = expert_embeds + layer_pos.unsqueeze(0).unsqueeze(0)
        
        # Flatten for attention processing
        flat_embeds = expert_embeds.view(batch_size, seq_len * context_length, self.model_dim)
        
        # Add token positional encoding
        token_pos = self.pos_encoding[:seq_len * context_length, :self.model_dim]
        flat_embeds = flat_embeds + token_pos.unsqueeze(0)
        
        # Create attention mask for flattened sequence
        if attention_mask is not None:
            flat_mask = attention_mask.unsqueeze(-1).expand(-1, -1, context_length)
            flat_mask = flat_mask.reshape(batch_size, seq_len * context_length)
        else:
            flat_mask = None
        
        # Apply attention layers
        hidden = flat_embeds
        for attention_layer in scale_model['attention_layers']:
            if flat_mask is not None:
                attn_mask = ~flat_mask.bool()
            else:
                attn_mask = None
            hidden = attention_layer(hidden, src_key_padding_mask=attn_mask)
        
        # Cross-attention with query from last layer
        query_embeds = expert_embeds[:, :, -1, :]  # [batch, seq, model_dim]
        
        attended_output, _ = scale_model['cross_attention'](
            query_embeds, hidden, hidden,
            key_padding_mask=flat_mask
        )
        
        # Normalize
        attended_output = scale_model['norm'](attended_output)
        
        return attended_output
    
    def forward(self, context_experts, layer_ids, attention_mask):
        """Multi-scale forward pass"""
        # Process at different scales
        short_features = self.process_context_scale(
            context_experts, self.short_context_model, 2, attention_mask
        )
        medium_features = self.process_context_scale(
            context_experts, self.medium_context_model, 3, attention_mask
        )
        long_features = self.process_context_scale(
            context_experts, self.long_context_model, 4, attention_mask
        )
        
        # Apply scale-specific adapters
        short_adapted = self.short_adapter(short_features)
        medium_adapted = self.medium_adapter(medium_features)
        long_adapted = self.long_adapter(long_features)
        
        # Normalize scale weights
        scale_weights = torch.softmax(self.scale_weights, dim=0)
        
        # Weighted combination of scales
        fused_features = (scale_weights[0] * short_adapted + 
                         scale_weights[1] * medium_adapted + 
                         scale_weights[2] * long_adapted)
        
        # Additional fusion through attention
        # Stack features for attention
        stacked_features = torch.stack([short_adapted, medium_adapted, long_adapted], dim=2)
        batch_size, seq_len, num_scales, model_dim = stacked_features.shape
        stacked_features = stacked_features.view(batch_size * seq_len, num_scales, model_dim)
        
        # Self-attention across scales
        fused_scales, _ = self.scale_fusion_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Take weighted average
        scale_weights_expanded = scale_weights.view(1, num_scales, 1)
        final_features = (fused_scales * scale_weights_expanded).sum(dim=1)
        final_features = final_features.view(batch_size, seq_len, model_dim)
        
        # Combine with original fused features
        final_features = self.fusion_norm(final_features + fused_features)
        
        # Generate predictions
        expert_logits = self.prediction_head(final_features)
        confidence = torch.sigmoid(self.confidence_head(final_features)).squeeze(-1)
        
        # Return scale weights as pattern weights for analysis
        pattern_weights = scale_weights.detach()
        
        return expert_logits, confidence, pattern_weights

class MultiScaleDataset(torch.utils.data.Dataset):
    """Dataset supporting multiple context window sizes"""
    
    def __init__(self, traces, max_layers=12, max_context_length=4, prediction_horizon=2):
        self.traces = []
        self.max_layers = max_layers
        self.max_context_length = max_context_length
        self.prediction_horizon = prediction_horizon
        
        logger.info(f"Processing traces for multi-scale training...")
        logger.info(f"Max context length: {max_context_length}, Prediction horizon: {prediction_horizon}")
        
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
            
            # Need at least 2 layers for shortest context + prediction
            if len(sorted_layers) < 2 + prediction_horizon:
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
            
            # Create sequences with variable context lengths
            max_possible_context = min(len(sorted_layers) - prediction_horizon, max_context_length)
            
            for context_len in range(2, max_possible_context + 1):
                for start_layer in range(len(sorted_layers) - context_len - prediction_horizon + 1):
                    context_layers = sorted_layers[start_layer:start_layer + context_len]
                    target_layers = sorted_layers[start_layer + context_len:start_layer + context_len + prediction_horizon]
                    
                    seq_lengths = [layer_experts[layer_id].size(0) for layer_id in context_layers]
                    max_seq_len = min(seq_lengths)
                    
                    if max_seq_len == 0:
                        continue
                    
                    # Pad context to max_context_length
                    padded_context = []
                    for i in range(max_context_length):
                        if i < len(context_layers):
                            padded_context.append(layer_experts[context_layers[i]][:max_seq_len])
                        else:
                            # Pad with zeros (will be masked)
                            padded_context.append(torch.zeros(max_seq_len, dtype=torch.long))
                    
                    context_expert_seq = torch.stack(padded_context, dim=1)
                    
                    target_expert_seq = torch.stack([
                        layer_experts[layer_id][:max_seq_len] for layer_id in target_layers
                    ], dim=1)
                    
                    self.traces.append({
                        'context_experts': context_expert_seq,
                        'target_experts': target_expert_seq[:, 0],
                        'target_layer_id': torch.tensor(target_layers[0]),
                        'actual_context_length': context_len,
                        'sample_id': sample_id,
                        'seq_len': max_seq_len
                    })
        
        logger.info(f"Created {len(self.traces)} multi-scale training sequences from {len(sample_groups)} samples")
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return self.traces[idx]

def collate_fn_multiscale(batch):
    """Collate function for multi-scale training"""
    max_seq_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    max_context_length = batch[0]['context_experts'].size(1)
    
    context_experts = torch.zeros(batch_size, max_seq_len, max_context_length, dtype=torch.long)
    target_experts = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    layer_ids = torch.zeros(batch_size, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    actual_context_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        context_experts[i, :seq_len] = item['context_experts']
        target_experts[i, :seq_len] = item['target_experts']
        layer_ids[i] = item['target_layer_id']
        attention_mask[i, :seq_len] = True
        actual_context_lengths[i] = item['actual_context_length']
    
    return {
        'context_experts': context_experts,
        'target_experts': target_experts,
        'layer_ids': layer_ids,
        'attention_mask': attention_mask,
        'actual_context_lengths': actual_context_lengths
    }

def evaluate_model(model, dataloader, device):
    """Evaluate multi-scale model"""
    model.eval()
    total_samples = 0
    top_k_correct = {1: 0, 3: 0, 5: 0, 10: 0}
    total_confidence = 0.0
    scale_weights_sum = torch.zeros(3)
    
    with torch.no_grad():
        for batch in dataloader:
            context_experts = batch['context_experts'].to(device)
            target_experts = batch['target_experts'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            expert_logits, confidence, pattern_weights = model(context_experts, layer_ids, attention_mask)
            
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
            
            # Safe scale weights accumulation
            if pattern_weights is not None and not torch.isnan(pattern_weights).any():
                scale_weights_sum += pattern_weights.cpu()
    
    accuracies = {}
    for k in [1, 3, 5, 10]:
        accuracies[f'top_{k}_accuracy'] = top_k_correct[k] / total_samples * 100
    
    accuracies['avg_confidence'] = total_confidence / total_samples if total_samples > 0 else 0.0
    
    # Add scale weight analysis with NaN protection
    num_batches = len(dataloader)
    if not torch.isnan(scale_weights_sum).any() and num_batches > 0:
        avg_scale_weights = scale_weights_sum / num_batches
        accuracies['scale_weights'] = {
            'short': avg_scale_weights[0].item(),
            'medium': avg_scale_weights[1].item(),
            'long': avg_scale_weights[2].item()
        }
    else:
        accuracies['scale_weights'] = {
            'short': 0.33,
            'medium': 0.33,
            'long': 0.33
        }
    
    return accuracies

def train_multiscale_speculation():
    """Train multi-scale speculation model"""
    
    config = {
        'num_experts': 128,
        'hidden_size': 512,
        'num_layers': 12,
        'model_dim': 384,
        'num_heads': 12,
        'ff_dim': 1536,
        'dropout': 0.1,
        'max_context_length': 4,
        'prediction_horizon': 2,
        'batch_size': 20,
        'learning_rate': 3e-5,
        'num_epochs': 75,
        'warmup_steps': 1200,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'label_smoothing': 0.05
    }
    
    logger.info("Starting Multi-Scale Speculation Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    traces = load_traces("routing_data/robust_traces.pkl")
    if traces is None:
        raise ValueError("Could not load traces")
    
    # Create multi-scale dataset
    dataset = MultiScaleDataset(
        traces,
        max_layers=config['num_layers'],
        max_context_length=config['max_context_length'],
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
        collate_fn=collate_fn_multiscale,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_multiscale,
        num_workers=2
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize multi-scale model
    model = MultiScaleSpeculationModel(
        num_experts=config['num_experts'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        model_dim=config['model_dim'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout']
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-7)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config['label_smoothing'])
    
    # Training loop
    best_accuracy = 0.0
    training_results = []
    patience = 18
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
            
            expert_logits, confidence, pattern_weights = model(context_experts, layer_ids, attention_mask)
            
            valid_mask = (target_experts != -100) & attention_mask
            valid_logits = expert_logits[valid_mask]
            valid_targets = target_experts[valid_mask]
            
            if valid_logits.size(0) > 0:
                loss = criterion(valid_logits, valid_targets)
                
                # Check for NaN and skip if found
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN or Inf loss detected, skipping batch")
                    continue
                
                # Simplified loss (remove problematic scale regularization)
                total_loss = loss
                
                total_loss.backward()
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    logger.warning("NaN gradients detected, skipping optimizer step")
                    optimizer.zero_grad()
                    continue
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Safe scale weights display
                if pattern_weights is not None and not torch.isnan(pattern_weights).any():
                    scales_display = f"S:{pattern_weights[0]:.2f} M:{pattern_weights[1]:.2f} L:{pattern_weights[2]:.2f}"
                else:
                    scales_display = "S:-- M:-- L:--"
                
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                    'scales': scales_display
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
            torch.save(model.state_dict(), 'multiscale_speculation_best.pt')
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
    
    logger.info("Multi-scale training completed!")
    logger.info(f"Best Top-1 Accuracy: {best_accuracy:.2f}%")
    
    # Save results
    results = {
        'config': config,
        'training_history': training_results,
        'final_metrics': final_metrics,
        'best_accuracy': best_accuracy
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"multiscale_speculation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    model, results = train_multiscale_speculation()