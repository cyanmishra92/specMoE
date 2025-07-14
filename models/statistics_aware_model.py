#!/usr/bin/env python3
"""
Statistics-Aware Expert Speculation Model
Incorporates layer-specific statistics for improved prediction accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

class StatisticsAwareSpeculationModel(nn.Module):
    """
    Model that uses layer statistics as prior knowledge for expert prediction
    
    Key innovations:
    1. Layer-specific priors based on real statistics
    2. Adaptive prediction strategies per layer
    3. Statistical features as additional inputs
    4. Entropy-aware attention mechanisms
    """
    
    def __init__(self, num_experts=128, hidden_size=512, model_dim=384, 
                 num_heads=12, ff_dim=1536, dropout=0.1, num_attention_layers=6,
                 context_length=3, prediction_horizon=2):
        super().__init__()
        
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.model_dim = model_dim
        self.num_attention_layers = num_attention_layers
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
        # Layer statistics from real data analysis
        self.layer_stats = {
            1: {'entropy': 6.77, 'cv': 0.596, 'coverage': 0.992, 'mean': 1055.54, 'std': 629.35},
            3: {'entropy': 6.19, 'cv': 1.658, 'coverage': 0.984, 'mean': 1055.54, 'std': 1750.22},
            5: {'entropy': 6.32, 'cv': 1.180, 'coverage': 0.977, 'mean': 1055.54, 'std': 1245.64},
            7: {'entropy': 6.06, 'cv': 1.413, 'coverage': 0.969, 'mean': 1055.54, 'std': 1491.90},
            9: {'entropy': 5.99, 'cv': 1.514, 'coverage': 0.969, 'mean': 1055.54, 'std': 1598.10},
            11: {'entropy': 5.40, 'cv': 2.227, 'coverage': 0.953, 'mean': 1055.54, 'std': 2350.47}
        }
        
        # Create learnable statistical embeddings
        self.stats_embedding = nn.ModuleDict({
            str(layer): nn.Linear(5, model_dim // 4)  # 5 statistical features
            for layer in [1, 3, 5, 7, 9, 11]
        })
        
        # Expert embedding with statistical conditioning
        self.expert_embedding = nn.Embedding(num_experts, model_dim)
        
        # Layer-specific embeddings
        self.layer_embedding = nn.Embedding(12, model_dim)
        
        # Position embedding
        self.position_embedding = nn.Embedding(1024, model_dim)
        
        # Statistical feature processor
        self.stats_processor = nn.Sequential(
            nn.Linear(model_dim // 4, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, model_dim)
        )
        
        # Entropy-aware attention layers
        self.entropy_attention_layers = nn.ModuleList([
            EntropyAwareAttention(model_dim, num_heads, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # Cross-layer statistical attention
        self.cross_layer_stats_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer-specific expert predictors (adapted to statistics)
        self.layer_predictors = nn.ModuleDict({
            str(layer): self._create_layer_predictor(layer, model_dim, ff_dim, dropout)
            for layer in [1, 3, 5, 7, 9, 11]
        })
        
        # Statistical prior networks
        self.stat_prior_networks = nn.ModuleDict({
            str(layer): nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim // 2, num_experts)
            ) for layer in [1, 3, 5, 7, 9, 11]
        })
        
        # Adaptive weighting based on layer characteristics
        self.adaptive_weights = nn.ModuleDict({
            str(layer): nn.Sequential(
                nn.Linear(model_dim + 5, model_dim // 2),  # +5 for stats
                nn.ReLU(),
                nn.Linear(model_dim // 2, 2),  # Weight between learned and prior
                nn.Softmax(dim=-1)
            ) for layer in [1, 3, 5, 7, 9, 11]
        })
        
        # Confidence predictor with statistical awareness
        self.confidence_predictor = nn.Sequential(
            nn.Linear(model_dim + 5, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _create_layer_predictor(self, layer_id, model_dim, ff_dim, dropout):
        """Create layer-specific predictor adapted to layer statistics"""
        stats = self.layer_stats[layer_id]
        
        # Adapt architecture based on layer characteristics
        if stats['entropy'] > 6.5:  # High entropy layers (1)
            # More capacity for diverse predictions
            return nn.Sequential(
                nn.Linear(model_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, self.num_experts)
            )
        elif stats['cv'] > 2.0:  # High specialization layers (11)
            # More focused predictions
            return nn.Sequential(
                nn.Linear(model_dim, ff_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim // 2, self.num_experts)
            )
        else:  # Medium entropy/specialization layers
            return nn.Sequential(
                nn.Linear(model_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, ff_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim // 2, self.num_experts)
            )
    
    def _get_layer_stats_tensor(self, layer_id, device):
        """Convert layer statistics to tensor"""
        stats = self.layer_stats.get(layer_id, self.layer_stats[1])
        return torch.tensor([
            stats['entropy'],
            stats['cv'],
            stats['coverage'],
            stats['mean'] / 10000.0,  # Normalize
            stats['std'] / 10000.0    # Normalize
        ], device=device, dtype=torch.float32)
    
    def _init_weights(self):
        """Initialize weights with statistical priors"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Initialize expert embeddings with statistical awareness
        for layer_id in [1, 3, 5, 7, 9, 11]:
            stats = self.layer_stats[layer_id]
            
            # Initialize layer predictor biases based on statistics
            if str(layer_id) in self.layer_predictors:
                predictor = self.layer_predictors[str(layer_id)]
                if hasattr(predictor, 'bias') and predictor.bias is not None:
                    # Bias towards more common experts based on entropy
                    bias_init = -stats['entropy'] / 10.0  # Lower entropy = higher bias
                    nn.init.constant_(predictor.bias, bias_init)
    
    def forward(self, context_experts, target_layer_id, context_layers, seq_lengths):
        """
        Forward pass with statistical awareness
        
        Args:
            context_experts: [batch_size, seq_len, context_length] expert IDs
            target_layer_id: [batch_size] target layer ID
            context_layers: [batch_size, context_length] context layer IDs
            seq_lengths: [batch_size] actual sequence lengths
        """
        batch_size, seq_len, context_length = context_experts.shape
        device = context_experts.device
        
        # Get statistical features for target layer
        target_stats = []
        for i in range(batch_size):
            layer_id = target_layer_id[i].item()
            stats_tensor = self._get_layer_stats_tensor(layer_id, device)
            target_stats.append(stats_tensor)
        
        target_stats = torch.stack(target_stats)  # [batch_size, 5]
        
        # Process statistical features
        stats_features = []
        for i in range(batch_size):
            layer_id = target_layer_id[i].item()
            stats_emb = self.stats_embedding[str(layer_id)](target_stats[i])
            stats_features.append(stats_emb)
        
        stats_features = torch.stack(stats_features)  # [batch_size, model_dim//4]
        stats_features = self.stats_processor(stats_features)  # [batch_size, model_dim]
        
        # Expert embeddings
        expert_emb = self.expert_embedding(context_experts)  # [batch_size, seq_len, context_length, model_dim]
        
        # Layer embeddings
        layer_emb = self.layer_embedding(context_layers)  # [batch_size, context_length, model_dim]
        layer_emb = layer_emb.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch_size, seq_len, context_length, model_dim]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # [batch_size, seq_len, model_dim]
        pos_emb = pos_emb.unsqueeze(2).expand(-1, -1, context_length, -1)  # [batch_size, seq_len, context_length, model_dim]
        
        # Combine embeddings
        combined_emb = expert_emb + layer_emb + pos_emb
        
        # Reshape for attention
        combined_emb = combined_emb.view(batch_size, seq_len * context_length, self.model_dim)
        
        # Add statistical features to each token
        stats_features_expanded = stats_features.unsqueeze(1).expand(-1, seq_len * context_length, -1)
        combined_emb = combined_emb + stats_features_expanded
        
        # Apply entropy-aware attention layers
        attention_output = combined_emb
        for layer_idx, attention_layer in enumerate(self.entropy_attention_layers):
            attention_output = attention_layer(attention_output, target_stats)
        
        # Cross-layer statistical attention
        cross_attended, _ = self.cross_layer_stats_attention(
            attention_output, attention_output, attention_output
        )
        
        # Pool across context dimension
        pooled_output = cross_attended.view(batch_size, seq_len, context_length, self.model_dim)
        pooled_output = torch.mean(pooled_output, dim=2)  # [batch_size, seq_len, model_dim]
        
        # Final sequence representation
        sequence_repr = torch.mean(pooled_output, dim=1)  # [batch_size, model_dim]
        
        # Layer-specific predictions
        expert_logits = []
        stat_priors = []
        adaptive_weights = []
        
        for i in range(batch_size):
            layer_id = target_layer_id[i].item()
            
            # Get layer-specific prediction
            layer_pred = self.layer_predictors[str(layer_id)](sequence_repr[i:i+1])
            
            # Get statistical prior
            stat_prior = self.stat_prior_networks[str(layer_id)](stats_features[i:i+1])
            
            # Get adaptive weight
            combined_input = torch.cat([sequence_repr[i:i+1], target_stats[i:i+1]], dim=1)
            adaptive_weight = self.adaptive_weights[str(layer_id)](combined_input)
            
            expert_logits.append(layer_pred)
            stat_priors.append(stat_prior)
            adaptive_weights.append(adaptive_weight)
        
        expert_logits = torch.cat(expert_logits, dim=0)
        stat_priors = torch.cat(stat_priors, dim=0)
        adaptive_weights = torch.cat(adaptive_weights, dim=0)
        
        # Combine learned predictions with statistical priors
        final_logits = (adaptive_weights[:, 0:1] * expert_logits + 
                       adaptive_weights[:, 1:2] * stat_priors)
        
        # Expand to match sequence length
        final_logits = final_logits.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Confidence prediction
        confidence_input = torch.cat([sequence_repr, target_stats], dim=1)
        confidence = self.confidence_predictor(confidence_input)
        
        return {
            'expert_logits': final_logits,
            'confidence': confidence,
            'statistical_priors': stat_priors,
            'adaptive_weights': adaptive_weights
        }

class EntropyAwareAttention(nn.Module):
    """Attention mechanism that adapts based on layer entropy"""
    
    def __init__(self, model_dim, num_heads, dropout):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Entropy-based attention scaling
        self.entropy_scaler = nn.Sequential(
            nn.Linear(5, model_dim // 4),  # 5 statistical features
            nn.ReLU(),
            nn.Linear(model_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, stats):
        """
        Args:
            x: [batch_size, seq_len, model_dim]
            stats: [batch_size, 5] statistical features
        """
        # Calculate entropy-based scaling
        entropy_scale = self.entropy_scaler(stats).unsqueeze(1)  # [batch_size, 1, 1]
        
        # Apply attention
        attended, attention_weights = self.attention(x, x, x)
        
        # Scale attention based on entropy
        attended = attended * entropy_scale
        
        # Residual connection and layer norm
        output = self.layer_norm(x + self.dropout(attended))
        
        return output