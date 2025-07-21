#!/usr/bin/env python3
"""
Multi-Expert Predictor for Qwen2-MoE (60 experts, top-4 routing)
More complex than Switch model - predicts multiple experts simultaneously
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence modeling"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiExpertAttention(nn.Module):
    """Multi-head attention for expert prediction"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)

class ExpertPredictionHead(nn.Module):
    """Prediction head for top-4 expert routing"""
    
    def __init__(self, d_model: int, num_experts: int = 60, experts_per_token: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        
        # Multi-layer prediction head
        self.expert_predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_experts)
        )
        
        # Auxiliary prediction for confidence scoring
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Predict expert logits
        expert_logits = self.expert_predictor(x)  # [batch, seq_len, num_experts]
        
        # Get confidence scores
        confidence = self.confidence_head(x)  # [batch, seq_len, 1]
        
        # Get top-k predictions
        top_k_logits, top_k_indices = torch.topk(expert_logits, k=self.experts_per_token, dim=-1)
        
        return {
            'expert_logits': expert_logits,
            'top_k_logits': top_k_logits,
            'top_k_indices': top_k_indices,
            'confidence': confidence
        }

class QwenMultiExpertPredictor(nn.Module):
    """
    Multi-Expert Predictor for Qwen2-MoE
    Predicts top-4 experts from 60 total experts using sequence context
    """
    
    def __init__(
        self,
        input_dim: int = 2048,  # Qwen hidden size
        d_model: int = 512,     # Internal model dimension
        num_layers: int = 6,    # Transformer layers
        num_heads: int = 8,     # Attention heads
        num_experts: int = 60,  # Total experts
        experts_per_token: int = 4,  # Top-k experts
        max_seq_len: int = 256, # Max sequence length
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers for sequence modeling
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Expert prediction head
        self.expert_head = ExpertPredictionHead(d_model, num_experts, experts_per_token)
        
        # Loss functions
        self.expert_loss_fn = nn.CrossEntropyLoss()
        self.confidence_loss_fn = nn.MSELoss()
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass
        
        Args:
            hidden_states: [batch_size, seq_len, input_dim]
            attention_mask: [batch_size, seq_len] (optional)
            
        Returns:
            Dictionary with predictions
        """
        # Project input to model dimension
        x = self.input_projection(hidden_states)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=attention_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Expert prediction
        predictions = self.expert_head(x)
        
        return predictions
    
    def compute_loss(self, predictions, targets):
        """
        Compute multi-objective loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth expert indices and routing weights
            
        Returns:
            Dictionary with loss components
        """
        expert_logits = predictions['expert_logits']
        confidence = predictions['confidence']
        
        # Multi-label loss for top-k experts
        target_experts = targets['expert_indices']  # [batch, seq_len, k]
        target_weights = targets['expert_weights']  # [batch, seq_len, k]
        
        # Create multi-hot target for all experts
        batch_size, seq_len, _ = expert_logits.shape
        multi_hot_targets = torch.zeros_like(expert_logits)
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.experts_per_token):
                    expert_idx = target_experts[b, s, k]
                    if expert_idx < self.num_experts:  # Valid expert
                        multi_hot_targets[b, s, expert_idx] = target_weights[b, s, k]
        
        # Expert prediction loss (multi-label)
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, multi_hot_targets, reduction='mean'
        )
        
        # Top-k accuracy loss (stricter)
        top_k_indices = predictions['top_k_indices']
        top_k_loss = 0.0
        
        for k in range(self.experts_per_token):
            # For each of the top-k positions, check if we predict correctly
            target_k = target_experts[:, :, k]  # [batch, seq_len]
            pred_k = top_k_indices[:, :, k]     # [batch, seq_len]
            
            # Only count valid targets (< num_experts)
            valid_mask = target_k < self.num_experts
            if valid_mask.sum() > 0:
                k_loss = F.cross_entropy(
                    expert_logits[valid_mask], 
                    target_k[valid_mask], 
                    reduction='mean'
                )
                top_k_loss += k_loss
        
        top_k_loss = top_k_loss / self.experts_per_token
        
        # Confidence loss (predict routing confidence)
        if 'confidence_target' in targets:
            confidence_target = targets['confidence_target']
            confidence_loss = self.confidence_loss_fn(confidence.squeeze(-1), confidence_target)
        else:
            confidence_loss = torch.tensor(0.0, device=expert_logits.device)
        
        # Combined loss
        total_loss = expert_loss + 0.5 * top_k_loss + 0.1 * confidence_loss
        
        return {
            'total_loss': total_loss,
            'expert_loss': expert_loss,
            'top_k_loss': top_k_loss,
            'confidence_loss': confidence_loss
        }
    
    def predict_experts(self, hidden_states, attention_mask=None, return_confidence=False):
        """
        Predict top-k experts for given hidden states
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask (optional)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predicted expert indices and optionally confidence scores
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(hidden_states, attention_mask)
            
            result = {
                'top_k_indices': predictions['top_k_indices'],
                'top_k_logits': predictions['top_k_logits']
            }
            
            if return_confidence:
                result['confidence'] = predictions['confidence']
            
            return result

def create_qwen_predictor(config_path=None, **kwargs):
    """Factory function to create Qwen predictor with optimal settings"""
    
    # Default configuration optimized for Qwen2-MoE
    default_config = {
        'input_dim': 2048,      # Qwen hidden size
        'd_model': 512,         # Internal dimension
        'num_layers': 6,        # Transformer layers
        'num_heads': 8,         # Attention heads
        'num_experts': 60,      # Total experts
        'experts_per_token': 4, # Top-k routing
        'max_seq_len': 256,     # Sequence length
        'dropout': 0.1
    }
    
    # Load config if provided
    if config_path:
        import json
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            if 'model_info' in file_config:
                default_config['num_experts'] = file_config['model_info']['num_experts']
                default_config['experts_per_token'] = file_config['model_info']['experts_per_token']
    
    # Override with kwargs
    default_config.update(kwargs)
    
    return QwenMultiExpertPredictor(**default_config)