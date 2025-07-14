#!/usr/bin/env python3
"""
Ensemble Expert Speculation Model
Combines multiple prediction strategies with entropy-aware weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class BranchPredictorModule(nn.Module):
    """Branch predictor-style expert prediction using local and global patterns"""
    
    def __init__(self, num_experts=128, context_length=8, hidden_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.context_length = context_length
        
        # Local pattern predictor (recent expert sequence)
        self.local_predictor = nn.Sequential(
            nn.Embedding(num_experts, hidden_dim // 2),
            nn.LSTM(hidden_dim // 2, hidden_dim // 2, batch_first=True),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Global pattern predictor (position-aware)
        self.global_predictor = nn.Sequential(
            nn.Linear(context_length + 1, hidden_dim),  # +1 for position
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Confidence predictor (hidden_dim//2 + context_length + 1)
        confidence_input_dim = hidden_dim // 2 + context_length + 1
        self.confidence_head = nn.Linear(confidence_input_dim, 1)
        
    def forward(self, expert_sequence, position):
        """
        Args:
            expert_sequence: [batch_size, seq_len] recent expert selections
            position: [batch_size] current position in sequence
        """
        batch_size = expert_sequence.size(0)
        
        # Local prediction using LSTM
        embedded = self.local_predictor[0](expert_sequence)
        lstm_out, _ = self.local_predictor[1](embedded)
        local_pred = self.local_predictor[2](lstm_out[:, -1, :])
        
        # Global prediction using position and pattern
        position_normalized = position.float().unsqueeze(1) / 512.0  # Normalize position
        
        # Create pattern features from expert sequence
        pattern_features = torch.zeros(batch_size, self.context_length, device=expert_sequence.device)
        for i in range(min(self.context_length, expert_sequence.size(1))):
            pattern_features[:, i] = expert_sequence[:, -(i+1)].float() / self.num_experts
        
        global_input = torch.cat([pattern_features, position_normalized], dim=1)
        global_pred = self.global_predictor(global_input)
        
        # Confidence estimation
        combined_features = torch.cat([lstm_out[:, -1, :], global_input], dim=1)
        confidence = torch.sigmoid(self.confidence_head(combined_features))
        
        return local_pred, global_pred, confidence

class PatternBasedPredictor(nn.Module):
    """Pattern-based predictor using attention mechanism"""
    
    def __init__(self, num_experts=128, context_length=16, hidden_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.context_length = context_length
        
        # Expert embedding
        self.expert_embedding = nn.Embedding(num_experts, hidden_dim)
        
        # Multi-head attention for pattern recognition
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Pattern classification head
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Layer-specific adaptation
        self.layer_adaptation = nn.ModuleDict({
            str(layer): nn.Linear(hidden_dim, hidden_dim)
            for layer in [1, 3, 5, 7, 9, 11]
        })
        
    def forward(self, expert_sequence, layer_id):
        """
        Args:
            expert_sequence: [batch_size, seq_len] expert sequence
            layer_id: [batch_size] layer ID for adaptation
        """
        batch_size = expert_sequence.size(0)
        
        # Embed expert sequence
        embedded = self.expert_embedding(expert_sequence)
        
        # Apply attention to find patterns
        attended, attention_weights = self.attention(embedded, embedded, embedded)
        
        # Use last attended representation
        pattern_repr = attended[:, -1, :]
        
        # Layer-specific adaptation
        adapted_features = []
        for i in range(batch_size):
            layer_key = str(layer_id[i].item())
            if layer_key in self.layer_adaptation:
                adapted = self.layer_adaptation[layer_key](pattern_repr[i:i+1])
                adapted_features.append(adapted)
            else:
                adapted_features.append(pattern_repr[i:i+1])
        
        adapted_repr = torch.cat(adapted_features, dim=0)
        
        # Predict next expert
        prediction = self.pattern_classifier(adapted_repr)
        
        return prediction, attention_weights

class TransformerSpeculationModel(nn.Module):
    """Enhanced transformer model for expert speculation"""
    
    def __init__(self, num_experts=128, context_length=32, hidden_dim=512):
        super().__init__()
        self.num_experts = num_experts
        self.context_length = context_length
        
        # Enhanced transformer with layer-specific tokens
        self.expert_embedding = nn.Embedding(num_experts, hidden_dim)
        self.layer_embedding = nn.Embedding(12, hidden_dim)  # Layer embeddings
        self.position_embedding = nn.Embedding(context_length, hidden_dim)
        
        # Multi-layer transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Layer-specific prediction heads
        self.layer_heads = nn.ModuleDict({
            str(layer): nn.Linear(hidden_dim, num_experts)
            for layer in [1, 3, 5, 7, 9, 11]
        })
        
        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, expert_sequence, layer_id, position=None):
        """
        Args:
            expert_sequence: [batch_size, seq_len] expert sequence
            layer_id: [batch_size] layer ID
            position: [batch_size] position in sequence
        """
        batch_size, seq_len = expert_sequence.shape
        
        # Create embeddings
        expert_emb = self.expert_embedding(expert_sequence)
        
        # Add layer information
        layer_emb = self.layer_embedding(layer_id).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Add position information
        if position is None:
            position = torch.arange(seq_len, device=expert_sequence.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(position % self.context_length)
        
        # Combine embeddings
        combined_emb = expert_emb + layer_emb + pos_emb
        
        # Apply transformer
        transformer_out = self.transformer(combined_emb)
        
        # Use last token representation
        last_repr = transformer_out[:, -1, :]
        
        # Layer-specific prediction
        predictions = []
        confidences = []
        
        for i in range(batch_size):
            layer_key = str(layer_id[i].item())
            if layer_key in self.layer_heads:
                pred = self.layer_heads[layer_key](last_repr[i:i+1])
                predictions.append(pred)
            else:
                # Default prediction for unknown layers
                pred = self.layer_heads['1'](last_repr[i:i+1])
                predictions.append(pred)
            
            conf = torch.sigmoid(self.confidence_head(last_repr[i:i+1]))
            confidences.append(conf)
        
        prediction = torch.cat(predictions, dim=0)
        confidence = torch.cat(confidences, dim=0)
        
        return prediction, confidence

class EnsembleSpeculationModel(nn.Module):
    """Ensemble model combining multiple prediction strategies"""
    
    def __init__(self, num_experts=128, context_length=32, hidden_dim=512):
        super().__init__()
        self.num_experts = num_experts
        
        # Initialize predictors
        self.branch_predictor = BranchPredictorModule(
            num_experts=num_experts,
            context_length=min(context_length, 8),
            hidden_dim=hidden_dim // 2
        )
        
        self.pattern_predictor = PatternBasedPredictor(
            num_experts=num_experts,
            context_length=min(context_length, 16),
            hidden_dim=hidden_dim // 2
        )
        
        self.transformer_predictor = TransformerSpeculationModel(
            num_experts=num_experts,
            context_length=context_length,
            hidden_dim=hidden_dim
        )
        
        # Entropy-aware weighting system
        self.entropy_weights = nn.Parameter(torch.ones(6))  # For 6 layers
        self.predictor_weights = nn.Parameter(torch.ones(3))  # For 3 predictors
        
        # Meta-learning for ensemble weights
        self.meta_network = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # entropy + layer + position + confidence
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # Output weights for 3 predictors
            nn.Softmax(dim=-1)
        )
        
        # Layer entropy values (from analysis)
        self.layer_entropy = {
            1: 6.77, 3: 6.19, 5: 6.32, 7: 6.06, 9: 5.99, 11: 5.40
        }
        
    def get_layer_entropy(self, layer_id):
        """Get entropy for given layer"""
        entropy_values = []
        for lid in layer_id:
            entropy_values.append(self.layer_entropy.get(lid.item(), 6.0))
        return torch.tensor(entropy_values, device=layer_id.device)
        
    def forward(self, expert_sequence, layer_id, position=None):
        """
        Forward pass through ensemble
        
        Args:
            expert_sequence: [batch_size, seq_len] expert sequence
            layer_id: [batch_size] layer ID
            position: [batch_size] position in sequence
        """
        batch_size = expert_sequence.size(0)
        
        if position is None:
            position = torch.full((batch_size,), expert_sequence.size(1), device=expert_sequence.device)
        
        # Get predictions from all predictors
        local_pred, global_pred, branch_conf = self.branch_predictor(
            expert_sequence[:, -8:], position
        )
        
        pattern_pred, attention_weights = self.pattern_predictor(
            expert_sequence[:, -16:], layer_id
        )
        
        transformer_pred, transformer_conf = self.transformer_predictor(
            expert_sequence, layer_id, position
        )
        
        # Get layer entropy for weighting
        layer_entropy = self.get_layer_entropy(layer_id)
        
        # Prepare meta-network input
        meta_input = torch.stack([
            layer_entropy,
            layer_id.float(),
            position.float() / 512.0,  # Normalized position
            (branch_conf.squeeze() + transformer_conf.squeeze()) / 2  # Average confidence
        ], dim=-1)
        
        # Get dynamic weights
        dynamic_weights = self.meta_network(meta_input)
        
        # Combine predictions
        branch_combined = (local_pred + global_pred) / 2
        
        # Stack predictions for ensemble
        all_predictions = torch.stack([
            branch_combined,
            pattern_pred,
            transformer_pred
        ], dim=1)  # [batch_size, 3, num_experts]
        
        # Apply dynamic weights
        weighted_predictions = all_predictions * dynamic_weights.unsqueeze(-1)
        ensemble_prediction = torch.sum(weighted_predictions, dim=1)
        
        # Calculate ensemble confidence
        ensemble_confidence = (branch_conf + transformer_conf) / 2
        
        return {
            'prediction': ensemble_prediction,
            'confidence': ensemble_confidence,
            'branch_pred': branch_combined,
            'pattern_pred': pattern_pred,
            'transformer_pred': transformer_pred,
            'weights': dynamic_weights,
            'attention_weights': attention_weights
        }
        
    def get_predictor_contributions(self, expert_sequence, layer_id, position=None):
        """Get individual predictor contributions for analysis"""
        with torch.no_grad():
            output = self.forward(expert_sequence, layer_id, position)
            
            return {
                'branch_contribution': output['weights'][:, 0].mean().item(),
                'pattern_contribution': output['weights'][:, 1].mean().item(),
                'transformer_contribution': output['weights'][:, 2].mean().item(),
                'layer_entropy': self.get_layer_entropy(layer_id).mean().item()
            }