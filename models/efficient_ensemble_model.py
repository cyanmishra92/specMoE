#!/usr/bin/env python3
"""
Efficient Ensemble Expert Speculation Model
Simplified architecture with better regularization and faster training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class EfficientBranchPredictor(nn.Module):
    """Simplified branch predictor with fewer parameters"""
    
    def __init__(self, num_experts=128, context_length=8, hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.context_length = context_length
        
        # Simplified local predictor
        self.expert_embedding = nn.Embedding(num_experts, hidden_dim // 2)
        self.local_lstm = nn.LSTM(hidden_dim // 2, hidden_dim // 2, batch_first=True)
        self.local_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Simplified global predictor
        self.global_head = nn.Sequential(
            nn.Linear(context_length + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
    def forward(self, expert_sequence, position):
        batch_size = expert_sequence.size(0)
        
        # Local prediction
        embedded = self.expert_embedding(expert_sequence)
        lstm_out, _ = self.local_lstm(embedded)
        local_pred = self.local_head(lstm_out[:, -1, :])
        
        # Global prediction
        position_normalized = position.float().unsqueeze(1) / 512.0
        pattern_features = torch.zeros(batch_size, self.context_length, device=expert_sequence.device)
        for i in range(min(self.context_length, expert_sequence.size(1))):
            pattern_features[:, i] = expert_sequence[:, -(i+1)].float() / self.num_experts
        
        global_input = torch.cat([pattern_features, position_normalized], dim=1)
        global_pred = self.global_head(global_input)
        
        return local_pred, global_pred

class EfficientPatternPredictor(nn.Module):
    """Simplified pattern predictor with single attention head"""
    
    def __init__(self, num_experts=128, context_length=16, hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts
        
        # Simplified embeddings
        self.expert_embedding = nn.Embedding(num_experts, hidden_dim)
        self.layer_embedding = nn.Embedding(12, hidden_dim)
        
        # Single attention head (much faster)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,  # Reduced from 8
            batch_first=True,
            dropout=0.2
        )
        
        # Simplified prediction head
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
    def forward(self, expert_sequence, layer_id):
        batch_size = expert_sequence.size(0)
        
        # Embed sequences
        expert_emb = self.expert_embedding(expert_sequence)
        layer_emb = self.layer_embedding(layer_id).unsqueeze(1).expand(-1, expert_sequence.size(1), -1)
        
        # Combine embeddings
        combined_emb = expert_emb + layer_emb
        
        # Apply attention
        attended, _ = self.attention(combined_emb, combined_emb, combined_emb)
        
        # Predict from last token
        prediction = self.prediction_head(attended[:, -1, :])
        
        return prediction

class EfficientTransformerPredictor(nn.Module):
    """Simplified transformer with shared heads"""
    
    def __init__(self, num_experts=128, context_length=32, hidden_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.context_length = context_length
        
        # Embeddings
        self.expert_embedding = nn.Embedding(num_experts, hidden_dim)
        self.layer_embedding = nn.Embedding(12, hidden_dim)
        self.position_embedding = nn.Embedding(context_length, hidden_dim)
        
        # Simplified transformer (fewer layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,  # Reduced from 8
            dim_feedforward=hidden_dim,  # Reduced from 2x
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)  # Reduced from 6
        
        # Shared prediction head with layer conditioning
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
    def forward(self, expert_sequence, layer_id, position=None):
        batch_size, seq_len = expert_sequence.shape
        
        # Embeddings
        expert_emb = self.expert_embedding(expert_sequence)
        layer_emb = self.layer_embedding(layer_id).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Position embedding
        pos_indices = torch.arange(seq_len, device=expert_sequence.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_indices % self.context_length)
        
        # Combine embeddings
        combined_emb = expert_emb + layer_emb + pos_emb
        
        # Apply transformer
        transformer_out = self.transformer(combined_emb)
        
        # Predict from last token
        prediction = self.prediction_head(transformer_out[:, -1, :])
        
        return prediction

class EfficientEnsembleModel(nn.Module):
    """Efficient ensemble model with better regularization"""
    
    def __init__(self, num_experts=128, context_length=32, hidden_dim=256):
        super().__init__()
        self.num_experts = num_experts
        
        # Simplified predictors
        self.branch_predictor = EfficientBranchPredictor(
            num_experts=num_experts,
            context_length=min(context_length, 8),
            hidden_dim=hidden_dim // 2
        )
        
        self.pattern_predictor = EfficientPatternPredictor(
            num_experts=num_experts,
            context_length=min(context_length, 16),
            hidden_dim=hidden_dim // 2
        )
        
        self.transformer_predictor = EfficientTransformerPredictor(
            num_experts=num_experts,
            context_length=context_length,
            hidden_dim=hidden_dim
        )
        
        # Simplified ensemble weighting
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)  # Equal initial weights
        
        # Layer-specific scaling (learned)
        self.layer_scaling = nn.Parameter(torch.ones(6))  # For 6 layers
        
        # Layer entropy mapping
        self.layer_to_index = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
        
    def forward(self, expert_sequence, layer_id, position=None):
        batch_size = expert_sequence.size(0)
        
        if position is None:
            position = torch.full((batch_size,), expert_sequence.size(1), device=expert_sequence.device)
        
        # Get predictions from all predictors
        local_pred, global_pred = self.branch_predictor(
            expert_sequence[:, -8:], position
        )
        branch_pred = (local_pred + global_pred) / 2  # Simple average
        
        pattern_pred = self.pattern_predictor(
            expert_sequence[:, -16:], layer_id
        )
        
        transformer_pred = self.transformer_predictor(
            expert_sequence, layer_id, position
        )
        
        # Stack predictions
        all_predictions = torch.stack([branch_pred, pattern_pred, transformer_pred], dim=1)
        
        # Apply ensemble weights
        ensemble_weights = F.softmax(self.ensemble_weights, dim=0)
        weighted_predictions = all_predictions * ensemble_weights.view(1, 3, 1)
        ensemble_prediction = torch.sum(weighted_predictions, dim=1)
        
        # Apply layer-specific scaling
        layer_scales = []
        for lid in layer_id:
            layer_idx = self.layer_to_index.get(lid.item(), 0)
            layer_scales.append(self.layer_scaling[layer_idx])
        
        layer_scale = torch.stack(layer_scales).unsqueeze(1)
        final_prediction = ensemble_prediction * layer_scale
        
        return {
            'prediction': final_prediction,
            'branch_pred': branch_pred,
            'pattern_pred': pattern_pred,
            'transformer_pred': transformer_pred,
            'ensemble_weights': ensemble_weights
        }

class EfficientEnsembleDataset(torch.utils.data.Dataset):
    """Efficient dataset with better memory usage"""
    
    def __init__(self, traces, context_length=32, max_samples_per_layer=15000):
        self.traces = []
        self.context_length = context_length
        
        print(f"Creating efficient dataset with max {max_samples_per_layer} samples per layer...")
        
        # Group traces by layer and limit samples per layer
        from collections import defaultdict
        layer_samples = defaultdict(list)
        
        for trace in traces:
            layer_id = trace['layer_id']
            target_routing = trace['target_routing']
            expert_sequence = np.argmax(target_routing, axis=1)
            
            # Create training sequences
            for i in range(len(expert_sequence) - 1):
                if i >= context_length:
                    if len(layer_samples[layer_id]) < max_samples_per_layer:
                        layer_samples[layer_id].append({
                            'context': expert_sequence[i-context_length:i],
                            'target': expert_sequence[i],
                            'layer_id': layer_id,
                            'position': i
                        })
        
        # Flatten and shuffle
        import random
        for layer_id, samples in layer_samples.items():
            random.shuffle(samples)
            self.traces.extend(samples)
        
        random.shuffle(self.traces)
        
        print(f"Created {len(self.traces)} efficient training sequences")
        layer_counts = {}
        for trace in self.traces:
            layer_id = trace['layer_id']
            layer_counts[layer_id] = layer_counts.get(layer_id, 0) + 1
        print(f"Layer distribution: {layer_counts}")
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        
        return {
            'context': torch.tensor(trace['context'], dtype=torch.long),
            'target': torch.tensor(trace['target'], dtype=torch.long),
            'layer_id': torch.tensor(trace['layer_id'], dtype=torch.long),
            'position': torch.tensor(trace['position'], dtype=torch.long)
        }