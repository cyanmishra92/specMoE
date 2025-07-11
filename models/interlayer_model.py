#!/usr/bin/env python3
"""
Inter-Layer MoE Expert Prediction Model
Learns spatio-temporal patterns across transformer layers to predict future expert selections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InterLayerSpeculationModel(nn.Module):
    """
    Advanced model for predicting expert selections across transformer layers.
    
    Takes sequence of expert selections from layers 1→n and predicts layer n+1.
    Learns spatio-temporal patterns in expert routing behavior.
    """
    
    def __init__(self, num_experts=128, hidden_size=512, num_layers=12, 
                 model_dim=256, num_heads=8, ff_dim=1024, dropout=0.1,
                 num_attention_layers=4, context_length=3, prediction_horizon=2):
        super().__init__()
        
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_attention_layers = num_attention_layers
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
        # Expert embedding: convert expert IDs to dense representations
        self.expert_embedding = nn.Embedding(num_experts, model_dim)
        
        # Layer positional encoding: encode which layer we're at
        self.layer_pos_encoding = nn.Embedding(num_layers, model_dim)
        
        # Token positional encoding: encode position within sequence
        # Need enough positions for max_seq_len * context_length
        max_positions = 1024  # Increased buffer size
        self.register_buffer('token_pos_encoding', 
                           self._create_sinusoidal_encoding(max_positions, model_dim))
        
        # Multi-head attention layers for spatio-temporal pattern learning
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_attention_layers)
        ])
        
        # Cross-layer attention: how do different layers influence each other
        self.cross_layer_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Expert pattern memory: learn common expert selection patterns
        self.pattern_memory = nn.Parameter(torch.randn(64, model_dim))  # 64 patterns
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final prediction layers
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-scale prediction heads
        self.expert_predictor = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, num_experts)
        )
        
        # Confidence predictor: how confident are we in this prediction
        self.confidence_predictor = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _create_sinusoidal_encoding(self, max_len, d_model):
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, expert_sequence, layer_ids, attention_mask=None):
        """
        Forward pass for inter-layer expert prediction.
        
        Args:
            expert_sequence: [batch_size, seq_len, num_layers] - expert IDs for each layer
            layer_ids: [batch_size, seq_len] - which layer we're predicting for
            attention_mask: [batch_size, seq_len] - mask for valid positions
            
        Returns:
            expert_logits: [batch_size, seq_len, num_experts] - prediction logits
            confidence: [batch_size, seq_len, 1] - prediction confidence
        """
        batch_size, seq_len, num_input_layers = expert_sequence.shape
        
        # Embed expert selections: [batch_size, seq_len, num_layers, model_dim]
        expert_embeds = self.expert_embedding(expert_sequence)
        
        # Add layer positional encoding
        layer_pos = self.layer_pos_encoding(torch.arange(num_input_layers, device=expert_sequence.device))
        expert_embeds = expert_embeds + layer_pos.unsqueeze(0).unsqueeze(0)
        
        # Flatten to process all layer-token combinations
        # [batch_size, seq_len * num_layers, model_dim]
        flat_embeds = expert_embeds.view(batch_size, seq_len * num_input_layers, self.model_dim)
        
        # Add token positional encoding (adjust for current model_dim)
        token_pos = self.token_pos_encoding[:seq_len * num_input_layers, :self.model_dim]
        flat_embeds = flat_embeds + token_pos.unsqueeze(0)
        
        # Create attention mask for flattened sequence
        if attention_mask is not None:
            flat_mask = attention_mask.unsqueeze(-1).expand(-1, -1, num_input_layers)
            flat_mask = flat_mask.reshape(batch_size, seq_len * num_input_layers)
        else:
            flat_mask = None
        
        # Apply attention layers for spatio-temporal learning
        hidden = flat_embeds
        for attention_layer in self.attention_layers:
            if flat_mask is not None:
                # Convert mask to attention mask format
                attn_mask = ~flat_mask.bool()
            else:
                attn_mask = None
            hidden = attention_layer(hidden, src_key_padding_mask=attn_mask)
        
        # Reshape back to sequence format and aggregate across layers
        # [batch_size, seq_len, num_layers, model_dim]
        layer_hidden = hidden.view(batch_size, seq_len, num_input_layers, self.model_dim)
        
        # Cross-layer attention: how do different layers influence the prediction
        layer_context = layer_hidden.mean(dim=2)  # [batch_size, seq_len, model_dim]
        layer_keys = layer_hidden.view(batch_size * seq_len, num_input_layers, self.model_dim)
        layer_queries = layer_context.view(batch_size * seq_len, 1, self.model_dim)
        
        cross_attended, _ = self.cross_layer_attention(
            layer_queries, layer_keys, layer_keys
        )
        cross_attended = cross_attended.view(batch_size, seq_len, self.model_dim)
        
        # Pattern memory attention: match against learned patterns
        pattern_queries = cross_attended.view(batch_size * seq_len, 1, self.model_dim)
        pattern_keys = self.pattern_memory.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        
        pattern_attended, pattern_weights = self.pattern_attention(
            pattern_queries, pattern_keys, pattern_keys
        )
        pattern_attended = pattern_attended.view(batch_size, seq_len, self.model_dim)
        
        # Combine cross-layer and pattern information
        final_hidden = self.layer_norm(cross_attended + pattern_attended)
        final_hidden = self.dropout(final_hidden)
        
        # Generate predictions
        expert_logits = self.expert_predictor(final_hidden)
        confidence = self.confidence_predictor(final_hidden)
        
        return expert_logits, confidence, pattern_weights.view(batch_size, seq_len, -1)

class InterlayerLoss(nn.Module):
    """
    Advanced loss function for inter-layer expert prediction.
    Combines cross-entropy with confidence weighting and pattern regularization.
    """
    
    def __init__(self, num_experts=128, confidence_weight=0.1, pattern_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.confidence_weight = confidence_weight
        self.pattern_weight = pattern_weight
        
        # Focal loss for handling class imbalance
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
    
    def forward(self, expert_logits, confidence, pattern_weights, targets, mask=None):
        """
        Compute advanced loss for inter-layer prediction.
        
        Args:
            expert_logits: [batch_size, seq_len, num_experts]
            confidence: [batch_size, seq_len, 1]
            pattern_weights: [batch_size, seq_len, num_patterns]
            targets: [batch_size, seq_len] - target expert IDs
            mask: [batch_size, seq_len] - valid positions
        """
        if mask is not None:
            # Only compute loss on valid positions
            valid_positions = mask.bool()
            expert_logits = expert_logits[valid_positions]
            confidence = confidence[valid_positions]
            pattern_weights = pattern_weights[valid_positions]
            targets = targets[valid_positions]
        
        # Focal cross-entropy loss
        ce_loss = F.cross_entropy(expert_logits, targets, reduction='none')
        
        # Focal loss weighting
        probs = F.softmax(expert_logits, dim=-1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.focal_alpha * (1 - target_probs) ** self.focal_gamma
        focal_loss = (focal_weight * ce_loss).mean()
        
        # Confidence loss: encourage high confidence for correct predictions
        correct_predictions = (expert_logits.argmax(dim=-1) == targets).float()
        confidence_loss = F.mse_loss(confidence.squeeze(-1), correct_predictions)
        
        # Pattern diversity loss: encourage diverse pattern usage
        pattern_entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-8), dim=-1)
        pattern_loss = -pattern_entropy.mean()  # Maximize entropy
        
        # Total loss
        total_loss = (focal_loss + 
                     self.confidence_weight * confidence_loss + 
                     self.pattern_weight * pattern_loss)
        
        return {
            'total_loss': total_loss,
            'focal_loss': focal_loss,
            'confidence_loss': confidence_loss,
            'pattern_loss': pattern_loss
        }