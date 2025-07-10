"""
Learnable Gating Models
Neural networks that learn to predict MoE routing based on input features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
from dataclasses import dataclass
import logging

from .gating_data_collector import GatingDataPoint

logger = logging.getLogger(__name__)

@dataclass
class GatingModelConfig:
    """Configuration for gating prediction models"""
    
    # Model architecture
    hidden_size: int = 512
    num_experts: int = 8
    num_layers: int = 6
    num_heads: int = 8
    
    # Context features
    context_layers: int = 4        # How many previous layers to use
    max_seq_len: int = 512
    
    # Training parameters
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Gating-specific parameters
    temperature: float = 1.0       # For gating softmax
    top_k: int = 2                # Top-k experts to predict
    
    # Loss weights
    routing_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.1
    diversity_loss_weight: float = 0.05

class ContextualGatingPredictor(nn.Module):
    """
    Neural network that predicts expert routing based on:
    1. Current layer hidden states
    2. Previous layer routing history
    3. Attention patterns
    4. Token-level features
    """
    
    def __init__(self, config: GatingModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection layers
        self.hidden_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.position_encoding = nn.Parameter(torch.randn(config.max_seq_len, config.hidden_size))
        
        # Context encoding from previous layers
        self.context_encoder = nn.ModuleList([
            nn.Linear(config.num_experts, config.hidden_size // 4)
            for _ in range(config.context_layers)
        ])
        
        # Multi-head attention for analyzing routing patterns
        self.routing_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_size + config.hidden_size // 4 * config.context_layers, 
                     config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Gating prediction head
        self.gating_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_experts)
        )
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,           # [batch_size, seq_len, hidden_size]
        prev_layer_gates: List[torch.Tensor],  # List of [batch_size, seq_len, num_experts]
        layer_id: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            gating_logits: [batch_size, seq_len, num_experts]
            confidence_scores: [batch_size, seq_len, 1]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. Project hidden states
        hidden_features = self.hidden_proj(hidden_states)
        
        # 2. Add positional encoding
        pos_encoding = self.position_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        hidden_features = hidden_features + pos_encoding
        
        # 3. Encode context from previous layers
        context_features = []
        for i, prev_gates in enumerate(prev_layer_gates[-self.config.context_layers:]):
            if i < len(self.context_encoder):
                # prev_gates: [batch_size, seq_len, num_experts]
                ctx_feat = self.context_encoder[i](prev_gates)
                context_features.append(ctx_feat)
            else:
                # If we have more context layers than encoders, pad with zeros
                zero_feat = torch.zeros(batch_size, seq_len, self.config.hidden_size // 4, device=hidden_states.device)
                context_features.append(zero_feat)
        
        if context_features:
            context_features = torch.cat(context_features, dim=-1)
        else:
            # No context available (early layers)
            context_features = torch.zeros(
                batch_size, seq_len, 
                self.config.hidden_size // 4 * self.config.context_layers,
                device=hidden_states.device
            )
        
        # 4. Apply multi-head attention for routing pattern analysis
        # Convert attention mask to proper format if provided
        attn_mask = None
        if attention_mask is not None:
            # attention_mask is [batch_size, seq_len], need key_padding_mask format
            attn_mask = attention_mask
        
        attended_features, attention_weights = self.routing_attention(
            hidden_features, hidden_features, hidden_features,
            key_padding_mask=attn_mask
        )
        
        # 5. Fuse features
        combined_features = torch.cat([attended_features, context_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        fused_features = self.layer_norm(fused_features + hidden_features)
        
        # 6. Predict gating logits
        gating_logits = self.gating_head(fused_features)
        
        # 7. Predict confidence scores
        confidence_scores = self.confidence_head(fused_features)
        
        return gating_logits, confidence_scores, attention_weights

class TransformerGatingPredictor(nn.Module):
    """
    Transformer-based gating predictor that uses sequence-to-sequence modeling
    """
    
    def __init__(self, config: GatingModelConfig):
        super().__init__()
        self.config = config
        
        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3  # Smaller than main model
        )
        
        # Input embeddings
        self.input_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_embedding = nn.Embedding(config.num_layers, config.hidden_size)
        
        # Context history encoder
        self.history_encoder = nn.LSTM(
            input_size=config.num_experts * config.context_layers,
            hidden_size=config.hidden_size // 4,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Output heads
        self.gating_head = nn.Linear(config.hidden_size, config.num_experts)
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_layer_gates: List[torch.Tensor],
        layer_id: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. Project input
        input_features = self.input_projection(hidden_states)
        
        # 2. Add layer embedding
        layer_emb = self.layer_embedding(
            torch.full((batch_size, seq_len), layer_id, device=hidden_states.device)
        )
        input_features = input_features + layer_emb
        
        # 3. Encode routing history
        if prev_layer_gates:
            # Stack previous gates: [batch_size, seq_len, num_experts * context_layers]
            history_input = torch.cat(prev_layer_gates[-self.config.context_layers:], dim=-1)
            history_features, _ = self.history_encoder(history_input)
            
            # Combine with input features
            input_features = input_features + F.adaptive_avg_pool1d(
                history_features.transpose(1, 2), hidden_size
            ).transpose(1, 2)
        
        # 4. Apply transformer
        transformer_output = self.transformer_encoder(input_features, src_key_padding_mask=attention_mask)
        
        # 5. Predict outputs
        gating_logits = self.gating_head(transformer_output)
        confidence_scores = torch.sigmoid(self.confidence_head(transformer_output))
        
        return gating_logits, confidence_scores, None

class HierarchicalGatingPredictor(nn.Module):
    """
    Hierarchical model that predicts routing at multiple granularities
    """
    
    def __init__(self, config: GatingModelConfig):
        super().__init__()
        self.config = config
        
        # Token-level predictor
        self.token_predictor = ContextualGatingPredictor(config)
        
        # Sequence-level predictor
        self.sequence_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.num_experts)
        )
        
        # Combining weights
        self.combination_weights = nn.Parameter(torch.tensor([0.7, 0.3]))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_layer_gates: List[torch.Tensor],
        layer_id: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Token-level prediction
        token_logits, token_confidence, attention_weights = self.token_predictor(
            hidden_states, prev_layer_gates, layer_id, attention_mask
        )
        
        # Sequence-level prediction
        seq_repr = torch.mean(hidden_states, dim=1)  # Average pooling
        seq_logits = self.sequence_predictor(seq_repr)
        seq_logits = seq_logits.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        
        # Combine predictions
        combined_logits = (self.combination_weights[0] * token_logits + 
                          self.combination_weights[1] * seq_logits)
        
        return combined_logits, token_confidence, attention_weights

class GatingDataset(Dataset):
    """Dataset for training gating prediction models"""
    
    def __init__(self, data_points: List[GatingDataPoint], max_seq_len: int = 512):
        self.data_points = data_points
        self.max_seq_len = max_seq_len
        
        # Filter and preprocess data
        self.valid_data = []
        for dp in data_points:
            if dp.sequence_length > 0 and dp.target_routing is not None:
                self.valid_data.append(dp)
        
        logger.info(f"Dataset created with {len(self.valid_data)} valid samples")
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        dp = self.valid_data[idx]
        
        # Truncate or pad sequences
        seq_len = min(dp.sequence_length, self.max_seq_len)
        
        # Hidden states
        hidden_states = dp.hidden_states[:seq_len]
        if hidden_states.size(0) < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - hidden_states.size(0), hidden_states.size(1))
            hidden_states = torch.cat([hidden_states, padding], dim=0)
        
        # Target routing
        target_routing = dp.target_routing[:seq_len]
        # Ensure target_routing has correct dimensions
        if target_routing.ndim == 3:
            target_routing = target_routing.squeeze(0)  # Remove batch dimension if present
        elif target_routing.ndim == 1:
            target_routing = target_routing.unsqueeze(1)  # Add feature dimension if missing
        
        if target_routing.size(0) < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - target_routing.size(0), target_routing.size(-1))
            target_routing = torch.cat([target_routing, padding], dim=0)
        
        # Previous layer gates
        prev_gates = []
        for prev_gate in dp.prev_layer_gates or []:
            gate = prev_gate[:seq_len]
            # Ensure gate has correct dimensions
            if gate.ndim == 3:
                gate = gate.squeeze(0)  # Remove batch dimension if present
            elif gate.ndim == 1:
                gate = gate.unsqueeze(1)  # Add feature dimension if missing
            
            if gate.size(0) < self.max_seq_len:
                padding = torch.zeros(self.max_seq_len - gate.size(0), gate.size(-1))
                gate = torch.cat([gate, padding], dim=0)
            prev_gates.append(gate)
        
        # Attention mask
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[seq_len:] = True  # Mask padded positions
        
        return {
            'hidden_states': hidden_states,
            'prev_layer_gates': prev_gates,
            'target_routing': target_routing,
            'target_top_k': dp.target_top_k[:seq_len] if dp.target_top_k is not None else None,
            'layer_id': dp.layer_id,
            'attention_mask': attention_mask,
            'sequence_length': seq_len
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    
    # Stack tensors
    hidden_states = torch.stack([item['hidden_states'] for item in batch])
    target_routing = torch.stack([item['target_routing'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Handle variable-length previous gates - always ensure we have exactly 4 context layers
    gate_lengths = [len(item['prev_layer_gates']) for item in batch]
    context_layers = 4  # Fixed number of context layers
    prev_layer_gates = []
    
    for i in range(context_layers):
        gates_for_layer = []
        for item in batch:
            if i < len(item['prev_layer_gates']) and len(item['prev_layer_gates']) > 0:
                gates_for_layer.append(item['prev_layer_gates'][i])
            else:
                # Pad with zeros if this batch item has fewer context layers
                # Use the target_routing shape as reference for zero padding
                reference_shape = item['target_routing'].shape
                gates_for_layer.append(torch.zeros(reference_shape))
        
        prev_layer_gates.append(torch.stack(gates_for_layer))
    
    return {
        'hidden_states': hidden_states,
        'prev_layer_gates': prev_layer_gates,
        'target_routing': target_routing,
        'layer_id': batch[0]['layer_id'],  # Assume same layer for batch
        'attention_mask': attention_mask,
        'sequence_length': [item['sequence_length'] for item in batch]
    }

def create_gating_model(model_type: str, config: GatingModelConfig) -> nn.Module:
    """Factory function to create gating prediction models"""
    
    if model_type == "contextual":
        return ContextualGatingPredictor(config)
    elif model_type == "transformer":
        return TransformerGatingPredictor(config)
    elif model_type == "hierarchical":
        return HierarchicalGatingPredictor(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test the models
    config = GatingModelConfig(
        hidden_size=512,
        num_experts=8,
        num_layers=6
    )
    
    # Create test data
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    prev_gates = [torch.randn(batch_size, seq_len, config.num_experts) for _ in range(3)]
    
    # Test all model types
    for model_type in ["contextual", "transformer", "hierarchical"]:
        print(f"\nTesting {model_type} model...")
        
        model = create_gating_model(model_type, config)
        
        with torch.no_grad():
            gating_logits, confidence, attention = model(hidden_states, prev_gates, layer_id=2)
        
        print(f"Gating logits shape: {gating_logits.shape}")
        print(f"Confidence shape: {confidence.shape}")
        print(f"Attention weights: {attention is not None}")
        
        # Check if predictions are reasonable
        gating_probs = F.softmax(gating_logits, dim=-1)
        print(f"Gating probabilities range: [{gating_probs.min():.4f}, {gating_probs.max():.4f}]")
        print(f"Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")
        
        print("✅ Model test passed!")
    
    print("\n🎉 All gating models created successfully!")