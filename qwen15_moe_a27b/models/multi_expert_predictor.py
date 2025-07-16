#!/usr/bin/env python3
"""
Multi-Expert Predictor for Qwen1.5-MoE-A2.7B
Predicts the top-2 expert pairs for each token based on previous layer activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math

class MultiExpertPredictor(nn.Module):
    """
    Multi-Expert Predictor for Qwen1.5-MoE-A2.7B
    
    Predicts top-2 expert pairs (120 possible combinations from 60 experts)
    Uses inter-layer attention mechanism similar to Switch-Base approach
    """
    
    def __init__(
        self,
        num_experts: int = 60,  # 60 routing experts (+ 4 shared always activated)
        num_layers: int = 24,
        hidden_size: int = 2048,
        intermediate_size: int = 1024,
        attention_heads: int = 8,
        dropout: float = 0.1,
        prediction_window: int = 3,  # Look at last 3 layers
        top_k: int = 4  # Top-4 routing (num_experts_per_tok)
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.prediction_window = prediction_window
        self.top_k = top_k
        
        # For top-4 routing, we predict individual experts rather than pairs
        # Each token activates 4 out of 60 routing experts
        self.num_expert_combinations = num_experts  # Predict each expert independently
        
        # Input projection for hidden states
        self.input_projection = nn.Linear(hidden_size, intermediate_size)
        
        # Inter-layer attention mechanism
        self.inter_layer_attention = nn.MultiheadAttention(
            embed_dim=intermediate_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer-specific embeddings
        self.layer_embeddings = nn.Embedding(num_layers, intermediate_size)
        
        # Multi-expert prediction head (predicts top-4 experts)
        self.multi_expert_predictor = nn.Sequential(
            nn.Linear(intermediate_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, intermediate_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size // 2, num_experts)
        )
        
        # Expert ranking prediction (auxiliary task for better training)
        self.expert_ranking_predictor = nn.Sequential(
            nn.Linear(intermediate_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, num_experts)
        )
        
        # Confidence estimation
        self.confidence_predictor = nn.Sequential(
            nn.Linear(intermediate_size, intermediate_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
    
    def predict_top_k_experts(self, logits: torch.Tensor, k: int = 4) -> torch.Tensor:
        """Get top-k expert predictions from logits"""
        return torch.topk(logits, k=k, dim=-1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            hidden_states: [batch_size, seq_len, window_size, hidden_size]
            layer_ids: [batch_size, window_size] - Layer IDs for each window position
            attention_mask: [batch_size, seq_len] - Attention mask
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        batch_size, seq_len, window_size, hidden_size = hidden_states.shape
        
        # Reshape for processing
        hidden_states = hidden_states.view(batch_size * seq_len, window_size, hidden_size)
        layer_ids = layer_ids.view(batch_size * seq_len, window_size)
        
        # Project input
        projected_states = self.input_projection(hidden_states)  # [B*S, W, intermediate_size]
        
        # Add layer embeddings
        layer_embs = self.layer_embeddings(layer_ids)  # [B*S, W, intermediate_size]
        projected_states = projected_states + layer_embs
        
        # Apply inter-layer attention
        attended_states, attention_weights = self.inter_layer_attention(
            projected_states, projected_states, projected_states
        )  # [B*S, W, intermediate_size]
        
        # Pool across window (weighted by attention to most recent layer)
        pooled_states = attended_states.mean(dim=1)  # [B*S, intermediate_size]
        
        # Predict top-k experts
        expert_logits = self.multi_expert_predictor(pooled_states)  # [B*S, num_experts]
        
        # Predict expert ranking (auxiliary task)
        ranking_logits = self.expert_ranking_predictor(pooled_states)  # [B*S, num_experts]
        
        # Predict confidence
        confidence = self.confidence_predictor(pooled_states)  # [B*S, 1]
        
        # Reshape back
        expert_logits = expert_logits.view(batch_size, seq_len, self.num_experts)
        ranking_logits = ranking_logits.view(batch_size, seq_len, self.num_experts)
        confidence = confidence.view(batch_size, seq_len, 1)
        
        # Get top-k expert predictions
        top_k_experts = torch.topk(expert_logits, k=self.top_k, dim=-1)
        
        return {
            'expert_logits': expert_logits,
            'ranking_logits': ranking_logits,
            'confidence': confidence,
            'top_k_expert_indices': top_k_experts.indices,
            'top_k_expert_scores': top_k_experts.values,
            'attention_weights': attention_weights.view(batch_size, seq_len, window_size, window_size)
        }
    
    def predict_top_k_experts_for_sequence(
        self,
        hidden_states: torch.Tensor,
        layer_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Predict top-k experts for a single sequence
        
        Returns:
            List of top-k expert lists for each token
        """
        outputs = self.forward(hidden_states, layer_ids, attention_mask)
        
        # Get top-k predictions for each token
        top_indices = outputs['top_k_expert_indices']  # [batch_size, seq_len, k]
        
        predictions = []
        for batch_idx in range(top_indices.shape[0]):
            batch_predictions = []
            for seq_idx in range(top_indices.shape[1]):
                expert_ids = top_indices[batch_idx, seq_idx].tolist()
                batch_predictions.append(expert_ids)
            predictions.append(batch_predictions)
        
        return predictions

class MultiExpertLoss(nn.Module):
    """
    Multi-task loss for top-4 expert prediction
    """
    
    def __init__(
        self,
        num_experts: int = 60,
        top_k: int = 4,
        expert_weight: float = 1.0,
        ranking_weight: float = 0.3,
        confidence_weight: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_weight = expert_weight
        self.ranking_weight = ranking_weight
        self.confidence_weight = confidence_weight
        
        # Use binary cross entropy for multi-label classification
        self.expert_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ranking_loss = nn.CrossEntropyLoss(reduction='none')
        self.confidence_loss = nn.BCELoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        target_experts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions
            target_experts: [batch_size, seq_len, top_k] - Target expert IDs
            attention_mask: [batch_size, seq_len] - Attention mask
        """
        batch_size, seq_len = target_experts.shape[:2]
        
        # Convert target experts to multi-hot encoding
        target_multi_hot = torch.zeros(batch_size, seq_len, self.num_experts, device=target_experts.device)
        for i in range(self.top_k):
            expert_indices = target_experts[:, :, i]
            valid_mask = (expert_indices >= 0) & (expert_indices < self.num_experts)
            target_multi_hot[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len).unsqueeze(0), expert_indices] = valid_mask.float()
        
        # Multi-label expert prediction loss
        expert_logits = predictions['expert_logits']
        expert_loss = self.expert_loss(expert_logits, target_multi_hot)
        
        # Ranking loss (predict which expert is most important)
        ranking_logits = predictions['ranking_logits']
        # Use first expert as the most important
        ranking_targets = target_experts[:, :, 0]
        valid_targets = (ranking_targets >= 0) & (ranking_targets < self.num_experts)
        ranking_loss = self.ranking_loss(
            ranking_logits.view(-1, ranking_logits.size(-1)),
            ranking_targets.view(-1)
        )
        
        # Confidence loss (predict accuracy)
        confidence = predictions['confidence'].squeeze(-1)
        # Calculate accuracy based on top-k predictions
        predicted_experts = torch.topk(expert_logits, k=self.top_k, dim=-1).indices
        
        # Check if predicted experts match targets
        correct_predictions = 0
        for i in range(self.top_k):
            for j in range(self.top_k):
                correct_predictions += (predicted_experts[:, :, i] == target_experts[:, :, j]).float()
        
        accuracy = correct_predictions / self.top_k
        confidence_loss = self.confidence_loss(confidence, accuracy)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            expert_loss = expert_loss * mask
            ranking_loss = ranking_loss.view(batch_size, seq_len) * attention_mask
            confidence_loss = confidence_loss * attention_mask
        
        # Combine losses
        total_loss = (
            self.expert_weight * expert_loss.mean() +
            self.ranking_weight * ranking_loss.mean() +
            self.confidence_weight * confidence_loss.mean()
        )
        
        return {
            'total_loss': total_loss,
            'expert_loss': expert_loss.mean(),
            'ranking_loss': ranking_loss.mean(),
            'confidence_loss': confidence_loss.mean()
        }

# Helper functions for data processing
def prepare_training_data(traces: List, prediction_window: int = 3):
    """
    Prepare training data from traces
    
    Args:
        traces: List of trace objects
        prediction_window: Number of previous layers to use for prediction
    
    Returns:
        Tuple of (inputs, targets) for training
    """
    inputs = []
    targets = []
    
    # Group traces by layer
    layer_traces = {}
    for trace in traces:
        layer_id = trace.layer_id
        if layer_id not in layer_traces:
            layer_traces[layer_id] = []
        layer_traces[layer_id].append(trace)
    
    # Create sequences for each layer (except the first few)
    for layer_id in sorted(layer_traces.keys()):
        if layer_id < prediction_window:
            continue
            
        current_traces = layer_traces[layer_id]
        
        for trace in current_traces:
            # Get previous layers' hidden states
            prev_layers = []
            prev_layer_ids = []
            
            for prev_layer_id in range(layer_id - prediction_window, layer_id):
                if prev_layer_id in layer_traces:
                    # Find corresponding trace in previous layer
                    prev_trace = None
                    for pt in layer_traces[prev_layer_id]:
                        if pt.sample_id == trace.sample_id:
                            prev_trace = pt
                            break
                    
                    if prev_trace is not None:
                        prev_layers.append(prev_trace.hidden_states)
                        prev_layer_ids.append(prev_layer_id)
            
            if len(prev_layers) == prediction_window:
                # Stack hidden states: [seq_len, window_size, hidden_size]
                input_states = torch.stack(prev_layers, dim=1)
                input_layer_ids = torch.tensor(prev_layer_ids)
                
                # Target: expert pair indices
                target_experts = trace.target_top_k  # [seq_len, 2]
                
                inputs.append((input_states, input_layer_ids))
                targets.append(target_experts)
    
    return inputs, targets

def pad_expert_sequence(expert_sequence: List[int], target_length: int = 4, pad_value: int = -1) -> List[int]:
    """Pad or truncate expert sequence to target length"""
    if len(expert_sequence) >= target_length:
        return expert_sequence[:target_length]
    else:
        return expert_sequence + [pad_value] * (target_length - len(expert_sequence))