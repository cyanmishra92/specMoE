"""
Simplified Qwen Multi-Expert Predictor with Stable Training
Focus on getting basic expert prediction working first
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleQwenPredictor(nn.Module):
    """Simplified predictor focusing on stable training"""
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256, 
                 num_experts: int = 60, experts_per_token: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        
        # Simple but effective architecture
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output head for expert logits
        self.expert_head = nn.Linear(hidden_dim, num_experts)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass
        
        Args:
            hidden_states: [batch, seq_len, input_dim]
            attention_mask: [batch, seq_len]
            
        Returns:
            predictions dictionary
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project input
        x = self.input_proj(hidden_states)  # [batch, seq, hidden_dim]
        
        # Pass through hidden layers
        x = self.hidden_layers(x)  # [batch, seq, hidden_dim]
        
        # Get expert logits
        expert_logits = self.expert_head(x)  # [batch, seq, num_experts]
        
        # Get top-k predictions
        top_k_logits, top_k_indices = torch.topk(expert_logits, 
                                                k=self.experts_per_token, 
                                                dim=-1)
        
        # Apply softmax to get probabilities
        expert_probs = F.softmax(expert_logits, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        return {
            'expert_logits': expert_logits,
            'expert_probs': expert_probs,
            'top_k_indices': top_k_indices,
            'top_k_logits': top_k_logits,
            'top_k_probs': top_k_probs
        }
    
    def compute_loss(self, predictions, targets):
        """
        Simplified and stable loss function
        
        Args:
            predictions: Model predictions
            targets: Ground truth expert indices and routing weights
        """
        expert_logits = predictions['expert_logits']
        target_experts = targets['expert_indices']  # [batch, seq, k]
        
        batch_size, seq_len, _ = expert_logits.shape
        
        # Create one-hot targets for the top-k experts
        target_one_hot = torch.zeros_like(expert_logits)
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.experts_per_token):
                    expert_idx = target_experts[b, s, k]
                    if expert_idx < self.num_experts:
                        target_one_hot[b, s, expert_idx] = 1.0
        
        # Simple multi-label cross-entropy loss
        # Normalize targets to sum to 1 for each position
        target_sum = target_one_hot.sum(dim=-1, keepdim=True)
        target_sum = torch.where(target_sum > 0, target_sum, torch.ones_like(target_sum))
        target_normalized = target_one_hot / target_sum
        
        # Compute cross-entropy loss
        log_probs = F.log_softmax(expert_logits, dim=-1)
        loss = -(target_normalized * log_probs).sum(dim=-1).mean()
        
        # Ensure loss is finite
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=expert_logits.device, requires_grad=True)
        
        return {
            'total_loss': loss,
            'expert_loss': loss
        }


def create_simple_qwen_predictor(config_file=None):
    """Create a simplified Qwen predictor"""
    # Use simple default config
    config = {
        'input_dim': 2048,
        'hidden_dim': 256,
        'num_experts': 60,
        'experts_per_token': 4
    }
    
    model = SimpleQwenPredictor(**config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Simple Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model