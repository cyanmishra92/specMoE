"""
Hybrid Expert Predictor: Classification + Temporal Speculation
Inspired by ExpertFlow + adds temporal prediction for speculation

Architecture:
1. Immediate Classification: Which experts are active NOW (like ExpertFlow)
2. Temporal Prediction: Which experts will be active in FUTURE tokens
3. Hybrid Training: Joint loss for both tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExpertClassificationHead(nn.Module):
    """
    ExpertFlow-inspired classification head
    Predicts which experts are active for current token (binary classification)
    """
    def __init__(self, hidden_dim, num_experts=60):
        super().__init__()
        self.num_experts = num_experts
        
        # Binary classifier for each expert (active/inactive)
        self.expert_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts),  # Binary classification per expert
            nn.Sigmoid()  # Output probabilities [0,1] for each expert
        )
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            expert_probs: [batch, seq_len, num_experts] - probability each expert is active
        """
        return self.expert_classifier(hidden_states)


class TemporalPredictionHead(nn.Module):
    """
    Transformer-based temporal prediction for speculation
    Predicts which experts will be active in future tokens
    """
    def __init__(self, hidden_dim, num_experts=60, lookahead_steps=4, num_layers=2, num_heads=8):
        super().__init__()
        self.num_experts = num_experts
        self.lookahead_steps = lookahead_steps
        
        # Transformer layers for temporal modeling (like Switch approach)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Position encoding for temporal awareness
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim) * 0.02)  # Support up to 1000 seq len
        
        # Prediction heads for each future step
        self.future_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_experts),
                nn.Sigmoid()
            ) for _ in range(lookahead_steps)
        ])
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            future_predictions: [batch, seq_len, lookahead_steps, num_experts]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Add positional encoding for temporal awareness
        if seq_len <= self.pos_encoding.shape[0]:
            pos_embed = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            x = hidden_states + pos_embed
        else:
            x = hidden_states
        
        # Apply transformer layers for temporal modeling
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)  # [batch, seq_len, hidden_dim]
        
        # Predict future expert activations for each step
        future_preds = []
        for step_predictor in self.future_predictors:
            step_pred = step_predictor(x)  # [batch, seq_len, num_experts]
            future_preds.append(step_pred)
        
        # Stack predictions: [batch, seq_len, lookahead_steps, num_experts]
        future_predictions = torch.stack(future_preds, dim=2)
        
        return future_predictions


class HybridExpertPredictor(nn.Module):
    """
    Hybrid system combining immediate classification + temporal prediction
    """
    def __init__(self, input_dim=2048, hidden_dim=512, num_experts=60, 
                 lookahead_steps=4, experts_per_token=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.lookahead_steps = lookahead_steps
        self.experts_per_token = experts_per_token
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific heads
        self.classification_head = ExpertClassificationHead(hidden_dim, num_experts)
        self.temporal_head = TemporalPredictionHead(hidden_dim, num_experts, lookahead_steps)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Transformer layers handle their own initialization
                pass
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass
        
        Args:
            hidden_states: [batch, seq_len, input_dim]
            attention_mask: [batch, seq_len]
        """
        # Extract shared features
        features = self.feature_extractor(hidden_states)  # [batch, seq_len, hidden_dim]
        
        # Immediate classification (ExpertFlow-style)
        current_expert_probs = self.classification_head(features)  # [batch, seq_len, num_experts]
        
        # Temporal prediction for speculation
        future_expert_probs = self.temporal_head(features)  # [batch, seq_len, lookahead_steps, num_experts]
        
        # Convert probabilities to top-k predictions
        current_topk_logits, current_topk_indices = torch.topk(
            current_expert_probs, k=self.experts_per_token, dim=-1
        )
        
        # Future top-k predictions for each step
        future_topk_predictions = []
        for step in range(self.lookahead_steps):
            step_probs = future_expert_probs[:, :, step, :]  # [batch, seq_len, num_experts]
            step_logits, step_indices = torch.topk(step_probs, k=self.experts_per_token, dim=-1)
            future_topk_predictions.append({
                'logits': step_logits,
                'indices': step_indices
            })
        
        return {
            # Immediate predictions (classification)
            'current_expert_probs': current_expert_probs,
            'current_topk_logits': current_topk_logits,
            'current_topk_indices': current_topk_indices,
            
            # Future predictions (temporal)
            'future_expert_probs': future_expert_probs,
            'future_topk_predictions': future_topk_predictions,
            
            # For compatibility with existing code
            'expert_logits': current_expert_probs,  # Use current predictions as main logits
            'top_k_indices': current_topk_indices
        }
    
    def compute_loss(self, predictions, targets):
        """
        Hybrid loss combining classification + temporal prediction
        
        Args:
            predictions: Model predictions
            targets: Ground truth with current and future expert indices
        """
        device = predictions['current_expert_probs'].device
        
        # Current expert targets
        current_targets = targets['expert_indices']  # [batch, seq_len, 4]
        batch_size, seq_len, _ = current_targets.shape
        
        # === IMMEDIATE CLASSIFICATION LOSS ===
        # Convert expert indices to binary targets
        current_binary_targets = torch.zeros_like(predictions['current_expert_probs'])
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.experts_per_token):
                    expert_idx = current_targets[b, s, k]
                    if expert_idx < self.num_experts:
                        current_binary_targets[b, s, expert_idx] = 1.0
        
        # Binary classification loss (like ExpertFlow)
        classification_loss = F.binary_cross_entropy(
            predictions['current_expert_probs'],
            current_binary_targets,
            reduction='mean'
        )
        
        # === TEMPORAL PREDICTION LOSS ===
        temporal_loss = torch.tensor(0.0, device=device)
        
        # If we have future targets, compute temporal loss
        if 'future_expert_indices' in targets:
            future_targets = targets['future_expert_indices']  # [batch, seq_len, lookahead_steps, 4]
            
            for step in range(self.lookahead_steps):
                # Binary targets for this future step
                step_binary_targets = torch.zeros(batch_size, seq_len, self.num_experts, device=device)
                
                for b in range(batch_size):
                    for s in range(seq_len):
                        if s + step + 1 < seq_len:  # Ensure we don't go beyond sequence
                            for k in range(self.experts_per_token):
                                expert_idx = future_targets[b, s, step, k]
                                if expert_idx < self.num_experts:
                                    step_binary_targets[b, s, expert_idx] = 1.0
                
                # Loss for this step
                step_loss = F.binary_cross_entropy(
                    predictions['future_expert_probs'][:, :, step, :],
                    step_binary_targets,
                    reduction='mean'
                )
                temporal_loss += step_loss
            
            temporal_loss = temporal_loss / self.lookahead_steps
        
        # === COMBINED LOSS ===
        total_loss = classification_loss + 0.3 * temporal_loss  # Weight temporal loss lower initially
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'temporal_loss': temporal_loss
        }


def create_hybrid_predictor(config=None):
    """Create hybrid expert predictor"""
    default_config = {
        'input_dim': 2048,
        'hidden_dim': 512,
        'num_experts': 60,
        'lookahead_steps': 4,
        'experts_per_token': 4
    }
    
    if config:
        default_config.update(config)
    
    model = HybridExpertPredictor(**default_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Hybrid Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"- Classification head: {sum(p.numel() for p in model.classification_head.parameters()):,}")
    print(f"- Temporal head: {sum(p.numel() for p in model.temporal_head.parameters()):,}")
    print(f"- Shared features: {sum(p.numel() for p in model.feature_extractor.parameters()):,}")
    
    return model