"""
Small Switch Transformer implementation optimized for RTX 3090
Based on Switch Transformer paper but scaled down for single GPU training/inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class SmallSwitchMLP(nn.Module):
    """
    Small MoE MLP layer with 8 experts, suitable for RTX 3090
    Each expert is ~4M parameters (512 -> 2048 -> 512)
    """
    def __init__(self, hidden_size: int = 512, num_experts: int = 8, top_k: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = 1.0
        
        # Gating network
        self.gating = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Expert networks - small but sufficient for experimentation
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),  # 512 -> 2048
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size),  # 2048 -> 512
            ) for _ in range(num_experts)
        ])
        
        # For collecting routing information (used by speculation engine)
        self.routing_history = []
        self.gate_scores_history = []
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        
        # Compute gating scores
        gate_logits = self.gating(x_flat)  # [batch_size * seq_len, num_experts]
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Store for speculation engine
        self.gate_scores_history.append(gate_scores.detach().clone())
        
        # Get top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # Store routing info for later use
        routing_info = {
            'gate_scores': gate_scores.view(batch_size, seq_len, self.num_experts),
            'top_k_indices': top_k_indices.view(batch_size, seq_len, self.top_k),
            'top_k_scores': top_k_scores.view(batch_size, seq_len, self.top_k)
        }
        
        # Continue with MoE computation...
        return self._complete_forward(x_flat, gate_scores, top_k_indices, batch_size, seq_len, routing_info)
    
    def forward_with_routing(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass that returns detailed routing information"""
        return self.forward(x)
    
    def _complete_forward(self, x_flat, gate_scores, top_k_indices_input, batch_size, seq_len, routing_info):
        if len(self.gate_scores_history) > 4:  # Keep last 4 layers
            self.gate_scores_history.pop(0)
        
        # Use the already computed top_k values
        top_k_gates = routing_info['top_k_scores'].view(-1, self.top_k)
        top_k_indices = top_k_indices_input
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Update routing information for analysis
        routing_info.update({
            'top_k_gates': top_k_gates.view(batch_size, seq_len, self.top_k),
            'load_balancing_loss': self._compute_load_balancing_loss(gate_scores)
        })
        
        # Store routing for speculation
        self.routing_history.append({
            'top_k_indices': top_k_indices.detach().clone(),
            'gate_scores': gate_scores.detach().clone()
        })
        if len(self.routing_history) > 4:
            self.routing_history.pop(0)
        
        # Process tokens through selected experts
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            expert_tokens = x_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Weighted combination for tokens using this expert
            for k in range(self.top_k):
                k_mask = (top_k_indices[:, k] == expert_idx) & expert_mask
                if k_mask.any():
                    weights = top_k_gates[k_mask, k:k+1]
                    output[k_mask] += weights * expert_output[:weights.shape[0]]
        
        output = output.view(batch_size, seq_len, self.hidden_size)
        return output, routing_info
    
    def _compute_load_balancing_loss(self, gate_scores: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage expert utilization"""
        # Fraction of tokens routed to each expert
        router_probs = gate_scores.mean(dim=0)
        # Fraction of total router probability allocated to each expert
        expert_mask = F.one_hot(gate_scores.argmax(dim=-1), self.num_experts).float()
        expert_usage = expert_mask.mean(dim=0)
        
        # Load balancing loss
        return self.num_experts * torch.sum(router_probs * expert_usage)
    
    def get_speculation_data(self) -> Dict:
        """Get data for speculation engine"""
        return {
            'routing_history': self.routing_history,
            'gate_scores_history': self.gate_scores_history
        }


class SmallSwitchTransformerLayer(nn.Module):
    """Single transformer layer with MoE MLP"""
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, num_experts: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # MoE MLP
        self.moe_mlp = SmallSwitchMLP(hidden_size, num_experts)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.attention_norm(x + attn_output)
        
        # MoE MLP with residual connection
        mlp_output, routing_info = self.moe_mlp(x)
        x = self.mlp_norm(x + mlp_output)
        
        return x, routing_info


class SmallSwitchTransformer(nn.Module):
    """
    Small Switch Transformer model for experimentation
    - 6 layers, 512 hidden size, 8 heads, 8 experts per layer
    - Total: ~50M parameters (manageable on RTX 3090)
    """
    def __init__(
        self, 
        vocab_size: int = 32000,
        hidden_size: int = 512, 
        num_layers: int = 6,
        num_heads: int = 8,
        num_experts: int = 8,
        max_position_embeddings: int = 512
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_experts = num_experts
        
        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SmallSwitchTransformerLayer(hidden_size, num_heads, num_experts)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # For collecting all routing information
        self.routing_statistics = []
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        collect_routing_stats: bool = True
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        
        # Track routing info across layers
        layer_routing_info = []
        load_balancing_losses = []
        
        # Pass through transformer layers
        for layer_idx, layer in enumerate(self.layers):
            x, routing_info = layer(x, attention_mask)
            layer_routing_info.append(routing_info)
            load_balancing_losses.append(routing_info['load_balancing_loss'])
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Collect routing statistics
        if collect_routing_stats:
            self._collect_routing_statistics(layer_routing_info)
        
        return {
            'logits': logits,
            'routing_info': layer_routing_info,
            'load_balancing_loss': torch.stack(load_balancing_losses).mean(),
            'speculation_data': self._get_speculation_data()
        }
    
    def _collect_routing_statistics(self, layer_routing_info: List[Dict]):
        """Collect routing statistics for analysis"""
        stats = {
            'expert_usage': [],
            'routing_entropy': [],
            'load_balance': []
        }
        
        for layer_info in layer_routing_info:
            gate_scores = layer_info['gate_scores']
            
            # Expert usage distribution
            expert_usage = gate_scores.mean(dim=0)
            stats['expert_usage'].append(expert_usage.cpu())
            
            # Routing entropy (measure of routing diversity)
            entropy = -torch.sum(gate_scores * torch.log(gate_scores + 1e-8), dim=-1).mean()
            stats['routing_entropy'].append(entropy.cpu())
            
            # Load balance (how evenly distributed)
            load_balance = 1.0 - torch.std(expert_usage) / torch.mean(expert_usage)
            stats['load_balance'].append(load_balance.cpu())
        
        self.routing_statistics.append(stats)
    
    def _get_speculation_data(self) -> Dict:
        """Get speculation data from all MoE layers"""
        speculation_data = {}
        for layer_idx, layer in enumerate(self.layers):
            speculation_data[f'layer_{layer_idx}'] = layer.moe_mlp.get_speculation_data()
        return speculation_data
    
    def get_model_info(self) -> Dict:
        """Get model information for device profiling"""
        total_params = sum(p.numel() for p in self.parameters())
        expert_params = sum(p.numel() for layer in self.layers for expert in layer.moe_mlp.experts for p in expert.parameters())
        
        return {
            'total_parameters': total_params,
            'expert_parameters': expert_params,
            'non_expert_parameters': total_params - expert_params,
            'num_layers': self.num_layers,
            'num_experts_per_layer': self.num_experts,
            'hidden_size': self.hidden_size,
            'memory_per_expert_mb': expert_params * 4 / (1024 * 1024)  # FP32
        }


def create_small_switch_model(vocab_size: int = 32000) -> SmallSwitchTransformer:
    """Create a small Switch Transformer suitable for RTX 3090"""
    return SmallSwitchTransformer(
        vocab_size=vocab_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        num_experts=8,
        max_position_embeddings=512
    )


if __name__ == "__main__":
    # Test model creation and basic forward pass
    model = create_small_switch_model()
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Load balancing loss: {outputs['load_balancing_loss'].item():.4f}")
        print(f"Number of layers with routing info: {len(outputs['routing_info'])}")