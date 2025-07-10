#!/usr/bin/env python3
"""
Edge-Optimized Speculation Model for Small GPUs/Jetson Devices
Extremely lightweight model for speculation prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinySpeculationModel(nn.Module):
    """Ultra-lightweight speculation model for edge devices"""
    
    def __init__(self, hidden_size, num_experts, prev_gate_size):
        super().__init__()
        
        # Extreme compression for edge devices
        self.tiny_hidden = 32   # Very small hidden size
        self.tiny_gate = 16     # Very small gate projection
        
        # Minimal projections
        self.hidden_proj = nn.Linear(hidden_size, self.tiny_hidden)
        self.prev_gate_proj = nn.Linear(prev_gate_size, self.tiny_gate)
        
        # Single layer prediction
        self.predictor = nn.Linear(self.tiny_hidden + self.tiny_gate, num_experts)
        
        # Total params: 512*32 + 8*16 + 48*8 = ~17K parameters
        
    def forward(self, hidden_states, prev_gate, mask=None):
        # Compress inputs maximally
        h_compressed = torch.relu(self.hidden_proj(hidden_states))
        g_compressed = torch.relu(self.prev_gate_proj(prev_gate))
        
        # Concat and predict
        combined = torch.cat([h_compressed, g_compressed], dim=-1)
        logits = self.predictor(combined)
        
        return logits

class SmallSpeculationModel(nn.Module):
    """Small but capable speculation model"""
    
    def __init__(self, hidden_size, num_experts, prev_gate_size):
        super().__init__()
        
        # Balanced compression
        self.small_hidden = 64
        self.small_gate = 32
        
        # Two-layer architecture
        self.hidden_proj = nn.Linear(hidden_size, self.small_hidden)
        self.prev_gate_proj = nn.Linear(prev_gate_size, self.small_gate)
        
        self.layer1 = nn.Linear(self.small_hidden + self.small_gate, 48)
        self.layer2 = nn.Linear(48, num_experts)
        self.dropout = nn.Dropout(0.1)
        
        # Total params: 512*64 + 8*32 + 96*48 + 48*8 = ~38K parameters
        
    def forward(self, hidden_states, prev_gate, mask=None):
        h_features = torch.relu(self.hidden_proj(hidden_states))
        g_features = torch.relu(self.prev_gate_proj(prev_gate))
        
        combined = torch.cat([h_features, g_features], dim=-1)
        
        x = torch.relu(self.layer1(combined))
        x = self.dropout(x)
        logits = self.layer2(x)
        
        return logits

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def compare_model_sizes():
    """Compare different model architectures"""
    
    hidden_size = 512
    num_experts = 8
    prev_gate_size = 8
    
    models = {
        'Tiny (Edge)': TinySpeculationModel(hidden_size, num_experts, prev_gate_size),
        'Small': SmallSpeculationModel(hidden_size, num_experts, prev_gate_size),
        'Current': RegularizedSpeculationModel(hidden_size, num_experts, prev_gate_size)
    }
    
    print("🔍 Model Size Comparison:")
    print("=" * 50)
    
    for name, model in models.items():
        total, trainable = count_parameters(model)
        memory_mb = total * 4 / (1024 * 1024)  # FP32 memory
        
        print(f"{name:12} | {total:6,} params | {memory_mb:.2f} MB")
    
    return models

class RegularizedSpeculationModel(nn.Module):
    """Current model for comparison"""
    
    def __init__(self, hidden_size, num_experts, prev_gate_size, dropout_rate=0.3):
        super().__init__()
        
        self.hidden_proj = nn.Linear(hidden_size, 128)
        self.prev_gate_proj = nn.Linear(prev_gate_size, 128)
        
        self.combined_layer1 = nn.Linear(256, 128)
        self.combined_layer2 = nn.Linear(128, 64)
        self.output_proj = nn.Linear(64, num_experts)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)
        
    def forward(self, hidden_states, prev_gate, mask=None):
        hidden_features = torch.relu(self.hidden_proj(hidden_states))
        hidden_features = self.dropout(hidden_features)
        
        gate_features = torch.relu(self.prev_gate_proj(prev_gate))
        gate_features = self.dropout(gate_features)
        
        combined = torch.cat([hidden_features, gate_features], dim=-1)
        
        combined = torch.relu(self.combined_layer1(combined))
        combined = self.layer_norm1(combined)
        combined = self.dropout(combined)
        
        combined = torch.relu(self.combined_layer2(combined))
        combined = self.layer_norm2(combined)
        combined = self.dropout(combined)
        
        logits = self.output_proj(combined)
        return logits

if __name__ == "__main__":
    models = compare_model_sizes()
    
    # Test inference speed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_states = torch.randn(1, 32, 512).to(device)
    prev_gate = torch.randn(1, 32, 8).to(device)
    
    print(f"\n⚡ Inference Speed Test on {device}:")
    print("=" * 40)
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(hidden_states, prev_gate)
        
        # Timing
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if device == "cuda":
            start_time.record()
        
        with torch.no_grad():
            for _ in range(100):
                logits = model(hidden_states, prev_gate)
        
        if device == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            elapsed_ms = start_time.elapsed_time(end_time)
        else:
            import time
            start = time.time()
            with torch.no_grad():
                for _ in range(100):
                    logits = model(hidden_states, prev_gate)
            elapsed_ms = (time.time() - start) * 1000
        
        avg_ms = elapsed_ms / 100
        print(f"{name:12} | {avg_ms:.3f} ms/inference")