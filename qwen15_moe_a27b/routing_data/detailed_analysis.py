#!/usr/bin/env python3
"""
Detailed analysis of Qwen1.5-MoE-A2.7B routing patterns
"""

import pickle
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional
import sys
import os

# Add the script directory to path so we can import the data classes
sys.path.insert(0, '/data/research/specMoE/specMoE/qwen15_moe_a27b/scripts/collection')

@dataclass
class QwenMoEGatingDataPoint:
    """Represents a Qwen1.5-MoE-A2.7B gating data point for training"""
    layer_id: int
    hidden_states: torch.Tensor
    input_embeddings: torch.Tensor
    target_routing: torch.Tensor
    target_top_k: torch.Tensor
    prev_layer_gates: List[torch.Tensor]
    sequence_length: int
    token_ids: Optional[torch.Tensor]
    dataset_name: str
    sample_id: str

def detailed_routing_analysis(trace_file_path):
    """Perform detailed routing analysis"""
    
    print("=" * 60)
    print("DETAILED ROUTING ANALYSIS")
    print("=" * 60)
    
    # Load the trace file
    with open(trace_file_path, 'rb') as f:
        traces = pickle.load(f)
    
    # Analyze first few traces in detail
    for i, trace in enumerate(traces[:3]):
        print(f"\n📊 TRACE {i} ANALYSIS")
        print(f"Layer: {trace.layer_id}")
        print(f"Dataset: {trace.dataset_name}")
        print(f"Sample ID: {trace.sample_id}")
        print(f"Sequence length: {trace.sequence_length}")
        
        # Analyze routing tensor
        routing = trace.target_routing.cpu().numpy()
        top_k = trace.target_top_k.cpu().numpy()
        
        print(f"Routing tensor shape: {routing.shape}")
        print(f"Top-k tensor shape: {top_k.shape}")
        
        # Check routing values
        print(f"Routing tensor statistics:")
        print(f"  Min: {np.min(routing):.6f}")
        print(f"  Max: {np.max(routing):.6f}")
        print(f"  Mean: {np.mean(routing):.6f}")
        print(f"  Std: {np.std(routing):.6f}")
        
        # Analyze top-k indices
        print(f"Top-k indices shape: {top_k.shape}")
        print(f"Top-k indices range: {np.min(top_k)} to {np.max(top_k)}")
        
        # Check specific tokens
        batch_size, seq_len, num_experts = routing.shape
        print(f"Examining first few tokens:")
        
        for b in range(min(2, batch_size)):
            for s in range(min(5, seq_len)):
                active_experts = np.nonzero(routing[b, s, :])[0]
                expert_probs = routing[b, s, active_experts]
                top_k_experts = top_k[b, s, :]
                
                print(f"  Batch {b}, Token {s}:")
                print(f"    Active experts: {active_experts}")
                print(f"    Expert probabilities: {expert_probs}")
                print(f"    Top-k experts: {top_k_experts}")
                
                # Check if routing matches top-k
                routing_active = set(active_experts)
                topk_active = set(top_k_experts)
                print(f"    Routing/Top-k match: {routing_active == topk_active}")
                
                if len(active_experts) > 0:
                    print(f"    Sum of probabilities: {np.sum(expert_probs):.6f}")
        
        print("-" * 50)
    
    # Analyze routing patterns across layers
    print(f"\n🔄 CROSS-LAYER ROUTING ANALYSIS")
    
    # Group traces by layer
    layer_traces = {}
    for trace in traces:
        if trace.layer_id not in layer_traces:
            layer_traces[trace.layer_id] = []
        layer_traces[trace.layer_id].append(trace)
    
    # Analyze each layer
    for layer_id in sorted(layer_traces.keys()):
        layer_trace_list = layer_traces[layer_id]
        print(f"\nLayer {layer_id}:")
        
        if len(layer_trace_list) > 0:
            trace = layer_trace_list[0]  # Take first trace for this layer
            routing = trace.target_routing.cpu().numpy()
            top_k = trace.target_top_k.cpu().numpy()
            
            # Analyze expert usage distribution
            batch_size, seq_len, num_experts = routing.shape
            expert_usage = np.zeros(num_experts)
            
            for b in range(batch_size):
                for s in range(seq_len):
                    active_experts = np.nonzero(routing[b, s, :])[0]
                    expert_usage[active_experts] += 1
            
            active_experts_count = np.sum(expert_usage > 0)
            print(f"  Active experts: {active_experts_count}/{num_experts}")
            print(f"  Total activations: {np.sum(expert_usage)}")
            
            # Show top experts
            top_expert_indices = np.argsort(expert_usage)[-10:][::-1]
            top_expert_counts = expert_usage[top_expert_indices]
            print(f"  Top 10 experts: {list(zip(top_expert_indices, top_expert_counts))}")
            
            # Check routing consistency
            all_top_k = top_k.flatten()
            unique_experts = np.unique(all_top_k)
            print(f"  Unique experts in top-k: {len(unique_experts)}")
            print(f"  Expert range: {np.min(unique_experts)} to {np.max(unique_experts)}")
    
    # Analyze token-level patterns
    print(f"\n🎭 TOKEN-LEVEL PATTERN ANALYSIS")
    
    # Take a sample trace to analyze token patterns
    sample_trace = traces[0]
    routing = sample_trace.target_routing.cpu().numpy()
    top_k = sample_trace.target_top_k.cpu().numpy()
    
    batch_size, seq_len, num_experts = routing.shape
    
    # Analyze consecutive tokens
    print(f"Analyzing consecutive token patterns (first 10 tokens):")
    for s in range(min(10, seq_len)):
        top_k_experts = top_k[0, s, :]
        routing_probs = routing[0, s, top_k_experts]
        
        print(f"  Token {s}: experts {top_k_experts} with probs {routing_probs}")
    
    # Check for pattern consistency
    print(f"\nChecking routing consistency:")
    consistent_count = 0
    total_count = 0
    
    for b in range(batch_size):
        for s in range(seq_len):
            top_k_experts = set(top_k[b, s, :])
            routing_active = set(np.nonzero(routing[b, s, :])[0])
            
            if top_k_experts == routing_active:
                consistent_count += 1
            total_count += 1
    
    consistency_rate = consistent_count / total_count if total_count > 0 else 0
    print(f"Routing consistency: {consistent_count}/{total_count} ({consistency_rate:.1%})")
    
    return {
        'layer_traces': layer_traces,
        'consistency_rate': consistency_rate,
        'num_experts': num_experts,
        'routing_shape': routing.shape
    }

def main():
    trace_file = "/data/research/specMoE/specMoE/qwen15_moe_a27b/routing_data/qwen15_moe_a27b_traces_small.pkl"
    
    if not os.path.exists(trace_file):
        print(f"Error: Trace file not found at {trace_file}")
        return
    
    try:
        results = detailed_routing_analysis(trace_file)
        print(f"\n✅ Detailed analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()