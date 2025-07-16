#!/usr/bin/env python3
"""
Analyze Qwen1.5-MoE-A2.7B trace file to understand multi-expert activation patterns
"""

import pickle
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
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

def analyze_trace_file(trace_file_path):
    """Analyze the Qwen1.5-MoE-A2.7B trace file"""
    
    print("=" * 60)
    print("QWEN1.5-MoE-A2.7B TRACE ANALYSIS")
    print("=" * 60)
    
    # Load the trace file
    with open(trace_file_path, 'rb') as f:
        traces = pickle.load(f)
    
    print(f"\n📊 BASIC STATISTICS")
    print(f"Total traces: {len(traces)}")
    print(f"Data type: {type(traces[0])}")
    
    # Analyze trace structure
    first_trace = traces[0]
    print(f"\nFirst trace structure:")
    print(f"  Layer ID: {first_trace.layer_id}")
    print(f"  Hidden states shape: {first_trace.hidden_states.shape}")
    print(f"  Input embeddings shape: {first_trace.input_embeddings.shape if first_trace.input_embeddings is not None else 'None'}")
    print(f"  Target routing shape: {first_trace.target_routing.shape}")
    print(f"  Target top-k shape: {first_trace.target_top_k.shape}")
    print(f"  Sequence length: {first_trace.sequence_length}")
    print(f"  Dataset: {first_trace.dataset_name}")
    print(f"  Sample ID: {first_trace.sample_id}")
    
    # Extract key metrics
    layers_count = defaultdict(int)
    dataset_count = defaultdict(int)
    sequence_lengths = []
    top_k_values = []
    expert_activations = defaultdict(list)
    
    # Analyze each trace
    for trace in traces:
        layers_count[trace.layer_id] += 1
        dataset_count[trace.dataset_name] += 1
        sequence_lengths.append(trace.sequence_length)
        
        # Extract top-k expert indices
        top_k_indices = trace.target_top_k.cpu().numpy()
        top_k_values.append(top_k_indices.shape[-1])  # Number of experts per token
        
        # Extract expert activation patterns
        routing_tensor = trace.target_routing.cpu().numpy()
        active_experts = np.nonzero(routing_tensor)[0]  # Get indices of active experts
        expert_activations[trace.layer_id].extend(active_experts)
    
    print(f"\n🔍 LAYER ANALYSIS")
    print(f"Number of layers: {len(layers_count)}")
    print(f"Layer distribution:")
    for layer_id in sorted(layers_count.keys()):
        print(f"  Layer {layer_id}: {layers_count[layer_id]} traces")
    
    print(f"\n📚 DATASET ANALYSIS")
    print(f"Number of datasets: {len(dataset_count)}")
    for dataset, count in sorted(dataset_count.items()):
        print(f"  {dataset}: {count} traces")
    
    print(f"\n📏 SEQUENCE LENGTH ANALYSIS")
    seq_lengths = np.array(sequence_lengths)
    print(f"Min sequence length: {np.min(seq_lengths)}")
    print(f"Max sequence length: {np.max(seq_lengths)}")
    print(f"Mean sequence length: {np.mean(seq_lengths):.2f}")
    print(f"Median sequence length: {np.median(seq_lengths):.2f}")
    
    print(f"\n⚡ TOP-K EXPERT ANALYSIS")
    top_k_array = np.array(top_k_values)
    print(f"Top-k values: {np.unique(top_k_array)}")
    print(f"Most common top-k: {Counter(top_k_values).most_common(3)}")
    
    # Analyze expert activation patterns
    print(f"\n🎯 EXPERT ACTIVATION PATTERNS")
    
    # Get number of experts from routing tensor shape
    first_routing = traces[0].target_routing.cpu().numpy()
    num_experts = first_routing.shape[-1]
    print(f"Number of experts: {num_experts}")
    
    # Analyze expert usage per layer
    print(f"\nExpert usage by layer:")
    for layer_id in sorted(expert_activations.keys()):
        layer_experts = expert_activations[layer_id]
        if layer_experts:
            expert_counter = Counter(layer_experts)
            total_activations = len(layer_experts)
            unique_experts = len(expert_counter)
            print(f"  Layer {layer_id}: {unique_experts}/{num_experts} experts used, {total_activations} total activations")
            
            # Show top 5 most used experts
            top_experts = expert_counter.most_common(5)
            print(f"    Top experts: {top_experts}")
    
    # Analyze routing tensor structure
    print(f"\n🧠 ROUTING TENSOR ANALYSIS")
    sample_routing = traces[0].target_routing.cpu().numpy()
    print(f"Routing tensor shape: {sample_routing.shape}")
    print(f"Routing tensor dtype: {sample_routing.dtype}")
    
    # Check sparsity
    non_zero_count = np.count_nonzero(sample_routing)
    total_elements = sample_routing.size
    sparsity = 1 - (non_zero_count / total_elements)
    print(f"Sparsity: {sparsity:.4f} ({non_zero_count}/{total_elements} non-zero)")
    
    # Analyze routing probability distribution
    non_zero_values = sample_routing[sample_routing > 0]
    if len(non_zero_values) > 0:
        print(f"Non-zero routing values:")
        print(f"  Min: {np.min(non_zero_values):.6f}")
        print(f"  Max: {np.max(non_zero_values):.6f}")
        print(f"  Mean: {np.mean(non_zero_values):.6f}")
        print(f"  Std: {np.std(non_zero_values):.6f}")
    
    # Multi-expert vs single-expert comparison
    print(f"\n🔄 MULTI-EXPERT vs SINGLE-EXPERT COMPARISON")
    
    # Count tokens with multiple active experts
    multi_expert_count = 0
    single_expert_count = 0
    
    for trace in traces[:1000]:  # Sample first 1000 traces for efficiency
        routing_tensor = trace.target_routing.cpu().numpy()
        active_experts_per_token = np.sum(routing_tensor > 0, axis=-1)
        
        for active_count in active_experts_per_token.flatten():
            if active_count > 1:
                multi_expert_count += 1
            elif active_count == 1:
                single_expert_count += 1
    
    total_analyzed = multi_expert_count + single_expert_count
    if total_analyzed > 0:
        print(f"Single expert tokens: {single_expert_count} ({single_expert_count/total_analyzed*100:.1f}%)")
        print(f"Multi-expert tokens: {multi_expert_count} ({multi_expert_count/total_analyzed*100:.1f}%)")
    
    # Token-level analysis
    print(f"\n🎭 TOKEN-LEVEL ANALYSIS")
    token_ids = [trace.token_ids for trace in traces[:100] if trace.token_ids is not None]
    print(f"Sample of token IDs: {token_ids[:10]}")
    
    return {
        'total_traces': len(traces),
        'num_layers': len(layers_count),
        'num_experts': num_experts,
        'layers_count': dict(layers_count),
        'dataset_count': dict(dataset_count),
        'sequence_lengths': seq_lengths,
        'top_k_values': top_k_array,
        'expert_activations': dict(expert_activations),
        'sparsity': sparsity,
        'multi_expert_ratio': multi_expert_count / total_analyzed if total_analyzed > 0 else 0
    }

def main():
    trace_file = "/data/research/specMoE/specMoE/qwen15_moe_a27b/routing_data/qwen15_moe_a27b_traces_small.pkl"
    
    if not os.path.exists(trace_file):
        print(f"Error: Trace file not found at {trace_file}")
        return
    
    try:
        results = analyze_trace_file(trace_file)
        print(f"\n✅ Analysis complete!")
        print(f"Key findings:")
        print(f"  - {results['total_traces']} traces across {results['num_layers']} layers")
        print(f"  - {results['num_experts']} experts per layer")
        print(f"  - {results['sparsity']:.1%} sparsity in routing")
        print(f"  - {results['multi_expert_ratio']:.1%} multi-expert tokens")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()