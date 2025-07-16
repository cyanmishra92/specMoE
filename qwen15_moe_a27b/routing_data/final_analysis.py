#!/usr/bin/env python3
"""
Final comprehensive analysis of Qwen1.5-MoE-A2.7B routing patterns
"""

import pickle
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional
import sys
import os
from collections import defaultdict, Counter

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

def analyze_multi_expert_patterns(trace_file_path):
    """Comprehensive analysis of Qwen1.5-MoE-A2.7B multi-expert activation patterns"""
    
    print("=" * 70)
    print("QWEN1.5-MoE-A2.7B MULTI-EXPERT ACTIVATION PATTERN ANALYSIS")
    print("=" * 70)
    
    # Load the trace file
    with open(trace_file_path, 'rb') as f:
        traces = pickle.load(f)
    
    print(f"📊 DATASET OVERVIEW")
    print(f"Total traces: {len(traces)}")
    print(f"Layers: {len(set(trace.layer_id for trace in traces))}")
    print(f"Datasets: {set(trace.dataset_name for trace in traces)}")
    
    # Get basic model info from first trace
    first_trace = traces[0]
    batch_size, seq_len, num_experts = first_trace.target_routing.shape
    top_k = first_trace.target_top_k.shape[-1]
    
    print(f"Model configuration:")
    print(f"  Number of experts: {num_experts}")
    print(f"  Top-k routing: {top_k}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # The key insight: target_routing contains negative log-probabilities (logits)
    # Only the top-k experts are actually "active" as indicated by target_top_k
    
    print(f"\n🎯 MULTI-EXPERT ACTIVATION ANALYSIS")
    
    # Analyze routing patterns
    layer_stats = defaultdict(lambda: {
        'expert_usage': defaultdict(int),
        'top_k_pairs': defaultdict(int),
        'routing_probs': [],
        'expert_diversity': [],
        'token_count': 0
    })
    
    for trace in traces:
        layer_id = trace.layer_id
        routing_logits = trace.target_routing.cpu().numpy()  # These are logits (negative log probs)
        top_k_indices = trace.target_top_k.cpu().numpy()
        
        batch_size, seq_len, num_experts = routing_logits.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                layer_stats[layer_id]['token_count'] += 1
                
                # Get the top-k experts for this token
                top_k_experts = top_k_indices[b, s, :]
                
                # Convert logits to probabilities for the top-k experts
                top_k_logits = routing_logits[b, s, top_k_experts]
                top_k_probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
                
                # Store routing probabilities
                layer_stats[layer_id]['routing_probs'].extend(top_k_probs)
                
                # Count expert usage
                for expert_idx in top_k_experts:
                    layer_stats[layer_id]['expert_usage'][expert_idx] += 1
                
                # Count top-k pairs (for diversity analysis)
                expert_pair = tuple(sorted(top_k_experts))
                layer_stats[layer_id]['top_k_pairs'][expert_pair] += 1
                
                # Expert diversity (number of unique experts in top-k)
                layer_stats[layer_id]['expert_diversity'].append(len(set(top_k_experts)))
    
    # Analyze results
    print(f"\n🔍 LAYER-BY-LAYER ANALYSIS")
    
    for layer_id in sorted(layer_stats.keys()):
        stats = layer_stats[layer_id]
        print(f"\nLayer {layer_id}:")
        print(f"  Total tokens analyzed: {stats['token_count']}")
        
        # Expert usage distribution
        expert_usage = stats['expert_usage']
        active_experts = len(expert_usage)
        total_activations = sum(expert_usage.values())
        
        print(f"  Expert usage: {active_experts}/{num_experts} experts used")
        print(f"  Total activations: {total_activations}")
        
        # Top used experts
        top_experts = Counter(expert_usage).most_common(10)
        print(f"  Top 10 experts: {top_experts}")
        
        # Expert load balancing
        if expert_usage:
            usage_values = list(expert_usage.values())
            usage_std = np.std(usage_values)
            usage_mean = np.mean(usage_values)
            print(f"  Load balancing (std/mean): {usage_std/usage_mean:.3f}")
        
        # Routing probability analysis
        routing_probs = np.array(stats['routing_probs'])
        print(f"  Routing probabilities:")
        print(f"    Mean: {np.mean(routing_probs):.4f}")
        print(f"    Std: {np.std(routing_probs):.4f}")
        print(f"    Min: {np.min(routing_probs):.4f}")
        print(f"    Max: {np.max(routing_probs):.4f}")
        
        # Top-k pair diversity
        top_k_pairs = stats['top_k_pairs']
        unique_pairs = len(top_k_pairs)
        total_pairs = sum(top_k_pairs.values())
        print(f"  Expert pair diversity: {unique_pairs} unique pairs from {total_pairs} total")
        
        # Most common expert pairs
        common_pairs = Counter(top_k_pairs).most_common(5)
        print(f"  Most common expert pairs: {common_pairs}")
        
        # Expert diversity per token
        diversity = stats['expert_diversity']
        print(f"  Expert diversity per token: {np.mean(diversity):.2f} (should be {top_k})")
    
    # Cross-layer analysis
    print(f"\n🔄 CROSS-LAYER PATTERNS")
    
    # Expert usage across layers
    global_expert_usage = defaultdict(int)
    for layer_id in layer_stats:
        for expert_id, count in layer_stats[layer_id]['expert_usage'].items():
            global_expert_usage[expert_id] += count
    
    print(f"Global expert usage across all layers:")
    total_global_activations = sum(global_expert_usage.values())
    active_global_experts = len(global_expert_usage)
    print(f"  Total experts used: {active_global_experts}/{num_experts}")
    print(f"  Total activations: {total_global_activations}")
    
    # Expert specialization analysis
    expert_layer_usage = defaultdict(lambda: defaultdict(int))
    for layer_id in layer_stats:
        for expert_id, count in layer_stats[layer_id]['expert_usage'].items():
            expert_layer_usage[expert_id][layer_id] = count
    
    print(f"\nExpert specialization analysis:")
    specialized_experts = 0
    for expert_id in sorted(global_expert_usage.keys())[:10]:  # Top 10 experts
        layer_usage = expert_layer_usage[expert_id]
        layers_used = len(layer_usage)
        total_usage = sum(layer_usage.values())
        
        if layers_used <= 3:  # Highly specialized
            specialized_experts += 1
            print(f"  Expert {expert_id}: {total_usage} activations across {layers_used} layers - SPECIALIZED")
        else:
            print(f"  Expert {expert_id}: {total_usage} activations across {layers_used} layers - GENERAL")
    
    print(f"  Specialized experts (≤3 layers): {specialized_experts}")
    
    # Compare to single-expert patterns
    print(f"\n🔄 MULTI-EXPERT vs SINGLE-EXPERT COMPARISON")
    
    print("Qwen1.5-MoE-A2.7B (Multi-Expert) characteristics:")
    print(f"  ✓ Top-{top_k} routing: Multiple experts per token")
    print(f"  ✓ {num_experts} experts per layer")
    print(f"  ✓ Load balancing across {active_global_experts} active experts")
    print(f"  ✓ Expert specialization: {specialized_experts} specialized experts")
    
    print("\\nTypical Switch Transformer (Single-Expert) characteristics:")
    print("  • Top-1 routing: One expert per token")
    print("  • 8-64 experts per layer")
    print("  • Load balancing through auxiliary loss")
    print("  • Expert specialization by domain/task")
    
    # Routing efficiency analysis
    print(f"\n⚡ ROUTING EFFICIENCY ANALYSIS")
    
    # Calculate routing concentration
    all_routing_probs = []
    for layer_id in layer_stats:
        all_routing_probs.extend(layer_stats[layer_id]['routing_probs'])
    
    all_routing_probs = np.array(all_routing_probs)
    
    # Entropy of routing distribution (lower = more concentrated)
    routing_entropy = -np.sum(all_routing_probs * np.log2(all_routing_probs + 1e-10))
    max_entropy = np.log2(top_k)  # Maximum possible entropy for top-k
    
    print(f"Routing concentration:")
    print(f"  Entropy: {routing_entropy:.3f} / {max_entropy:.3f} (max)")
    print(f"  Concentration: {1 - routing_entropy/max_entropy:.3f}")
    
    # Expert load balance
    usage_distribution = np.array(list(global_expert_usage.values()))
    gini_coefficient = np.sum(np.abs(usage_distribution[:, None] - usage_distribution[None, :])) / (2 * len(usage_distribution) * np.sum(usage_distribution))
    
    print(f"Load balancing:")
    print(f"  Gini coefficient: {gini_coefficient:.3f} (0=perfect balance, 1=maximum imbalance)")
    print(f"  Active experts: {active_global_experts}/{num_experts} ({active_global_experts/num_experts:.1%})")
    
    return {
        'model_config': {
            'num_experts': num_experts,
            'top_k': top_k,
            'num_layers': len(layer_stats)
        },
        'layer_stats': dict(layer_stats),
        'global_expert_usage': dict(global_expert_usage),
        'routing_efficiency': {
            'entropy': routing_entropy,
            'concentration': 1 - routing_entropy/max_entropy,
            'gini_coefficient': gini_coefficient,
            'active_experts_ratio': active_global_experts/num_experts
        }
    }

def main():
    trace_file = "/data/research/specMoE/specMoE/qwen15_moe_a27b/routing_data/qwen15_moe_a27b_traces_small.pkl"
    
    if not os.path.exists(trace_file):
        print(f"Error: Trace file not found at {trace_file}")
        return
    
    try:
        results = analyze_multi_expert_patterns(trace_file)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"Key insights:")
        print(f"  • {results['model_config']['num_experts']} experts with top-{results['model_config']['top_k']} routing")
        print(f"  • {results['routing_efficiency']['active_experts_ratio']:.1%} of experts are actively used")
        print(f"  • Routing concentration: {results['routing_efficiency']['concentration']:.3f}")
        print(f"  • Load balance (Gini): {results['routing_efficiency']['gini_coefficient']:.3f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()