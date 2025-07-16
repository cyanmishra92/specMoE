#!/usr/bin/env python3
"""
Analyze Qwen1.5-MoE-A2.7B trace patterns to understand multi-expert activation
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from pathlib import Path
import sys
sys.path.append('../scripts/collection')
from dataclasses import dataclass
from typing import List, Optional

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

def load_traces(trace_file):
    """Load traces from pickle file"""
    with open(trace_file, 'rb') as f:
        traces = pickle.load(f)
    print(f"Loaded {len(traces)} traces from {trace_file}")
    return traces

def analyze_routing_patterns(traces):
    """Analyze routing patterns in traces"""
    print("=== Qwen1.5-MoE-A2.7B Routing Analysis ===")
    
    # Collect routing statistics
    layer_stats = defaultdict(lambda: {
        'expert_counts': defaultdict(int),
        'top_k_patterns': defaultdict(int),
        'routing_entropy': [],
        'expert_pairs': defaultdict(int)
    })
    
    for trace in traces:
        layer_id = trace.layer_id
        
        # Get routing logits and top-k indices
        routing_logits = trace.target_routing  # Shape: [batch, seq_len, num_experts]
        top_k_indices = trace.target_top_k     # Shape: [batch, seq_len, k]
        
        # Handle different tensor shapes
        if routing_logits.dim() == 2:
            routing_logits = routing_logits.unsqueeze(1)
        if top_k_indices.dim() == 2:
            top_k_indices = top_k_indices.unsqueeze(1)
        
        batch_size, seq_len, num_experts = routing_logits.shape
        k = top_k_indices.shape[-1]
        
        # Analyze each token's routing
        for b in range(batch_size):
            for s in range(seq_len):
                # Get routing distribution for this token
                logits = routing_logits[b, s]
                top_k = top_k_indices[b, s]
                
                # Count expert activations
                for expert_idx in top_k:
                    if expert_idx < num_experts:  # Valid expert
                        layer_stats[layer_id]['expert_counts'][expert_idx.item()] += 1
                
                # Record top-k pattern
                if k == 2:  # Top-2 routing
                    expert1, expert2 = top_k[0].item(), top_k[1].item()
                    if expert1 < num_experts and expert2 < num_experts:
                        pattern = tuple(sorted([expert1, expert2]))
                        layer_stats[layer_id]['top_k_patterns'][pattern] += 1
                        layer_stats[layer_id]['expert_pairs'][pattern] += 1
                
                # Calculate routing entropy
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                layer_stats[layer_id]['routing_entropy'].append(entropy.item())
    
    return layer_stats, num_experts

def print_routing_statistics(layer_stats, num_experts):
    """Print detailed routing statistics"""
    print(f"\n=== Multi-Expert Activation Analysis (8 experts) ===")
    
    for layer_id in sorted(layer_stats.keys()):
        stats = layer_stats[layer_id]
        print(f"\nLayer {layer_id}:")
        
        # Expert usage distribution
        expert_counts = stats['expert_counts']
        total_activations = sum(expert_counts.values())
        print(f"  Total activations: {total_activations}")
        
        # Show expert usage percentages
        print("  Expert usage:")
        for expert_id in range(num_experts):
            count = expert_counts.get(expert_id, 0)
            percentage = (count / total_activations) * 100 if total_activations > 0 else 0
            print(f"    Expert {expert_id}: {count:4d} ({percentage:5.1f}%)")
        
        # Top expert pairs (for Top-2 routing)
        print("  Top expert pairs:")
        top_pairs = sorted(stats['expert_pairs'].items(), key=lambda x: x[1], reverse=True)[:5]
        for pair, count in top_pairs:
            percentage = (count / len(stats['routing_entropy'])) * 100 if stats['routing_entropy'] else 0
            print(f"    Experts {pair}: {count:3d} ({percentage:5.1f}%)")
        
        # Routing entropy
        entropies = stats['routing_entropy']
        if entropies:
            avg_entropy = np.mean(entropies)
            print(f"  Avg routing entropy: {avg_entropy:.3f}")
    
    # Overall statistics
    print(f"\n=== Overall Statistics ===")
    all_expert_counts = defaultdict(int)
    total_tokens = 0
    
    for layer_id, stats in layer_stats.items():
        for expert_id, count in stats['expert_counts'].items():
            all_expert_counts[expert_id] += count
            total_tokens += count
    
    print(f"Total tokens processed: {total_tokens}")
    print("Overall expert usage:")
    for expert_id in range(num_experts):
        count = all_expert_counts.get(expert_id, 0)
        percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"  Expert {expert_id}: {count:5d} ({percentage:5.1f}%)")

def visualize_routing_patterns(layer_stats, num_experts, output_dir):
    """Create visualizations of routing patterns"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Expert usage heatmap across layers
    layers = sorted(layer_stats.keys())
    expert_usage = np.zeros((len(layers), num_experts))
    
    for i, layer_id in enumerate(layers):
        stats = layer_stats[layer_id]
        total_activations = sum(stats['expert_counts'].values())
        
        for expert_id in range(num_experts):
            count = stats['expert_counts'].get(expert_id, 0)
            expert_usage[i, expert_id] = (count / total_activations) * 100 if total_activations > 0 else 0
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(expert_usage, 
                xticklabels=[f'Expert {i}' for i in range(num_experts)],
                yticklabels=[f'Layer {i}' for i in layers],
                annot=True, fmt='.1f', cmap='Blues')
    plt.title('Expert Usage Percentage Across Layers')
    plt.ylabel('Layer')
    plt.xlabel('Expert')
    plt.tight_layout()
    plt.savefig(output_dir / 'expert_usage_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Routing entropy distribution
    plt.figure(figsize=(12, 6))
    for layer_id in layers[:6]:  # Show first 6 layers
        entropies = layer_stats[layer_id]['routing_entropy']
        if entropies:
            plt.hist(entropies, bins=30, alpha=0.7, label=f'Layer {layer_id}')
    
    plt.xlabel('Routing Entropy')
    plt.ylabel('Frequency')
    plt.title('Routing Entropy Distribution by Layer')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'routing_entropy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Expert pair co-occurrence matrix
    # Aggregate expert pairs across all layers
    pair_counts = defaultdict(int)
    for layer_id, stats in layer_stats.items():
        for pair, count in stats['expert_pairs'].items():
            pair_counts[pair] += count
    
    # Create co-occurrence matrix
    cooccurrence = np.zeros((num_experts, num_experts))
    for (expert1, expert2), count in pair_counts.items():
        cooccurrence[expert1, expert2] += count
        cooccurrence[expert2, expert1] += count
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence, 
                xticklabels=[f'Expert {i}' for i in range(num_experts)],
                yticklabels=[f'Expert {i}' for i in range(num_experts)],
                annot=True, fmt='.0f', cmap='Reds')
    plt.title('Expert Co-occurrence Matrix (Top-2 Routing)')
    plt.xlabel('Expert')
    plt.ylabel('Expert')
    plt.tight_layout()
    plt.savefig(output_dir / 'expert_cooccurrence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def analyze_prediction_challenges(layer_stats, num_experts):
    """Analyze challenges for multi-expert prediction"""
    print(f"\n=== Multi-Expert Prediction Challenges ===")
    
    # Calculate prediction complexity metrics
    total_unique_pairs = 0
    total_pair_entropy = 0
    
    for layer_id, stats in layer_stats.items():
        pairs = stats['expert_pairs']
        unique_pairs = len(pairs)
        total_unique_pairs += unique_pairs
        
        # Calculate pair distribution entropy
        pair_counts = list(pairs.values())
        if pair_counts:
            total_count = sum(pair_counts)
            probs = [count / total_count for count in pair_counts]
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            total_pair_entropy += entropy
            
            print(f"Layer {layer_id}: {unique_pairs} unique pairs, entropy: {entropy:.3f}")
    
    avg_unique_pairs = total_unique_pairs / len(layer_stats)
    avg_pair_entropy = total_pair_entropy / len(layer_stats)
    
    print(f"Average unique pairs per layer: {avg_unique_pairs:.1f}")
    print(f"Average pair entropy per layer: {avg_pair_entropy:.3f}")
    
    # Theoretical maximum pairs for 8 experts with top-2 routing
    max_pairs = (num_experts * (num_experts - 1)) // 2
    print(f"Maximum possible pairs: {max_pairs}")
    print(f"Pair diversity: {avg_unique_pairs / max_pairs:.3f}")

def main():
    """Main analysis function"""
    trace_file = "routing_data/qwen15_moe_a27b_traces_small.pkl"
    
    # Load traces
    traces = load_traces(trace_file)
    
    # Analyze routing patterns
    layer_stats, num_experts = analyze_routing_patterns(traces)
    
    # Print statistics
    print_routing_statistics(layer_stats, num_experts)
    
    # Analyze prediction challenges
    analyze_prediction_challenges(layer_stats, num_experts)
    
    # Create visualizations
    visualize_routing_patterns(layer_stats, num_experts, "analysis/routing_patterns")
    
    # Save analysis results
    results = {
        'layer_stats': dict(layer_stats),
        'num_experts': num_experts,
        'num_layers': len(layer_stats),
        'total_traces': len(traces)
    }
    
    with open('analysis/routing_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nAnalysis complete! Results saved to analysis/routing_analysis_results.pkl")

if __name__ == "__main__":
    main()