#!/usr/bin/env python3
"""
Analyze trace structure and reorganize statistics by layer
"""

import pickle
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_traces(pkl_path):
    """Load traces and analyze structure"""
    print(f"Loading traces from {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        traces = pickle.load(f)
    
    print(f"Loaded {len(traces)} traces")
    
    # Analyze structure
    sample_trace = traces[0]
    print(f"\nSample trace structure:")
    print(f"Keys: {list(sample_trace.keys())}")
    print(f"Layer ID: {sample_trace['layer_id']}")
    print(f"Sequence length: {sample_trace['sequence_length']}")
    print(f"Target routing shape: {sample_trace['target_routing'].shape}")
    print(f"Dataset: {sample_trace['dataset_name']}")
    
    # Find all layers
    layers = set()
    for trace in traces:
        layers.add(trace['layer_id'])
    
    print(f"\nFound layers: {sorted(layers)}")
    return traces, sorted(layers)

def create_3d_expert_statistics(traces, layers, num_experts=128):
    """Create 3D expert usage statistics: (layer, expert, frequency)"""
    print(f"\nCreating 3D expert statistics...")
    
    # Initialize 3D structure
    layer_expert_usage = defaultdict(lambda: defaultdict(int))
    
    # Process each trace
    for trace in traces:
        layer_id = trace['layer_id']
        target_routing = trace['target_routing']  # shape: (seq_len, num_experts)
        
        # Get top expert for each token
        top_experts = np.argmax(target_routing, axis=1)
        
        # Count expert usage for this layer
        for expert_idx in top_experts:
            layer_expert_usage[layer_id][expert_idx] += 1
    
    # Convert to proper 3D structure
    statistics_3d = {}
    for layer_id in layers:
        statistics_3d[layer_id] = {}
        for expert_id in range(num_experts):
            statistics_3d[layer_id][expert_id] = layer_expert_usage[layer_id][expert_id]
    
    return statistics_3d

def analyze_expert_diversity(statistics_3d, layers, num_experts=128):
    """Analyze expert diversity across layers"""
    print(f"\nAnalyzing expert diversity...")
    
    analysis = {}
    
    for layer_id in layers:
        layer_stats = statistics_3d[layer_id]
        
        # Count active experts
        active_experts = sum(1 for count in layer_stats.values() if count > 0)
        
        # Calculate entropy (diversity measure)
        total_usage = sum(layer_stats.values())
        if total_usage > 0:
            probabilities = [count / total_usage for count in layer_stats.values() if count > 0]
            entropy = -sum(p * np.log2(p) for p in probabilities)
        else:
            entropy = 0
        
        # Most used experts
        sorted_experts = sorted(layer_stats.items(), key=lambda x: x[1], reverse=True)
        top_5_experts = sorted_experts[:5]
        
        analysis[layer_id] = {
            'active_experts': active_experts,
            'diversity_percentage': (active_experts / num_experts) * 100,
            'entropy': entropy,
            'total_usage': total_usage,
            'top_5_experts': top_5_experts
        }
    
    return analysis

def save_restructured_statistics(statistics_3d, analysis, output_path):
    """Save restructured statistics"""
    print(f"\nSaving restructured statistics to {output_path}...")
    
    # Convert to serializable format
    serializable_stats = {}
    for layer_id, layer_data in statistics_3d.items():
        serializable_stats[str(layer_id)] = {str(k): v for k, v in layer_data.items()}
    
    # Add analysis
    serializable_analysis = {}
    for layer_id, layer_analysis in analysis.items():
        serializable_analysis[str(layer_id)] = {
            'active_experts': layer_analysis['active_experts'],
            'diversity_percentage': layer_analysis['diversity_percentage'],
            'entropy': layer_analysis['entropy'],
            'total_usage': layer_analysis['total_usage'],
            'top_5_experts': [(str(exp), count) for exp, count in layer_analysis['top_5_experts']]
        }
    
    output_data = {
        'format': '3D_layer_expert_frequency',
        'dimensions': ['layer', 'expert', 'frequency'],
        'num_experts': 128,
        'layers': list(statistics_3d.keys()),
        'expert_usage_by_layer': serializable_stats,
        'layer_analysis': serializable_analysis,
        'generation_time': str(np.datetime64('now'))
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved restructured statistics")

def create_layer_expert_heatmap(statistics_3d, layers, output_path):
    """Create heatmap visualization of expert usage across layers"""
    print(f"\nCreating layer-expert heatmap...")
    
    try:
        # Create matrix with smaller size to avoid memory issues
        matrix = np.zeros((len(layers), 128), dtype=np.float32)
        
        for i, layer_id in enumerate(layers):
            for expert_id in range(128):
                matrix[i, expert_id] = statistics_3d[layer_id][expert_id]
        
        # Log normalize to handle large values
        matrix = np.log1p(matrix)
        
        # Create heatmap with reduced DPI
        plt.figure(figsize=(16, 6))
        sns.heatmap(matrix, 
                    xticklabels=range(0, 128, 8),  # Reduced tick labels
                    yticklabels=[f'Layer {l}' for l in layers],
                    cmap='viridis',
                    cbar_kws={'label': 'Log(Expert Usage + 1)'})
        
        plt.title('Expert Usage Across Layers', fontsize=14)
        plt.xlabel('Expert ID', fontsize=11)
        plt.ylabel('Layer ID', fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved heatmap to {output_path}")
        
    except Exception as e:
        print(f"❌ Heatmap creation failed: {e}")
        print("Continuing without visualization...")

def validate_speculative_training(traces):
    """Validate the speculative training approach"""
    print(f"\nValidating speculative training approach...")
    
    # Check sequence structure
    sample_trace = traces[0]
    print(f"Sample routing shape: {sample_trace['target_routing'].shape}")
    print(f"Sample top_k shape: {sample_trace['target_top_k'].shape}")
    
    # Verify token-to-expert mapping
    target_routing = sample_trace['target_routing']
    target_top_k = sample_trace['target_top_k']
    
    # Check if top_k matches routing
    computed_top_k = np.argmax(target_routing, axis=1)
    matches = np.array_equal(computed_top_k, target_top_k.flatten())
    
    print(f"Top-k consistency: {matches}")
    
    # Check expert distribution in sequences
    seq_len = target_routing.shape[0]
    print(f"Sequence length: {seq_len}")
    print(f"Expert transitions in first 10 tokens: {computed_top_k[:10]}")
    
    # Calculate transition patterns
    transitions = []
    for i in range(1, len(computed_top_k)):
        if computed_top_k[i] != computed_top_k[i-1]:
            transitions.append((computed_top_k[i-1], computed_top_k[i]))
    
    print(f"Number of expert transitions: {len(transitions)}")
    print(f"Transition rate: {len(transitions) / seq_len:.2%}")
    
    return {
        'sequence_length': seq_len,
        'transition_rate': len(transitions) / seq_len,
        'top_k_consistent': matches,
        'sample_transitions': transitions[:5]
    }

def main():
    """Main analysis function"""
    print("🔍 Analyzing trace structure and reorganizing statistics...")
    
    # Load traces
    pkl_path = "routing_data/robust_traces.pkl"
    traces, layers = load_and_analyze_traces(pkl_path)
    
    # Create 3D statistics
    statistics_3d = create_3d_expert_statistics(traces, layers)
    
    # Analyze diversity
    analysis = analyze_expert_diversity(statistics_3d, layers)
    
    # Print analysis
    print(f"\n📊 Layer-wise Expert Analysis:")
    for layer_id in layers:
        layer_analysis = analysis[layer_id]
        print(f"\nLayer {layer_id}:")
        print(f"  Active experts: {layer_analysis['active_experts']}/128 ({layer_analysis['diversity_percentage']:.1f}%)")
        print(f"  Entropy: {layer_analysis['entropy']:.2f}")
        print(f"  Total usage: {layer_analysis['total_usage']}")
        print(f"  Top 3 experts: {layer_analysis['top_5_experts'][:3]}")
    
    # Save restructured statistics
    output_path = "routing_data/expert_statistics_3d.json"
    save_restructured_statistics(statistics_3d, analysis, output_path)
    
    # Create visualization
    heatmap_path = "routing_data/layer_expert_heatmap.png"
    create_layer_expert_heatmap(statistics_3d, layers, heatmap_path)
    
    # Validate training approach
    validation = validate_speculative_training(traces)
    print(f"\n🎯 Speculative Training Validation:")
    print(f"  Sequence length: {validation['sequence_length']}")
    print(f"  Expert transition rate: {validation['transition_rate']:.2%}")
    print(f"  Top-k consistency: {validation['top_k_consistent']}")
    
    print(f"\n✅ Analysis complete!")
    print(f"   - 3D statistics: {output_path}")
    print(f"   - Heatmap: {heatmap_path}")

if __name__ == "__main__":
    main()