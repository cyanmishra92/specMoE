#!/usr/bin/env python3
"""
Visualize expert selection traces to understand routing patterns
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random

def load_traces(pkl_path="routing_data/small_experimental_traces.pkl"):
    """Load traces from pickle file"""
    print(f"Loading traces from {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        traces = pickle.load(f)
    
    print(f"Loaded {len(traces)} traces")
    return traces

def extract_expert_sequences(traces, n_traces=10):
    """Extract expert selection sequences from random traces"""
    print(f"Extracting expert sequences from {n_traces} random traces...")
    
    # Group traces by layer for better visualization
    traces_by_layer = {}
    for trace in traces:
        layer_id = trace['layer_id']
        if layer_id not in traces_by_layer:
            traces_by_layer[layer_id] = []
        traces_by_layer[layer_id].append(trace)
    
    # Select random traces from each layer
    selected_traces = {}
    for layer_id in sorted(traces_by_layer.keys()):
        layer_traces = traces_by_layer[layer_id]
        # Select fewer traces per layer to get total n_traces
        n_per_layer = max(1, n_traces // len(traces_by_layer))
        selected = random.sample(layer_traces, min(n_per_layer, len(layer_traces)))
        selected_traces[layer_id] = selected
    
    # Extract expert sequences
    expert_sequences = {}
    for layer_id, layer_traces in selected_traces.items():
        expert_sequences[layer_id] = []
        for trace in layer_traces:
            target_routing = trace['target_routing']
            expert_sequence = np.argmax(target_routing, axis=1)
            expert_sequences[layer_id].append({
                'sequence': expert_sequence,
                'sample_id': trace['sample_id'],
                'sequence_length': len(expert_sequence)
            })
    
    return expert_sequences

def visualize_expert_traces(expert_sequences, output_dir="routing_data"):
    """Create visualizations of expert traces"""
    print("Creating expert trace visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = sns.color_palette("husl", 128)
    
    # Create subplot for each layer
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, layer_id in enumerate(sorted(expert_sequences.keys())):
        ax = axes[i]
        layer_traces = expert_sequences[layer_id]
        
        # Plot each trace
        for j, trace_data in enumerate(layer_traces):
            expert_seq = trace_data['sequence']
            positions = range(len(expert_seq))
            
            # Use different line styles for different traces
            linestyle = ['-', '--', '-.', ':'][j % 4]
            alpha = 0.7
            
            ax.plot(positions, expert_seq, 
                   linestyle=linestyle, 
                   alpha=alpha,
                   linewidth=2,
                   label=f"{trace_data['sample_id'][:15]}...")
        
        ax.set_title(f'Layer {layer_id} - Expert Selection Traces', fontsize=14)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Expert ID', fontsize=12)
        ax.set_ylim(-5, 133)
        ax.grid(True, alpha=0.3)
        
        # Add legend for first few traces
        if len(layer_traces) <= 4:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "expert_selection_traces.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved trace visualization: {output_dir}/expert_selection_traces.png")

def create_expert_transition_heatmap(expert_sequences, output_dir="routing_data"):
    """Create heatmap of expert transitions"""
    print("Creating expert transition heatmap...")
    
    output_dir = Path(output_dir)
    
    # Collect all transitions
    all_transitions = {}
    
    for layer_id, layer_traces in expert_sequences.items():
        layer_transitions = []
        
        for trace_data in layer_traces:
            expert_seq = trace_data['sequence']
            for i in range(len(expert_seq) - 1):
                from_expert = expert_seq[i]
                to_expert = expert_seq[i + 1]
                layer_transitions.append((from_expert, to_expert))
        
        all_transitions[layer_id] = layer_transitions
    
    # Create transition matrix for each layer
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    for i, layer_id in enumerate(sorted(all_transitions.keys())):
        ax = axes[i]
        transitions = all_transitions[layer_id]
        
        # Create transition matrix (limited to most common experts)
        transition_matrix = np.zeros((50, 50))  # Top 50 experts only
        
        # Get most common experts in this layer
        expert_counts = {}
        for from_exp, to_exp in transitions:
            expert_counts[from_exp] = expert_counts.get(from_exp, 0) + 1
            expert_counts[to_exp] = expert_counts.get(to_exp, 0) + 1
        
        top_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        expert_to_idx = {expert: idx for idx, (expert, _) in enumerate(top_experts)}
        
        # Fill transition matrix
        for from_exp, to_exp in transitions:
            if from_exp in expert_to_idx and to_exp in expert_to_idx:
                from_idx = expert_to_idx[from_exp]
                to_idx = expert_to_idx[to_exp]
                transition_matrix[from_idx, to_idx] += 1
        
        # Plot heatmap
        sns.heatmap(transition_matrix, 
                   cmap='viridis',
                   ax=ax,
                   cbar_kws={'label': 'Transition Count'},
                   xticklabels=False,
                   yticklabels=False)
        
        ax.set_title(f'Layer {layer_id} - Expert Transitions\n(Top 50 experts)', fontsize=12)
        ax.set_xlabel('To Expert', fontsize=10)
        ax.set_ylabel('From Expert', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "expert_transition_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved transition heatmap: {output_dir}/expert_transition_heatmap.png")

def create_expert_usage_timeline(expert_sequences, output_dir="routing_data"):
    """Create timeline visualization of expert usage"""
    print("Creating expert usage timeline...")
    
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, layer_id in enumerate(sorted(expert_sequences.keys())):
        ax = axes[i]
        layer_traces = expert_sequences[layer_id]
        
        # Create timeline data
        all_positions = []
        all_experts = []
        all_traces = []
        
        for trace_idx, trace_data in enumerate(layer_traces):
            expert_seq = trace_data['sequence']
            positions = range(len(expert_seq))
            
            all_positions.extend(positions)
            all_experts.extend(expert_seq)
            all_traces.extend([trace_idx] * len(expert_seq))
        
        # Create scatter plot
        scatter = ax.scatter(all_positions, all_experts, 
                           c=all_traces, 
                           cmap='tab10',
                           alpha=0.7,
                           s=30)
        
        ax.set_title(f'Layer {layer_id} - Expert Usage Timeline', fontsize=14)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Expert ID', fontsize=12)
        ax.set_ylim(-5, 133)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Trace ID', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "expert_usage_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved usage timeline: {output_dir}/expert_usage_timeline.png")

def analyze_expert_patterns(expert_sequences):
    """Analyze patterns in expert selection"""
    print("\n📊 Analyzing Expert Selection Patterns")
    print("=" * 50)
    
    for layer_id in sorted(expert_sequences.keys()):
        layer_traces = expert_sequences[layer_id]
        
        print(f"\nLayer {layer_id}:")
        
        # Collect all expert sequences for this layer
        all_experts = []
        all_transitions = []
        
        for trace_data in layer_traces:
            expert_seq = trace_data['sequence']
            all_experts.extend(expert_seq)
            
            # Get transitions
            for i in range(len(expert_seq) - 1):
                all_transitions.append((expert_seq[i], expert_seq[i + 1]))
        
        # Calculate statistics
        unique_experts = set(all_experts)
        expert_counts = {}
        for exp in all_experts:
            expert_counts[exp] = expert_counts.get(exp, 0) + 1
        
        # Most common experts
        top_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"  Active experts: {len(unique_experts)}/128")
        print(f"  Total expert selections: {len(all_experts)}")
        print(f"  Top 5 experts: {top_experts}")
        
        # Transition analysis
        unique_transitions = len(set(all_transitions))
        total_transitions = len(all_transitions)
        
        print(f"  Unique transitions: {unique_transitions}")
        print(f"  Total transitions: {total_transitions}")
        print(f"  Transition diversity: {unique_transitions/total_transitions:.2%}")
        
        # Calculate expert persistence (how often same expert is used consecutively)
        same_expert_count = sum(1 for from_exp, to_exp in all_transitions if from_exp == to_exp)
        persistence_rate = same_expert_count / total_transitions if total_transitions > 0 else 0
        
        print(f"  Expert persistence: {persistence_rate:.2%}")

def main():
    """Main visualization function"""
    print("🎨 Visualizing Expert Selection Traces")
    print("=" * 50)
    
    # Load traces
    traces = load_traces()
    
    # Extract expert sequences
    expert_sequences = extract_expert_sequences(traces, n_traces=12)
    
    # Create visualizations
    visualize_expert_traces(expert_sequences)
    create_expert_transition_heatmap(expert_sequences)
    create_expert_usage_timeline(expert_sequences)
    
    # Analyze patterns
    analyze_expert_patterns(expert_sequences)
    
    print(f"\n✅ Visualization complete!")
    print(f"Generated files:")
    print(f"  - expert_selection_traces.png: Line plots of expert sequences")
    print(f"  - expert_transition_heatmap.png: Heatmap of expert transitions")
    print(f"  - expert_usage_timeline.png: Timeline scatter plot")

if __name__ == "__main__":
    main()