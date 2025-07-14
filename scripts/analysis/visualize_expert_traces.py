#!/usr/bin/env python3
"""
Visualize expert selection traces to understand token routing patterns across layers
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from collections import defaultdict

def load_traces(pkl_path="routing_data/maximum_real_traces.pkl"):
    """Load traces from pickle file"""
    print(f"Loading traces from {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        traces = pickle.load(f)
    
    print(f"Loaded {len(traces)} traces")
    return traces

def extract_token_journeys(traces, n_tokens=10):
    """Extract token journeys across layers (1,3,5,7,9,11)"""
    print(f"Extracting token journeys for {n_tokens} random tokens...")
    
    # Group traces by sample_id and layer
    traces_by_sample = defaultdict(dict)
    for trace in traces:
        sample_id = trace['sample_id']
        layer_id = trace['layer_id']
        traces_by_sample[sample_id][layer_id] = trace
    
    # Find samples that have all required layers
    target_layers = [1, 3, 5, 7, 9, 11]
    complete_samples = []
    
    for sample_id, layer_traces in traces_by_sample.items():
        if all(layer in layer_traces for layer in target_layers):
            complete_samples.append(sample_id)
    
    print(f"Found {len(complete_samples)} complete samples with all layers")
    
    # Select random samples
    selected_samples = random.sample(complete_samples, min(n_tokens, len(complete_samples)))
    
    # Extract token journeys
    token_journeys = []
    for sample_id in selected_samples:
        layer_traces = traces_by_sample[sample_id]
        
        # Get sequence length (assume same for all layers)
        seq_length = len(layer_traces[1]['target_routing'])
        
        # For each token position, extract expert journey across layers
        for token_pos in range(min(seq_length, 20)):  # Limit to first 20 tokens
            journey = {}
            for layer_id in target_layers:
                target_routing = layer_traces[layer_id]['target_routing']
                expert_id = np.argmax(target_routing[token_pos])
                journey[layer_id] = expert_id
            
            token_journeys.append({
                'sample_id': sample_id,
                'token_position': token_pos,
                'journey': journey
            })
    
    return token_journeys

def visualize_token_journeys(token_journeys, output_dir="routing_data"):
    """Create visualization of token journeys across layers"""
    print("Creating token journey visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Select a subset of journeys for visualization
    selected_journeys = random.sample(token_journeys, min(50, len(token_journeys)))
    
    target_layers = [1, 3, 5, 7, 9, 11]
    layer_positions = {layer: i for i, layer in enumerate(target_layers)}
    
    # Plot each token journey
    for i, journey_data in enumerate(selected_journeys):
        journey = journey_data['journey']
        
        # Extract x and y coordinates
        x_coords = [layer_positions[layer] for layer in target_layers]
        y_coords = [journey[layer] for layer in target_layers]
        
        # Use different colors for different tokens
        color = plt.cm.tab20(i % 20)
        alpha = 0.7
        
        # Plot the journey
        ax.plot(x_coords, y_coords, 
               marker='o', 
               markersize=4,
               alpha=alpha,
               color=color,
               linewidth=1.5)
    
    # Customize the plot
    ax.set_xticks(range(len(target_layers)))
    ax.set_xticklabels([f'Layer {layer}' for layer in target_layers])
    ax.set_ylabel('Expert ID', fontsize=12)
    ax.set_title('Token Journeys Across MoE Layers', fontsize=14)
    ax.set_ylim(-5, 133)
    ax.grid(True, alpha=0.3)
    
    # Add some statistics
    ax.text(0.02, 0.98, f'Showing {len(selected_journeys)} token journeys', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "expert_selection_traces.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved token journey visualization: {output_dir}/expert_selection_traces.png")

def create_journey_transition_heatmap(token_journeys, output_dir="routing_data"):
    """Create heatmap showing transitions between layers"""
    print("Creating journey transition heatmap...")
    
    output_dir = Path(output_dir)
    
    target_layers = [1, 3, 5, 7, 9, 11]
    layer_pairs = [(target_layers[i], target_layers[i+1]) for i in range(len(target_layers)-1)]
    
    # Create subplots for each layer transition
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (from_layer, to_layer) in enumerate(layer_pairs):
        ax = axes[i]
        
        # Collect transitions between these layers
        transitions = []
        for journey_data in token_journeys:
            journey = journey_data['journey']
            from_expert = journey[from_layer]
            to_expert = journey[to_layer]
            transitions.append((from_expert, to_expert))
        
        # Create transition matrix (limited to most active experts)
        expert_counts = defaultdict(int)
        for from_exp, to_exp in transitions:
            expert_counts[from_exp] += 1
            expert_counts[to_exp] += 1
        
        # Get top 40 experts
        top_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:40]
        expert_to_idx = {expert: idx for idx, (expert, _) in enumerate(top_experts)}
        
        # Create transition matrix
        transition_matrix = np.zeros((40, 40))
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
        
        ax.set_title(f'Layer {from_layer} → Layer {to_layer}\nToken Transitions', fontsize=12)
        ax.set_xlabel(f'Expert in Layer {to_layer}', fontsize=10)
        ax.set_ylabel(f'Expert in Layer {from_layer}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "expert_transition_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved transition heatmap: {output_dir}/expert_transition_heatmap.png")

def analyze_journey_patterns(token_journeys):
    """Analyze patterns in token journeys"""
    print("\n📊 Analyzing Token Journey Patterns")
    print("=" * 50)
    
    target_layers = [1, 3, 5, 7, 9, 11]
    layer_pairs = [(target_layers[i], target_layers[i+1]) for i in range(len(target_layers)-1)]
    
    for from_layer, to_layer in layer_pairs:
        print(f"\nLayer {from_layer} → Layer {to_layer}:")
        
        # Collect transitions
        transitions = []
        expert_persistence = 0
        
        for journey_data in token_journeys:
            journey = journey_data['journey']
            from_expert = journey[from_layer]
            to_expert = journey[to_layer]
            transitions.append((from_expert, to_expert))
            
            if from_expert == to_expert:
                expert_persistence += 1
        
        # Calculate statistics
        unique_transitions = len(set(transitions))
        total_transitions = len(transitions)
        persistence_rate = expert_persistence / total_transitions if total_transitions > 0 else 0
        
        print(f"  Unique transitions: {unique_transitions}")
        print(f"  Total transitions: {total_transitions}")
        print(f"  Transition diversity: {unique_transitions/total_transitions:.2%}")
        print(f"  Expert persistence: {persistence_rate:.2%}")

def main():
    """Main visualization function"""
    print("🎨 Visualizing Token Journeys Across MoE Layers")
    print("=" * 50)
    
    # Load traces
    traces = load_traces()
    
    # Extract token journeys
    token_journeys = extract_token_journeys(traces, n_tokens=500)
    
    # Create visualizations
    visualize_token_journeys(token_journeys)
    create_journey_transition_heatmap(token_journeys)
    
    # Analyze patterns
    analyze_journey_patterns(token_journeys)
    
    print(f"\n✅ Visualization complete!")
    print(f"Generated files:")
    print(f"  - expert_selection_traces.png: Token journeys across layers")
    print(f"  - expert_transition_heatmap.png: Layer-to-layer transition heatmaps")

if __name__ == "__main__":
    main()