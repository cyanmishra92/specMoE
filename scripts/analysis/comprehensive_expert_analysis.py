#!/usr/bin/env python3
"""
Comprehensive expert analysis with detailed statistics per layer
"""

import pickle
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

def load_traces(pkl_path="routing_data/maximum_real_traces.pkl"):
    """Load traces from pickle file"""
    print(f"Loading traces from {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        traces = pickle.load(f)
    
    print(f"Loaded {len(traces)} traces")
    return traces

def calculate_expert_statistics(expert_usage_counts, layer_id):
    """Calculate comprehensive statistics for expert usage"""
    usage_values = np.array(list(expert_usage_counts.values()))
    
    # Basic statistics
    mean_usage = np.mean(usage_values)
    std_usage = np.std(usage_values)
    coefficient_of_variation = std_usage / mean_usage if mean_usage > 0 else 0
    
    # Entropy calculation
    total_usage = np.sum(usage_values)
    if total_usage > 0:
        probabilities = usage_values / total_usage
        # Remove zero probabilities for entropy calculation
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
    else:
        entropy = 0
    
    # Min/Max usage
    min_usage = np.min(usage_values)
    max_usage = np.max(usage_values)
    
    # Skewness and Kurtosis
    skewness = stats.skew(usage_values)
    kurtosis = stats.kurtosis(usage_values)
    
    # Additional useful metrics
    median_usage = np.median(usage_values)
    q25 = np.percentile(usage_values, 25)
    q75 = np.percentile(usage_values, 75)
    iqr = q75 - q25
    
    # Active experts (non-zero usage)
    active_experts = np.sum(usage_values > 0)
    expert_diversity = active_experts / len(usage_values)
    
    statistics = {
        'layer_id': layer_id,
        'mean': float(mean_usage),
        'std': float(std_usage),
        'coefficient_of_variation': float(coefficient_of_variation),
        'entropy_bits': float(entropy),
        'min_usage': int(min_usage),
        'max_usage': int(max_usage),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'median': float(median_usage),
        'q25': float(q25),
        'q75': float(q75),
        'iqr': float(iqr),
        'active_experts': int(active_experts),
        'expert_diversity': float(expert_diversity),
        'total_usage': int(total_usage),
        'zero_usage_experts': int(128 - active_experts)
    }
    
    return statistics

def analyze_layer_statistics(traces, num_experts=128):
    """Analyze statistics for each layer"""
    print("\n📊 Calculating comprehensive layer statistics...")
    
    # Group traces by layer and collect expert usage
    layer_expert_usage = defaultdict(lambda: defaultdict(int))
    
    for trace in traces:
        layer_id = trace['layer_id']
        target_routing = trace['target_routing']
        
        # Get top expert for each token
        top_experts = np.argmax(target_routing, axis=1)
        
        # Count expert usage for this layer
        for expert_idx in top_experts:
            layer_expert_usage[layer_id][expert_idx] += 1
    
    # Calculate statistics for each layer
    layer_statistics = {}
    
    for layer_id in sorted(layer_expert_usage.keys()):
        # Ensure all experts are represented (even with 0 usage)
        expert_counts = {}
        for expert_id in range(num_experts):
            expert_counts[expert_id] = layer_expert_usage[layer_id].get(expert_id, 0)
        
        # Calculate comprehensive statistics
        stats = calculate_expert_statistics(expert_counts, layer_id)
        layer_statistics[layer_id] = stats
        
        # Print layer summary
        print(f"\nLayer {layer_id}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Std: {stats['std']:.2f}")
        print(f"  CV: {stats['coefficient_of_variation']:.3f}")
        print(f"  Entropy: {stats['entropy_bits']:.2f} bits")
        print(f"  Min/Max: {stats['min_usage']}/{stats['max_usage']}")
        print(f"  Skewness: {stats['skewness']:.3f}")
        print(f"  Kurtosis: {stats['kurtosis']:.3f}")
        print(f"  Active experts: {stats['active_experts']}/128 ({stats['expert_diversity']:.1%})")
    
    return layer_statistics, layer_expert_usage

def create_expert_frequency_plots(layer_expert_usage, layer_statistics, output_dir="routing_data"):
    """Create frequency plots for each layer"""
    print("\n📈 Creating expert frequency plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create individual plots for each layer
    for layer_id in sorted(layer_expert_usage.keys()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        expert_counts = [layer_expert_usage[layer_id].get(i, 0) for i in range(128)]
        stats = layer_statistics[layer_id]
        
        # Plot 1: Bar chart of expert usage
        ax1.bar(range(128), expert_counts, alpha=0.7, color='skyblue')
        ax1.set_title(f'Layer {layer_id} - Expert Usage Frequency', fontsize=14)
        ax1.set_xlabel('Expert ID', fontsize=12)
        ax1.set_ylabel('Usage Count', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: {stats['mean']:.2f}
Std: {stats['std']:.2f}
CV: {stats['coefficient_of_variation']:.3f}
Entropy: {stats['entropy_bits']:.2f} bits
Active: {stats['active_experts']}/128
Skewness: {stats['skewness']:.3f}"""
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Histogram of usage distribution
        non_zero_counts = [c for c in expert_counts if c > 0]
        if non_zero_counts:
            ax2.hist(non_zero_counts, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title(f'Layer {layer_id} - Usage Distribution (Non-zero)', fontsize=14)
            ax2.set_xlabel('Usage Count', fontsize=12)
            ax2.set_ylabel('Number of Experts', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add mean line
            ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.2f}')
            ax2.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.2f}')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"layer_{layer_id}_expert_frequency.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved frequency plots for all layers")

def create_comparative_statistics_plot(layer_statistics, output_dir="routing_data"):
    """Create comparative plot of statistics across layers"""
    print("\n📊 Creating comparative statistics plot...")
    
    output_dir = Path(output_dir)
    
    # Extract data for plotting
    layers = sorted(layer_statistics.keys())
    metrics = ['mean', 'std', 'coefficient_of_variation', 'entropy_bits', 'skewness', 'kurtosis', 'expert_diversity']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [layer_statistics[layer][metric] for layer in layers]
        
        ax.bar(layers, values, alpha=0.7, color=f'C{i}')
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            ax.text(layers[j], v + 0.01 * max(values), f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    # Remove empty subplot
    axes[7].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparative_layer_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparative statistics plot")

def save_trace_paths_for_visualization(traces, output_dir="routing_data", n_samples=10):
    """Save sample trace paths for visualization"""
    print(f"\n💾 Saving {n_samples} sample trace paths...")
    
    output_dir = Path(output_dir)
    
    # Sample traces from different layers
    traces_by_layer = defaultdict(list)
    for trace in traces:
        traces_by_layer[trace['layer_id']].append(trace)
    
    sample_traces = {}
    for layer_id in sorted(traces_by_layer.keys()):
        layer_traces = traces_by_layer[layer_id]
        n_per_layer = min(n_samples // len(traces_by_layer), len(layer_traces))
        sampled = np.random.choice(layer_traces, n_per_layer, replace=False)
        
        sample_traces[layer_id] = []
        for trace in sampled:
            expert_sequence = np.argmax(trace['target_routing'], axis=1)
            sample_traces[layer_id].append({
                'sample_id': trace['sample_id'],
                'expert_sequence': expert_sequence.tolist(),
                'sequence_length': len(expert_sequence),
                'dataset_name': trace['dataset_name']
            })
    
    # Save to JSON
    with open(output_dir / "sample_trace_paths.json", 'w') as f:
        json.dump(sample_traces, f, indent=2)
    
    print(f"✅ Saved sample trace paths")

def save_comprehensive_statistics(layer_statistics, output_dir="routing_data"):
    """Save comprehensive statistics to JSON"""
    print("\n💾 Saving comprehensive statistics...")
    
    output_dir = Path(output_dir)
    
    # Create summary statistics
    summary = {
        'analysis_type': 'comprehensive_expert_statistics',
        'num_layers': len(layer_statistics),
        'num_experts': 128,
        'metrics_included': [
            'mean', 'std', 'coefficient_of_variation', 'entropy_bits',
            'min_usage', 'max_usage', 'skewness', 'kurtosis',
            'median', 'q25', 'q75', 'iqr', 'active_experts',
            'expert_diversity', 'total_usage', 'zero_usage_experts'
        ],
        'layer_statistics': layer_statistics
    }
    
    with open(output_dir / "comprehensive_expert_statistics.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved comprehensive statistics")

def cleanup_old_traces(routing_data_dir="routing_data"):
    """Clean up old trace files"""
    print("\n🧹 Cleaning up old trace files...")
    
    routing_dir = Path(routing_data_dir)
    
    # Files to clean up
    files_to_remove = [
        "robust_traces.pkl",
        "robust_traces.json",
        "realistic_traces.pkl",
        "realistic_predictions.pkl"
    ]
    
    removed_count = 0
    for file_name in files_to_remove:
        file_path = routing_dir / file_name
        if file_path.exists():
            print(f"  Removing {file_path}")
            file_path.unlink()
            removed_count += 1
    
    print(f"✅ Cleaned up {removed_count} old trace files")

def print_robust_trace_collection_instructions():
    """Print instructions for running robust trace collection"""
    print("\n📋 Instructions for Robust Trace Collection")
    print("=" * 60)
    print("""
To collect traces from all 170+ papers, run:

1. Basic collection (3000 traces):
   python scripts/collection/collect_robust_traces.py --traces 3000

2. Large collection (10000 traces):
   python scripts/collection/collect_robust_traces.py --traces 10000

3. Mixed mode (real + synthetic):
   python scripts/collection/collect_robust_traces.py --traces 5000 --mode mixed --real-ratio 0.7

4. Real data only:
   python scripts/collection/collect_robust_traces.py --traces 5000 --mode real

Options:
  --traces: Number of traces to collect
  --mode: real, synthetic, or mixed
  --real-ratio: For mixed mode, ratio of real vs synthetic
  --model: Specific Switch model (auto-selects by default)
  --output: Custom output path

After collection, run this script again to analyze the new traces:
  python scripts/analysis/comprehensive_expert_analysis.py
""")

def main():
    """Main analysis function"""
    print("🔍 Comprehensive Expert Analysis")
    print("=" * 50)
    
    # Load traces
    traces = load_traces()
    
    # Analyze layer statistics
    layer_statistics, layer_expert_usage = analyze_layer_statistics(traces)
    
    # Create visualizations
    create_expert_frequency_plots(layer_expert_usage, layer_statistics)
    create_comparative_statistics_plot(layer_statistics)
    
    # Save trace paths
    save_trace_paths_for_visualization(traces)
    
    # Save comprehensive statistics
    save_comprehensive_statistics(layer_statistics)
    
    # Clean up old traces
    cleanup_old_traces()
    
    # Print collection instructions
    print_robust_trace_collection_instructions()
    
    print(f"\n✅ Comprehensive analysis complete!")
    print(f"Generated files:")
    print(f"  - layer_X_expert_frequency.png: Frequency plots for each layer")
    print(f"  - comparative_layer_statistics.png: Cross-layer comparison")
    print(f"  - comprehensive_expert_statistics.json: All statistics")
    print(f"  - sample_trace_paths.json: Sample expert sequences")

if __name__ == "__main__":
    main()