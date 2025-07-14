#!/usr/bin/env python3
"""
Ensemble Model Analysis and Comparison
Analyze predictor contributions and layer-specific performance
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.ensemble_speculation_model import EnsembleSpeculationModel
from utils.data_processing import load_traces

def analyze_layer_entropy_correlation(model, traces, device):
    """Analyze correlation between layer entropy and predictor performance"""
    print("Analyzing layer entropy correlation...")
    
    model.eval()
    layer_results = defaultdict(lambda: {'predictions': [], 'targets': [], 'contributions': []})
    
    # Process traces layer by layer
    for trace in traces[:1000]:  # Sample for analysis
        layer_id = trace['layer_id']
        target_routing = trace['target_routing']
        expert_sequence = np.argmax(target_routing, axis=1)
        
        if len(expert_sequence) < 33:  # Need context + prediction
            continue
            
        # Create context and target
        context = torch.tensor(expert_sequence[-33:-1], dtype=torch.long).unsqueeze(0).to(device)
        target = torch.tensor(expert_sequence[-1], dtype=torch.long).unsqueeze(0).to(device)
        layer_tensor = torch.tensor(layer_id, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(context, layer_tensor)
            prediction = output['prediction']
            weights = output['weights']
            
            _, predicted = torch.max(prediction, 1)
            
            layer_results[layer_id]['predictions'].append(predicted.item())
            layer_results[layer_id]['targets'].append(target.item())
            layer_results[layer_id]['contributions'].append(weights.cpu().numpy())
    
    # Calculate accuracies and contributions per layer
    layer_analysis = {}
    for layer_id, results in layer_results.items():
        if len(results['predictions']) > 0:
            accuracy = np.mean(np.array(results['predictions']) == np.array(results['targets']))
            avg_contributions = np.mean(results['contributions'], axis=0)
            
            layer_analysis[layer_id] = {
                'accuracy': accuracy,
                'branch_contribution': avg_contributions[0],
                'pattern_contribution': avg_contributions[1],
                'transformer_contribution': avg_contributions[2],
                'entropy': model.layer_entropy.get(layer_id, 6.0),
                'samples': len(results['predictions'])
            }
    
    return layer_analysis

def create_ensemble_analysis_plots(layer_analysis, output_dir="routing_data"):
    """Create analysis plots for ensemble performance"""
    print("Creating ensemble analysis plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    layers = sorted(layer_analysis.keys())
    accuracies = [layer_analysis[layer]['accuracy'] for layer in layers]
    entropies = [layer_analysis[layer]['entropy'] for layer in layers]
    branch_contrib = [layer_analysis[layer]['branch_contribution'] for layer in layers]
    pattern_contrib = [layer_analysis[layer]['pattern_contribution'] for layer in layers]
    transformer_contrib = [layer_analysis[layer]['transformer_contribution'] for layer in layers]
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy vs Entropy
    axes[0, 0].scatter(entropies, accuracies, s=100, alpha=0.7, c='blue')
    for i, layer in enumerate(layers):
        axes[0, 0].annotate(f'L{layer}', (entropies[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[0, 0].set_xlabel('Layer Entropy (bits)')
    axes[0, 0].set_ylabel('Prediction Accuracy')
    axes[0, 0].set_title('Accuracy vs Layer Entropy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Predictor Contributions by Layer
    x = range(len(layers))
    width = 0.25
    
    axes[0, 1].bar([i - width for i in x], branch_contrib, width, label='Branch Predictor', alpha=0.8)
    axes[0, 1].bar(x, pattern_contrib, width, label='Pattern Predictor', alpha=0.8)
    axes[0, 1].bar([i + width for i in x], transformer_contrib, width, label='Transformer', alpha=0.8)
    
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Average Contribution Weight')
    axes[0, 1].set_title('Predictor Contributions by Layer')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'L{layer}' for layer in layers])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Entropy vs Predictor Preference
    axes[1, 0].scatter(entropies, branch_contrib, label='Branch', alpha=0.7, s=80)
    axes[1, 0].scatter(entropies, pattern_contrib, label='Pattern', alpha=0.7, s=80)
    axes[1, 0].scatter(entropies, transformer_contrib, label='Transformer', alpha=0.7, s=80)
    axes[1, 0].set_xlabel('Layer Entropy (bits)')
    axes[1, 0].set_ylabel('Predictor Weight')
    axes[1, 0].set_title('Predictor Preference vs Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Layer Performance Heatmap
    data_matrix = np.array([accuracies, branch_contrib, pattern_contrib, transformer_contrib])
    sns.heatmap(data_matrix, 
                xticklabels=[f'L{layer}' for layer in layers],
                yticklabels=['Accuracy', 'Branch', 'Pattern', 'Transformer'],
                annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 1])
    axes[1, 1].set_title('Layer Performance Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved ensemble analysis: {output_dir}/ensemble_analysis.png")

def compare_with_baseline(ensemble_results, baseline_results=None):
    """Compare ensemble with baseline performance"""
    print("Comparing ensemble with baseline...")
    
    if baseline_results is None:
        baseline_results = {
            1: 0.28, 3: 0.31, 5: 0.29, 7: 0.30, 9: 0.32, 11: 0.27
        }  # Approximate baseline from previous runs
    
    comparison = {}
    for layer_id, ensemble_data in ensemble_results.items():
        ensemble_acc = ensemble_data['accuracy']
        baseline_acc = baseline_results.get(layer_id, 0.30)
        
        comparison[layer_id] = {
            'ensemble_accuracy': ensemble_acc,
            'baseline_accuracy': baseline_acc,
            'improvement': ensemble_acc - baseline_acc,
            'relative_improvement': (ensemble_acc - baseline_acc) / baseline_acc if baseline_acc > 0 else 0
        }
    
    return comparison

def main():
    """Main analysis function"""
    print("🔍 Ensemble Model Analysis")
    print("=" * 50)
    
    # Load model (you would load the trained model here)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load traces
    traces = load_traces("routing_data/maximum_real_traces.pkl")
    if traces is None:
        raise ValueError("Could not load traces")
    
    # Create untrained model for structure analysis
    model = EnsembleSpeculationModel(
        num_experts=128,
        context_length=32,
        hidden_dim=512
    ).to(device)
    
    print(f"Model components:")
    print(f"  - Branch Predictor: {sum(p.numel() for p in model.branch_predictor.parameters()):,} params")
    print(f"  - Pattern Predictor: {sum(p.numel() for p in model.pattern_predictor.parameters()):,} params")
    print(f"  - Transformer: {sum(p.numel() for p in model.transformer_predictor.parameters()):,} params")
    print(f"  - Meta Network: {sum(p.numel() for p in model.meta_network.parameters()):,} params")
    
    # Analyze layer entropy correlation
    layer_analysis = analyze_layer_entropy_correlation(model, traces, device)
    
    # Create analysis plots
    create_ensemble_analysis_plots(layer_analysis)
    
    # Compare with baseline
    comparison = compare_with_baseline(layer_analysis)
    
    # Print results
    print("\n📊 Layer Analysis Results:")
    print("=" * 50)
    for layer_id in sorted(layer_analysis.keys()):
        data = layer_analysis[layer_id]
        print(f"\nLayer {layer_id}:")
        print(f"  Entropy: {data['entropy']:.2f} bits")
        print(f"  Accuracy: {data['accuracy']:.3f}")
        print(f"  Branch contribution: {data['branch_contribution']:.3f}")
        print(f"  Pattern contribution: {data['pattern_contribution']:.3f}")
        print(f"  Transformer contribution: {data['transformer_contribution']:.3f}")
        print(f"  Samples: {data['samples']}")
    
    print("\n🚀 Ensemble Design Insights:")
    print("=" * 50)
    print("1. Branch Predictor: Good for low-entropy layers with predictable patterns")
    print("2. Pattern Predictor: Effective for mid-entropy layers with attention patterns")
    print("3. Transformer: Best for high-entropy layers with complex dependencies")
    print("4. Meta Network: Dynamically weights predictors based on layer characteristics")
    
    # Save analysis results
    analysis_path = "routing_data/ensemble_analysis_results.json"
    with open(analysis_path, 'w') as f:
        json.dump({
            'layer_analysis': {str(k): v for k, v in layer_analysis.items()},
            'comparison': {str(k): v for k, v in comparison.items()}
        }, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to: {analysis_path}")

if __name__ == "__main__":
    main()