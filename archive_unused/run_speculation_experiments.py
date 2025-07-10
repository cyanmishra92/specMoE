#!/usr/bin/env python3
"""
Speculation Experiments Runner
Run comprehensive experiments to evaluate and improve speculation accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple

# Import our modules
from gating.speculation_engine import SpeculationEngine, SpeculationMode, create_speculation_engine
from evaluation.speculation_benchmark import SpeculationBenchmark, create_test_inputs
from models.small_switch_transformer import SmallSwitchTransformer
from models.pretrained_switch_model import PretrainedSwitchWrapper
from utils.device_profiler import DeviceProfiler

def run_baseline_experiments():
    """Run baseline experiments with current implementation"""
    
    print("🔬 Running Baseline Speculation Experiments")
    print("=" * 50)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create different speculation engines
    engines = {
        "layer_minus_1": create_speculation_engine(8, 6, "layer_minus_1"),
        "multi_layer": create_speculation_engine(8, 6, "multi_layer"),
        "pattern": create_speculation_engine(8, 6, "pattern"),
        "adaptive": create_speculation_engine(8, 6, "adaptive")
    }
    
    # Generate test data
    test_inputs = create_test_inputs(num_samples=50)
    
    # Run experiments for each engine
    all_results = {}
    
    for name, engine in engines.items():
        print(f"\n🧪 Testing {name} speculation...")
        
        benchmark = SpeculationBenchmark(None, engine, device)
        results = benchmark.run_comprehensive_benchmark(test_inputs)
        all_results[name] = results
        
        # Print quick summary
        acc_1 = results['accuracy']['mean_top_k_accuracy'][1]
        acc_2 = results['accuracy']['mean_top_k_accuracy'][2]
        confidence = results['calibration']['confidence_accuracy_correlation']
        
        print(f"  Top-1 Accuracy: {acc_1:.3f}")
        print(f"  Top-2 Accuracy: {acc_2:.3f}")
        print(f"  Confidence Correlation: {confidence:.3f}")
    
    # Save combined results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"baseline_experiments_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to: {results_file}")
    
    # Generate comparison plots
    create_comparison_plots(all_results)
    
    return all_results

def create_comparison_plots(results: Dict):
    """Create visualization plots comparing different approaches"""
    
    print("📈 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Speculation Engine Comparison', fontsize=16)
    
    # Extract data for plotting
    methods = list(results.keys())
    top1_accs = [results[m]['accuracy']['mean_top_k_accuracy'][1] for m in methods]
    top2_accs = [results[m]['accuracy']['mean_top_k_accuracy'][2] for m in methods]
    correlations = [results[m]['calibration']['confidence_accuracy_correlation'] for m in methods]
    kl_divs = [results[m]['accuracy']['mean_kl_divergence'] for m in methods]
    
    # Top-k Accuracy comparison
    axes[0, 0].bar(methods, top1_accs, alpha=0.7, label='Top-1')
    axes[0, 0].bar(methods, top2_accs, alpha=0.7, label='Top-2')
    axes[0, 0].set_title('Top-k Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Confidence correlation
    axes[0, 1].bar(methods, correlations, alpha=0.7, color='green')
    axes[0, 1].set_title('Confidence-Accuracy Correlation')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # KL Divergence (lower is better)
    axes[1, 0].bar(methods, kl_divs, alpha=0.7, color='red')
    axes[1, 0].set_title('KL Divergence (Lower is Better)')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Combined score (accuracy - kl_div)
    combined_scores = [acc - kl for acc, kl in zip(top1_accs, kl_divs)]
    axes[1, 1].bar(methods, combined_scores, alpha=0.7, color='purple')
    axes[1, 1].set_title('Combined Score (Accuracy - KL Div)')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('speculation_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comparison plots saved to: speculation_comparison.png")

def run_model_integration_experiments():
    """Run experiments with actual models"""
    
    print("\n🔗 Running Model Integration Experiments")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Test with custom model
        print("Testing with custom SmallSwitchTransformer...")
        custom_model = SmallSwitchTransformer(
            vocab_size=1000,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            num_experts=8
        ).to(device)
        
        # Create speculation engine
        engine = create_speculation_engine(8, 6, "adaptive")
        
        # Test with actual model forward pass
        test_input_ids = torch.randint(0, 1000, (2, 128)).to(device)
        
        with torch.no_grad():
            outputs = custom_model(test_input_ids)
        
        print(f"✅ Custom model test passed - Output shape: {outputs.shape}")
        
    except Exception as e:
        print(f"❌ Custom model test failed: {e}")
    
    try:
        # Test with pretrained model (if available)
        print("\nTesting with pretrained Switch Transformer...")
        pretrained_wrapper = PretrainedSwitchWrapper("google/switch-base-8")
        
        test_text = "The quick brown fox jumps over the lazy dog."
        outputs = pretrained_wrapper.generate_text(test_text, max_length=50)
        
        print(f"✅ Pretrained model test passed")
        print(f"Generated: {outputs[:100]}...")
        
    except Exception as e:
        print(f"❌ Pretrained model test failed: {e}")

def run_hardware_optimization_experiments():
    """Run experiments focused on hardware optimization"""
    
    print("\n⚡ Running Hardware Optimization Experiments")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping hardware experiments")
        return
    
    # Profile device
    profiler = DeviceProfiler()
    profile = profiler.profile_device()
    
    print(f"Device: {profile.device_name}")
    print(f"Memory: {profile.memory_gb:.1f} GB")
    print(f"Compute Capability: {profile.compute_capability}")
    
    # Test different batch sizes and sequence lengths
    batch_sizes = [1, 2, 4, 8]
    seq_lengths = [32, 64, 128, 256]
    
    optimization_results = {}
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print(f"Testing batch_size={batch_size}, seq_len={seq_len}")
            
            # Create test input
            test_input = torch.randn(batch_size, seq_len, 512).cuda()
            
            # Create engine
            engine = create_speculation_engine(8, 6, "multi_layer")
            
            # Time the speculation
            torch.cuda.synchronize()
            start_time = time.time()
            
            for layer_id in range(5):
                pred_probs, confidence = engine.predict_next_experts(layer_id, test_input)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Store results
            key = f"b{batch_size}_s{seq_len}"
            optimization_results[key] = {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'time_ms': (end_time - start_time) * 1000,
                'memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            }
            
            torch.cuda.empty_cache()
    
    # Find optimal settings
    best_throughput = max(optimization_results.values(), 
                         key=lambda x: (x['batch_size'] * x['seq_len']) / x['time_ms'])
    
    print(f"\n🏆 Best throughput configuration:")
    print(f"  Batch size: {best_throughput['batch_size']}")
    print(f"  Sequence length: {best_throughput['seq_len']}")
    print(f"  Time: {best_throughput['time_ms']:.2f} ms")
    print(f"  Memory: {best_throughput['memory_mb']:.1f} MB")
    
    return optimization_results

def analyze_speculation_patterns():
    """Analyze patterns in speculation accuracy"""
    
    print("\n🔍 Analyzing Speculation Patterns")
    print("=" * 50)
    
    # Create engine with pattern learning
    engine = create_speculation_engine(8, 6, "pattern")
    
    # Simulate realistic expert patterns
    patterns = {
        "sequential": [0, 1, 2, 3, 4, 5, 6, 7],  # Sequential expert usage
        "alternating": [0, 2, 1, 3, 0, 2, 1, 3],  # Alternating pattern
        "focused": [0, 0, 1, 0, 0, 1, 0, 0],      # Focused on few experts
        "random": np.random.randint(0, 8, 8).tolist()  # Random pattern
    }
    
    pattern_results = {}
    
    for pattern_name, expert_sequence in patterns.items():
        print(f"Testing {pattern_name} pattern...")
        
        # Reset engine
        engine.reset_statistics()
        
        # Simulate the pattern
        accuracies = []
        confidences = []
        
        for i in range(len(expert_sequence) - 1):
            # Create fake routing info
            current_expert = expert_sequence[i]
            next_expert = expert_sequence[i + 1]
            
            # Simulate gate scores favoring current expert
            gate_scores = torch.zeros(1, 8)
            gate_scores[0, current_expert] = 0.8
            gate_scores[0, :] = F.softmax(gate_scores[0, :], dim=0)
            
            routing_info = {
                'gate_scores': gate_scores,
                'top_k_indices': torch.tensor([[current_expert]])
            }
            
            # Update engine
            engine.update_routing_history(i, routing_info)
            
            # Get prediction
            hidden_states = torch.randn(1, 32, 512)
            pred_probs, confidence = engine.predict_next_experts(i, hidden_states)
            
            # Check if prediction matches actual next expert
            pred_expert = torch.argmax(pred_probs)
            accuracy = 1.0 if pred_expert == next_expert else 0.0
            
            accuracies.append(accuracy)
            confidences.append(confidence)
        
        pattern_results[pattern_name] = {
            'accuracy': np.mean(accuracies),
            'confidence': np.mean(confidences),
            'final_accuracy': engine.get_statistics()['overall_accuracy']
        }
        
        print(f"  Accuracy: {np.mean(accuracies):.3f}")
        print(f"  Confidence: {np.mean(confidences):.3f}")
    
    return pattern_results

def main():
    """Main experiment runner"""
    
    print("🚀 Starting Comprehensive Speculation Experiments")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    # Run all experiments
    results = {}
    
    # 1. Baseline experiments
    results['baseline'] = run_baseline_experiments()
    
    # 2. Model integration experiments
    run_model_integration_experiments()
    
    # 3. Hardware optimization experiments
    results['hardware'] = run_hardware_optimization_experiments()
    
    # 4. Pattern analysis
    results['patterns'] = analyze_speculation_patterns()
    
    # Save all results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_results_file = results_dir / f"comprehensive_experiments_{timestamp}.json"
    
    with open(final_results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 All experiment results saved to: {final_results_file}")
    
    # Generate final summary
    print("\n" + "=" * 60)
    print("🎯 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    if 'baseline' in results:
        best_method = max(results['baseline'].keys(), 
                         key=lambda x: results['baseline'][x]['accuracy']['mean_top_k_accuracy'][1])
        best_acc = results['baseline'][best_method]['accuracy']['mean_top_k_accuracy'][1]
        print(f"🏆 Best speculation method: {best_method} (Accuracy: {best_acc:.3f})")
    
    if 'patterns' in results:
        best_pattern = max(results['patterns'].keys(), 
                          key=lambda x: results['patterns'][x]['accuracy'])
        best_pattern_acc = results['patterns'][best_pattern]['accuracy']
        print(f"🔍 Best pattern type: {best_pattern} (Accuracy: {best_pattern_acc:.3f})")
    
    print("\n✅ All experiments completed successfully!")

if __name__ == "__main__":
    main()