"""
Main script for Enhanced Pre-gated MoE on RTX 3090
Integrates all components and provides a simple interface
"""

import torch
import argparse
import json
from pathlib import Path
import sys

from models.small_switch_transformer import create_small_switch_model
from gating.speculation_engine import create_speculation_engine, SpeculativeGatingWrapper
from memory.adaptive_memory_manager import create_memory_manager
from utils.device_profiler import profile_current_device
from benchmarks.moe_benchmark import MoEBenchmark, BenchmarkConfig


def setup_model_and_systems(args):
    """Set up model and all supporting systems"""
    print("Setting up Enhanced Pre-gated MoE system...")
    
    # Device profiling
    device_profile = profile_current_device()
    print(f"Device: {device_profile.device_name} ({device_profile.memory_capacity_gb:.1f} GB)")
    
    # Create model
    model = create_small_switch_model().cuda()
    model_info = model.get_model_info()
    print(f"Model: {model_info['total_parameters']:,} parameters, {model_info['num_experts_per_layer']} experts/layer")
    
    # Create speculation engine
    speculation_engine = create_speculation_engine(
        num_experts=model_info['num_experts_per_layer'],
        num_layers=model_info['num_layers'],
        mode=args.speculation_mode
    )
    print(f"Speculation mode: {args.speculation_mode}")
    
    # Create synthetic expert weights for memory manager
    expert_weights = {}
    hidden_size = model_info['hidden_size']
    
    for layer_id in range(model_info['num_layers']):
        for expert_id in range(model_info['num_experts_per_layer']):
            # Create realistic expert weights
            fc1_weight = torch.randn(hidden_size * 4, hidden_size)
            fc2_weight = torch.randn(hidden_size, hidden_size * 4)
            expert_key = f"layer_{layer_id}_expert_{expert_id}"
            expert_weights[expert_key] = torch.cat([fc1_weight.flatten(), fc2_weight.flatten()])
    
    # Create memory manager
    memory_manager = create_memory_manager(device_profile, model, expert_weights)
    print(f"Memory strategy: {memory_manager.buffer_strategy.value}")
    print(f"Compression: {memory_manager.compression_type.value}")
    
    # Wrap model with speculation
    enhanced_model = SpeculativeGatingWrapper(model, speculation_engine)
    
    return enhanced_model, memory_manager, device_profile, model_info


def run_inference_demo(enhanced_model, args):
    """Run a simple inference demonstration"""
    print("\nRunning inference demonstration...")
    
    # Create sample input
    batch_size = args.batch_size
    seq_length = args.sequence_length
    input_ids = torch.randint(0, 32000, (batch_size, seq_length)).cuda()
    
    print(f"Input shape: {input_ids.shape}")
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = enhanced_model.forward(input_ids)
        torch.cuda.synchronize()
    
    # Benchmark inference
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    with torch.no_grad():
        outputs = enhanced_model.forward(input_ids)
    end_time.record()
    torch.cuda.synchronize()
    
    inference_time = start_time.elapsed_time(end_time)
    tokens_per_second = (batch_size * seq_length) / (inference_time / 1000)
    
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Throughput: {tokens_per_second:.0f} tokens/sec")
    print(f"Load balancing loss: {outputs['load_balancing_loss'].item():.4f}")
    
    # Show speculation statistics
    if 'speculation_stats' in outputs:
        stats = outputs['speculation_stats']
        print(f"Speculation accuracy: {stats['overall_accuracy']:.3f}")
        print(f"Total predictions: {stats['total_predictions']}")
    
    return outputs


def run_comprehensive_benchmark(args):
    """Run comprehensive benchmarking"""
    print("\nRunning comprehensive benchmark...")
    
    config = BenchmarkConfig(
        batch_sizes=args.benchmark_batch_sizes,
        sequence_lengths=args.benchmark_seq_lengths,
        speculation_modes=args.benchmark_modes,
        num_iterations=args.benchmark_iterations
    )
    
    benchmark = MoEBenchmark(config)
    results = benchmark.run_full_benchmark()
    
    # Save results
    results_file = f"benchmark_results_{args.speculation_mode}.json"
    benchmark.save_results(results_file)
    
    # Generate report
    report = benchmark.generate_report()
    print(report)
    
    # Save report
    report_file = f"benchmark_report_{args.speculation_mode}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Create plots
    plot_dir = f"benchmark_plots_{args.speculation_mode}"
    benchmark.plot_results(plot_dir)
    
    print(f"Benchmark complete. Results saved to {results_file}")
    print(f"Report saved to {report_file}")
    print(f"Plots saved to {plot_dir}/")
    
    return benchmark, results


def compare_speculation_modes():
    """Compare different speculation modes"""
    print("\nComparing speculation modes...")
    
    modes = ["none", "layer_minus_1", "multi_layer", "adaptive"]
    results = {}
    
    for mode in modes:
        print(f"\nTesting {mode} mode...")
        
        # Setup
        enhanced_model, memory_manager, device_profile, model_info = setup_model_and_systems(
            type('Args', (), {'speculation_mode': mode})()
        )
        
        # Quick test
        input_ids = torch.randint(0, 32000, (2, 128)).cuda()
        
        # Warmup and benchmark
        for _ in range(3):
            with torch.no_grad():
                _ = enhanced_model.forward(input_ids)
        
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            outputs = enhanced_model.forward(input_ids)
        end_time.record()
        torch.cuda.synchronize()
        
        inference_time = start_time.elapsed_time(end_time)
        tokens_per_second = (2 * 128) / (inference_time / 1000)
        
        results[mode] = {
            'inference_time_ms': inference_time,
            'tokens_per_second': tokens_per_second,
            'load_balancing_loss': outputs['load_balancing_loss'].item()
        }
        
        if 'speculation_stats' in outputs:
            results[mode]['speculation_accuracy'] = outputs['speculation_stats']['overall_accuracy']
        
        memory_stats = memory_manager.get_memory_stats()
        results[mode]['cache_hit_rate'] = memory_stats['gpu_cache']['hit_rate']
        results[mode]['compression_ratio'] = memory_stats.get('compression_ratio', 1.0)
    
    # Print comparison
    print("\n" + "="*80)
    print("SPECULATION MODE COMPARISON")
    print("="*80)
    print(f"{'Mode':<15} {'Time (ms)':<12} {'Tokens/sec':<12} {'Accuracy':<10} {'Cache Hit':<10}")
    print("-"*80)
    
    for mode, data in results.items():
        accuracy = data.get('speculation_accuracy', 0.0)
        print(f"{mode:<15} {data['inference_time_ms']:<12.1f} {data['tokens_per_second']:<12.0f} {accuracy:<10.3f} {data['cache_hit_rate']:<10.3f}")
    
    # Find best mode
    best_mode = max(results.keys(), key=lambda x: results[x]['tokens_per_second'])
    speedup = results[best_mode]['tokens_per_second'] / results['none']['tokens_per_second']
    
    print("-"*80)
    print(f"Best mode: {best_mode} ({speedup:.2f}x speedup over baseline)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Enhanced Pre-gated MoE for RTX 3090")
    parser.add_argument('--mode', choices=['demo', 'benchmark', 'compare'], default='demo',
                       help='Run mode: demo, benchmark, or compare')
    
    # Model and inference parameters
    parser.add_argument('--speculation-mode', default='multi_layer',
                       choices=['none', 'layer_minus_1', 'multi_layer', 'pattern', 'adaptive'],
                       help='Speculation mode to use')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for inference')
    parser.add_argument('--sequence-length', type=int, default=128,
                       help='Sequence length for inference')
    
    # Benchmark parameters
    parser.add_argument('--benchmark-batch-sizes', nargs='+', type=int, default=[1, 2, 4],
                       help='Batch sizes for benchmarking')
    parser.add_argument('--benchmark-seq-lengths', nargs='+', type=int, default=[128, 256],
                       help='Sequence lengths for benchmarking')
    parser.add_argument('--benchmark-modes', nargs='+', default=['none', 'multi_layer', 'adaptive'],
                       help='Speculation modes for benchmarking')
    parser.add_argument('--benchmark-iterations', type=int, default=10,
                       help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    print("Enhanced Pre-gated MoE for RTX 3090")
    print("=" * 50)
    
    if args.mode == 'demo':
        enhanced_model, memory_manager, device_profile, model_info = setup_model_and_systems(args)
        outputs = run_inference_demo(enhanced_model, args)
        
        # Show memory statistics
        memory_stats = memory_manager.get_memory_stats()
        print(f"\nMemory Statistics:")
        print(f"GPU cache hit rate: {memory_stats['gpu_cache']['hit_rate']:.3f}")
        print(f"Average load time: {memory_stats['avg_load_time_ms']:.2f} ms")
        if memory_stats.get('compression_ratio', 1.0) > 1.0:
            print(f"Compression ratio: {memory_stats['compression_ratio']:.1f}x")
    
    elif args.mode == 'benchmark':
        benchmark, results = run_comprehensive_benchmark(args)
        
    elif args.mode == 'compare':
        comparison_results = compare_speculation_modes()
        
        # Save comparison results
        with open('speculation_mode_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print("\nComparison results saved to 'speculation_mode_comparison.json'")
    
    print("\nDone!")


if __name__ == "__main__":
    main()