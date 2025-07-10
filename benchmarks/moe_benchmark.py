"""
Comprehensive MoE benchmarking suite for speculation and memory optimization
"""

import torch
import torch.nn.functional as F
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append('..')
from models.small_switch_transformer import create_small_switch_model
from gating.speculation_engine import create_speculation_engine, SpeculativeGatingWrapper, SpeculationMode
from memory.adaptive_memory_manager import create_memory_manager
from utils.device_profiler import profile_current_device


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs"""
    model_name: str = "small_switch"
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    speculation_modes: List[str] = None
    num_warmup: int = 5
    num_iterations: int = 20
    use_compression: bool = True
    collect_detailed_stats: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512]
        if self.speculation_modes is None:
            self.speculation_modes = ["none", "layer_minus_1", "multi_layer", "adaptive"]


@dataclass
class BenchmarkResults:
    """Results from a single benchmark run"""
    config: Dict
    inference_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    speculation_accuracy: float
    cache_hit_rate: float
    expert_load_time_ms: float
    tokens_per_second: float
    load_balancing_loss: float
    routing_entropy: float
    compression_ratio: float = 1.0
    expert_utilization: float = 0.0


class MoEBenchmark:
    """Comprehensive MoE benchmark suite"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_profile = profile_current_device()
        
        # Initialize model
        self.model = create_small_switch_model().to(self.device)
        self.model_info = self.model.get_model_info()
        
        # Create synthetic expert weights for memory manager testing
        self.expert_weights = self._create_synthetic_expert_weights()
        
        # Results storage
        self.results = []
        
        print(f"Initialized benchmark on {self.device_profile.device_name}")
        print(f"Model: {self.model_info['total_parameters']:,} parameters")
    
    def _create_synthetic_expert_weights(self) -> Dict[str, torch.Tensor]:
        """Create synthetic expert weights for memory management testing"""
        expert_weights = {}
        hidden_size = self.model_info['hidden_size']
        num_layers = self.model_info['num_layers']
        num_experts = self.model_info['num_experts_per_layer']
        
        for layer_id in range(num_layers):
            for expert_id in range(num_experts):
                # Create expert weights (fc1 + fc2)
                fc1_weight = torch.randn(hidden_size * 4, hidden_size)
                fc2_weight = torch.randn(hidden_size, hidden_size * 4)
                
                expert_key = f"layer_{layer_id}_expert_{expert_id}"
                expert_weights[expert_key] = torch.cat([fc1_weight.flatten(), fc2_weight.flatten()])
        
        return expert_weights
    
    def run_single_benchmark(
        self, 
        batch_size: int, 
        seq_length: int, 
        speculation_mode: str
    ) -> BenchmarkResults:
        """Run a single benchmark configuration"""
        print(f"Running benchmark: batch_size={batch_size}, seq_len={seq_length}, speculation={speculation_mode}")
        
        # Create speculation engine
        speculation_engine = create_speculation_engine(
            num_experts=self.model_info['num_experts_per_layer'],
            num_layers=self.model_info['num_layers'],
            mode=speculation_mode
        )
        
        # Create memory manager
        memory_manager = create_memory_manager(
            self.device_profile, 
            self.model, 
            self.expert_weights
        )
        
        # Wrap model with speculation
        wrapped_model = SpeculativeGatingWrapper(self.model, speculation_engine)
        
        # Create input
        input_ids = torch.randint(0, 32000, (batch_size, seq_length), device=self.device)
        
        # Warmup
        for _ in range(self.config.num_warmup):
            with torch.no_grad():
                _ = wrapped_model.forward(input_ids)
            torch.cuda.synchronize()
        
        # Clear memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        inference_times = []
        memory_usages = []
        
        for _ in range(self.config.num_iterations):
            start_memory = torch.cuda.memory_allocated()
            
            start_time = time.time()
            with torch.no_grad():
                outputs = wrapped_model.forward(input_ids)
            torch.cuda.synchronize()
            end_time = time.time()
            
            end_memory = torch.cuda.memory_allocated()
            
            inference_times.append((end_time - start_time) * 1000)  # ms
            memory_usages.append((end_memory - start_memory) / (1024**2))  # MB
        
        # Get statistics
        speculation_stats = speculation_engine.get_statistics()
        memory_stats = memory_manager.get_memory_stats()
        
        # Calculate metrics
        inference_time_ms = np.mean(inference_times)
        memory_usage_mb = np.mean(memory_usages)
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        
        tokens_per_second = (batch_size * seq_length) / (inference_time_ms / 1000)
        
        # Analyze routing behavior
        routing_entropies = []
        expert_usage = torch.zeros(self.model_info['num_experts_per_layer'])
        
        for layer_info in outputs['routing_info']:
            gate_scores = layer_info['gate_scores']
            entropy = -torch.sum(gate_scores * torch.log(gate_scores + 1e-8), dim=-1).mean()
            routing_entropies.append(entropy.item())
            
            # Track expert usage
            expert_usage += gate_scores.mean(dim=0).cpu()
        
        expert_usage /= len(outputs['routing_info'])
        expert_utilization = 1.0 - torch.std(expert_usage) / torch.mean(expert_usage)
        
        return BenchmarkResults(
            config={
                'batch_size': batch_size,
                'sequence_length': seq_length,
                'speculation_mode': speculation_mode
            },
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb,
            peak_memory_mb=peak_memory_mb,
            speculation_accuracy=speculation_stats.get('overall_accuracy', 0.0),
            cache_hit_rate=memory_stats['gpu_cache']['hit_rate'],
            expert_load_time_ms=memory_stats.get('avg_load_time_ms', 0.0),
            tokens_per_second=tokens_per_second,
            load_balancing_loss=outputs['load_balancing_loss'].item(),
            routing_entropy=np.mean(routing_entropies),
            compression_ratio=memory_stats.get('compression_ratio', 1.0),
            expert_utilization=expert_utilization.item()
        )
    
    def run_full_benchmark(self) -> List[BenchmarkResults]:
        """Run complete benchmark suite"""
        print("Starting full benchmark suite...")
        
        total_runs = len(self.config.batch_sizes) * len(self.config.sequence_lengths) * len(self.config.speculation_modes)
        current_run = 0
        
        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.sequence_lengths:
                for speculation_mode in self.config.speculation_modes:
                    current_run += 1
                    print(f"Progress: {current_run}/{total_runs}")
                    
                    try:
                        result = self.run_single_benchmark(batch_size, seq_length, speculation_mode)
                        self.results.append(result)
                    except Exception as e:
                        print(f"Benchmark failed: {e}")
                        continue
        
        return self.results
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON"""
        results_dict = []
        for result in self.results:
            result_dict = asdict(result)
            results_dict.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump({
                'device_profile': asdict(self.device_profile),
                'model_info': self.model_info,
                'benchmark_config': asdict(self.config),
                'results': results_dict
            }, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def analyze_results(self) -> Dict:
        """Analyze benchmark results and provide insights"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        analysis = {
            'performance_summary': {},
            'speculation_analysis': {},
            'memory_analysis': {},
            'optimization_recommendations': []
        }
        
        # Performance analysis
        baseline_results = [r for r in self.results if r.config['speculation_mode'] == 'none']
        speculative_results = [r for r in self.results if r.config['speculation_mode'] != 'none']
        
        if baseline_results and speculative_results:
            baseline_avg_time = np.mean([r.inference_time_ms for r in baseline_results])
            speculative_avg_time = np.mean([r.inference_time_ms for r in speculative_results])
            speedup = baseline_avg_time / speculative_avg_time
            
            analysis['performance_summary'] = {
                'baseline_avg_time_ms': baseline_avg_time,
                'speculative_avg_time_ms': speculative_avg_time,
                'average_speedup': speedup,
                'max_tokens_per_second': max(r.tokens_per_second for r in self.results)
            }
        
        # Speculation analysis
        spec_accuracies = [r.speculation_accuracy for r in speculative_results if r.speculation_accuracy > 0]
        if spec_accuracies:
            analysis['speculation_analysis'] = {
                'avg_accuracy': np.mean(spec_accuracies),
                'max_accuracy': max(spec_accuracies),
                'min_accuracy': min(spec_accuracies),
                'accuracy_std': np.std(spec_accuracies)
            }
        
        # Memory analysis
        memory_usages = [r.memory_usage_mb for r in self.results]
        cache_hit_rates = [r.cache_hit_rate for r in self.results if r.cache_hit_rate > 0]
        
        analysis['memory_analysis'] = {
            'avg_memory_usage_mb': np.mean(memory_usages),
            'max_memory_usage_mb': max(memory_usages),
            'avg_cache_hit_rate': np.mean(cache_hit_rates) if cache_hit_rates else 0.0,
            'compression_ratios': [r.compression_ratio for r in self.results if r.compression_ratio > 1.0]
        }
        
        # Optimization recommendations
        best_speculation_mode = max(speculative_results, key=lambda x: x.tokens_per_second).config['speculation_mode']
        analysis['optimization_recommendations'].append(
            f"Best speculation mode: {best_speculation_mode}"
        )
        
        if np.mean(cache_hit_rates) < 0.5:
            analysis['optimization_recommendations'].append(
                "Low cache hit rate - consider increasing cache size or improving prefetching"
            )
        
        avg_expert_util = np.mean([r.expert_utilization for r in self.results])
        if avg_expert_util < 0.7:
            analysis['optimization_recommendations'].append(
                "Low expert utilization - consider load balancing improvements"
            )
        
        return analysis
    
    def plot_results(self, save_dir: str = "benchmark_plots"):
        """Create visualization plots of benchmark results"""
        Path(save_dir).mkdir(exist_ok=True)
        
        if not self.results:
            print("No results to plot")
            return
        
        # 1. Performance comparison
        plt.figure(figsize=(12, 8))
        
        # Group by speculation mode
        modes = list(set(r.config['speculation_mode'] for r in self.results))
        batch_sizes = list(set(r.config['batch_size'] for r in self.results))
        
        for mode in modes:
            mode_results = [r for r in self.results if r.config['speculation_mode'] == mode]
            mode_batch_sizes = [r.config['batch_size'] for r in mode_results]
            mode_throughputs = [r.tokens_per_second for r in mode_results]
            
            plt.plot(mode_batch_sizes, mode_throughputs, marker='o', label=mode, linewidth=2)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Tokens per Second')
        plt.title('Throughput vs Batch Size by Speculation Mode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Memory usage analysis
        plt.figure(figsize=(10, 6))
        
        memory_data = []
        labels = []
        
        for mode in modes:
            mode_results = [r for r in self.results if r.config['speculation_mode'] == mode]
            mode_memory = [r.memory_usage_mb for r in mode_results]
            memory_data.append(mode_memory)
            labels.append(mode)
        
        plt.boxplot(memory_data, labels=labels)
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Distribution by Speculation Mode')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Speculation accuracy heatmap
        speculation_results = [r for r in self.results if r.speculation_accuracy > 0]
        if speculation_results:
            plt.figure(figsize=(10, 6))
            
            # Create heatmap data
            modes = sorted(set(r.config['speculation_mode'] for r in speculation_results))
            batch_sizes = sorted(set(r.config['batch_size'] for r in speculation_results))
            
            heatmap_data = np.zeros((len(modes), len(batch_sizes)))
            
            for i, mode in enumerate(modes):
                for j, batch_size in enumerate(batch_sizes):
                    matching_results = [
                        r for r in speculation_results 
                        if r.config['speculation_mode'] == mode and r.config['batch_size'] == batch_size
                    ]
                    if matching_results:
                        heatmap_data[i, j] = np.mean([r.speculation_accuracy for r in matching_results])
            
            sns.heatmap(heatmap_data, 
                       xticklabels=batch_sizes, 
                       yticklabels=modes,
                       annot=True, 
                       fmt='.3f',
                       cmap='viridis')
            
            plt.xlabel('Batch Size')
            plt.ylabel('Speculation Mode')
            plt.title('Speculation Accuracy Heatmap')
            plt.savefig(f"{save_dir}/speculation_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {save_dir}/")
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report"""
        analysis = self.analyze_results()
        
        report = f"""
# MoE Speculation Benchmark Report

## Device Information
- **Device**: {self.device_profile.device_name}
- **Memory**: {self.device_profile.memory_capacity_gb:.1f} GB
- **Bandwidth**: {self.device_profile.memory_bandwidth_gbps:.1f} GB/s

## Model Information
- **Total Parameters**: {self.model_info['total_parameters']:,}
- **Expert Parameters**: {self.model_info['expert_parameters']:,}
- **Layers**: {self.model_info['num_layers']}
- **Experts per Layer**: {self.model_info['num_experts_per_layer']}

## Performance Summary
"""
        
        if 'performance_summary' in analysis:
            perf = analysis['performance_summary']
            report += f"""
- **Average Speedup**: {perf.get('average_speedup', 0):.2f}x
- **Max Throughput**: {perf.get('max_tokens_per_second', 0):.0f} tokens/sec
- **Baseline Time**: {perf.get('baseline_avg_time_ms', 0):.1f} ms
- **Speculative Time**: {perf.get('speculative_avg_time_ms', 0):.1f} ms
"""
        
        if 'speculation_analysis' in analysis:
            spec = analysis['speculation_analysis']
            report += f"""
## Speculation Analysis
- **Average Accuracy**: {spec.get('avg_accuracy', 0):.3f}
- **Max Accuracy**: {spec.get('max_accuracy', 0):.3f}
- **Min Accuracy**: {spec.get('min_accuracy', 0):.3f}
"""
        
        if 'memory_analysis' in analysis:
            mem = analysis['memory_analysis']
            report += f"""
## Memory Analysis
- **Average Usage**: {mem.get('avg_memory_usage_mb', 0):.1f} MB
- **Peak Usage**: {mem.get('max_memory_usage_mb', 0):.1f} MB
- **Cache Hit Rate**: {mem.get('avg_cache_hit_rate', 0):.3f}
"""
        
        if analysis.get('optimization_recommendations'):
            report += "\n## Optimization Recommendations\n"
            for rec in analysis['optimization_recommendations']:
                report += f"- {rec}\n"
        
        return report


def run_quick_benchmark():
    """Run a quick benchmark for testing"""
    config = BenchmarkConfig(
        batch_sizes=[1, 2],
        sequence_lengths=[128, 256],
        speculation_modes=["none", "multi_layer"],
        num_iterations=5
    )
    
    benchmark = MoEBenchmark(config)
    results = benchmark.run_full_benchmark()
    
    # Save results
    benchmark.save_results("quick_benchmark_results.json")
    
    # Generate report
    report = benchmark.generate_report()
    print(report)
    
    # Create plots
    benchmark.plot_results("quick_benchmark_plots")
    
    return benchmark, results


if __name__ == "__main__":
    print("Running quick benchmark...")
    benchmark, results = run_quick_benchmark()
    print(f"Completed {len(results)} benchmark runs")