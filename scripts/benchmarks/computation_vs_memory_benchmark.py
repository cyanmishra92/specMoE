#!/usr/bin/env python3
"""
Computation vs Memory Transfer Benchmark
Measures computation time vs memory transfer time for different batch sizes and expert counts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from scipy import stats

@dataclass
class BenchmarkConfig:
    """Configuration for computation vs memory benchmark"""
    model_dim: int = 768
    ff_dim: int = 3072
    num_experts: int = 128
    batch_sizes: List[int] = None
    expert_counts: List[int] = None
    num_trials: int = 100
    warmup_trials: int = 10
    seq_length: int = 512
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4]
        if self.expert_counts is None:
            self.expert_counts = [1, 10]

class MoEExpertLayer(nn.Module):
    """Simulates a single MoE expert for computation benchmarking"""
    
    def __init__(self, model_dim: int, ff_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        
        # Standard MoE expert architecture
        self.up_proj = nn.Linear(model_dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, model_dim, bias=False)
        self.gate_proj = nn.Linear(model_dim, ff_dim, bias=False)
        self.activation = nn.SiLU()  # Swish activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert"""
        # Gate and Up projections
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        
        # Element-wise multiplication
        hidden = gate * up
        
        # Down projection
        output = self.down_proj(hidden)
        
        return output

class MoELayer(nn.Module):
    """Simulates a complete MoE layer with multiple experts"""
    
    def __init__(self, model_dim: int, ff_dim: int, num_experts: int):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.num_experts = num_experts
        
        # Create experts
        self.experts = nn.ModuleList([
            MoEExpertLayer(model_dim, ff_dim) 
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(model_dim, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass with specified expert indices"""
        batch_size, seq_len, _ = x.shape
        
        # Get gating scores (for realism, not used in routing)
        gate_scores = F.softmax(self.gate(x), dim=-1)
        
        # Route to specified experts
        outputs = []
        for i in range(batch_size):
            # Get expert index for this batch item
            expert_idx = expert_indices[i].item()
            
            # Process through specified expert
            expert_output = self.experts[expert_idx](x[i:i+1])
            outputs.append(expert_output)
        
        # Combine outputs
        output = torch.cat(outputs, dim=0)
        
        return output

class ComputationMemoryBenchmark:
    """Benchmark computation vs memory transfer times"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Running benchmark on: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        # Initialize MoE layer
        self.moe_layer = MoELayer(
            config.model_dim, 
            config.ff_dim, 
            config.num_experts
        ).to(self.device)
        
        # Results storage
        self.results = {
            'computation_times': defaultdict(list),
            'memory_transfer_times': defaultdict(list),
            'total_times': defaultdict(list),
            'config': config.__dict__
        }
        
        # Calculate expert size
        self.expert_size_mb = self._calculate_expert_size()
        print(f"Expert size: {self.expert_size_mb:.2f} MB")
    
    def _calculate_expert_size(self) -> float:
        """Calculate expert size in MB"""
        expert = self.moe_layer.experts[0]
        total_params = sum(p.numel() for p in expert.parameters())
        bytes_size = total_params * 4  # float32 = 4 bytes
        return bytes_size / (1024 * 1024)
    
    def _create_expert_weights(self, expert_count: int) -> torch.Tensor:
        """Create dummy expert weights for memory transfer"""
        # Simulate expert weights as concatenated parameter tensors
        total_params = sum(p.numel() for p in self.moe_layer.experts[0].parameters())
        return torch.randn(expert_count, total_params, device='cpu', dtype=torch.float32)
    
    def _get_unique_experts(self, batch_size: int, expert_count: int) -> Tuple[List[int], int]:
        """Generate expert indices with deduplication"""
        all_experts = []
        
        # Each batch item requests expert_count experts
        for batch_idx in range(batch_size):
            # Generate expert_count random experts for this batch item
            batch_experts = np.random.choice(
                self.config.num_experts, 
                size=expert_count, 
                replace=False
            ).tolist()
            all_experts.extend(batch_experts)
        
        # Remove duplicates while preserving order
        unique_experts = []
        seen = set()
        for expert in all_experts:
            if expert not in seen:
                unique_experts.append(expert)
                seen.add(expert)
        
        return unique_experts, len(unique_experts)
    
    def benchmark_computation_time(self, batch_size: int, expert_count: int, 
                                 num_trials: int) -> List[float]:
        """Benchmark computation time for a single layer"""
        print(f"  Benchmarking computation: batch_size={batch_size}, expert_count={expert_count}")
        
        # Create input tensor
        input_tensor = torch.randn(
            batch_size, 
            self.config.seq_length, 
            self.config.model_dim,
            device=self.device
        )
        
        # Generate expert indices (one per batch item)
        expert_indices = torch.randint(
            0, self.config.num_experts, 
            (batch_size,), 
            device=self.device
        )
        
        # Warmup
        for _ in range(self.config.warmup_trials):
            with torch.no_grad():
                _ = self.moe_layer(input_tensor, expert_indices)
            torch.cuda.synchronize()
        
        # Benchmark
        computation_times = []
        for _ in range(num_trials):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = self.moe_layer(input_tensor, expert_indices)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            computation_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return computation_times
    
    def benchmark_memory_transfer_time(self, batch_size: int, expert_count: int, 
                                     num_trials: int) -> List[float]:
        """Benchmark memory transfer time for expert loading"""
        print(f"  Benchmarking memory transfer: batch_size={batch_size}, expert_count={expert_count}")
        
        # Get unique experts (with deduplication)
        unique_experts, actual_expert_count = self._get_unique_experts(batch_size, expert_count)
        
        # Create expert weights on CPU
        expert_weights = self._create_expert_weights(actual_expert_count)
        
        # Create GPU buffer
        gpu_buffer = torch.empty_like(expert_weights, device=self.device)
        
        # Warmup
        for _ in range(self.config.warmup_trials):
            gpu_buffer.copy_(expert_weights, non_blocking=True)
            torch.cuda.synchronize()
        
        # Benchmark
        transfer_times = []
        for _ in range(num_trials):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Simulate expert loading
            gpu_buffer.copy_(expert_weights, non_blocking=True)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            transfer_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return transfer_times
    
    def benchmark_total_time(self, batch_size: int, expert_count: int, 
                           num_trials: int) -> List[float]:
        """Benchmark total time (computation + memory transfer)"""
        print(f"  Benchmarking total time: batch_size={batch_size}, expert_count={expert_count}")
        
        # Get unique experts
        unique_experts, actual_expert_count = self._get_unique_experts(batch_size, expert_count)
        
        # Create input tensor
        input_tensor = torch.randn(
            batch_size, 
            self.config.seq_length, 
            self.config.model_dim,
            device=self.device
        )
        
        # Create expert weights and GPU buffer
        expert_weights = self._create_expert_weights(actual_expert_count)
        gpu_buffer = torch.empty_like(expert_weights, device=self.device)
        
        # Generate expert indices
        expert_indices = torch.randint(
            0, self.config.num_experts, 
            (batch_size,), 
            device=self.device
        )
        
        # Warmup
        for _ in range(self.config.warmup_trials):
            gpu_buffer.copy_(expert_weights, non_blocking=True)
            with torch.no_grad():
                _ = self.moe_layer(input_tensor, expert_indices)
            torch.cuda.synchronize()
        
        # Benchmark
        total_times = []
        for _ in range(num_trials):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Memory transfer
            gpu_buffer.copy_(expert_weights, non_blocking=True)
            
            # Computation
            with torch.no_grad():
                output = self.moe_layer(input_tensor, expert_indices)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            total_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return total_times
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark across all configurations"""
        print("=" * 60)
        print("COMPUTATION VS MEMORY TRANSFER BENCHMARK")
        print("=" * 60)
        
        total_configs = len(self.config.batch_sizes) * len(self.config.expert_counts)
        config_count = 0
        
        for batch_size in self.config.batch_sizes:
            for expert_count in self.config.expert_counts:
                config_count += 1
                print(f"\nConfiguration {config_count}/{total_configs}:")
                print(f"  Batch size: {batch_size}")
                print(f"  Expert count: {expert_count}")
                
                # Get unique expert count for this configuration
                _, actual_expert_count = self._get_unique_experts(batch_size, expert_count)
                print(f"  Actual unique experts: {actual_expert_count}")
                
                # Benchmark computation time
                comp_times = self.benchmark_computation_time(
                    batch_size, expert_count, self.config.num_trials
                )
                
                # Benchmark memory transfer time
                mem_times = self.benchmark_memory_transfer_time(
                    batch_size, expert_count, self.config.num_trials
                )
                
                # Benchmark total time
                total_times = self.benchmark_total_time(
                    batch_size, expert_count, self.config.num_trials
                )
                
                # Store results
                config_key = f"batch_{batch_size}_experts_{expert_count}"
                self.results['computation_times'][config_key] = comp_times
                self.results['memory_transfer_times'][config_key] = mem_times
                self.results['total_times'][config_key] = total_times
                
                # Print summary statistics
                print(f"  Computation time: {np.mean(comp_times):.2f}±{np.std(comp_times):.2f} ms")
                print(f"  Memory transfer time: {np.mean(mem_times):.2f}±{np.std(mem_times):.2f} ms")
                print(f"  Total time: {np.mean(total_times):.2f}±{np.std(total_times):.2f} ms")
                
                # Analysis
                comp_mean = np.mean(comp_times)
                mem_mean = np.mean(mem_times)
                if comp_mean > mem_mean:
                    print(f"  → Computation dominates ({comp_mean/mem_mean:.2f}× slower)")
                else:
                    print(f"  → Memory transfer dominates ({mem_mean/comp_mean:.2f}× slower)")
        
        return self.results
    
    def save_results(self, filename: str = "computation_vs_memory_results.json"):
        """Save benchmark results"""
        output_path = Path("benchmarks") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def create_box_plots(self):
        """Create box plots with error bounds"""
        print("\nCreating box plots...")
        
        # Prepare data for plotting
        plot_data = []
        
        for config_key in self.results['computation_times'].keys():
            batch_size = int(config_key.split('_')[1])
            expert_count = int(config_key.split('_')[3])
            
            # Computation times
            for time_val in self.results['computation_times'][config_key]:
                plot_data.append({
                    'Configuration': f'B{batch_size}_E{expert_count}',
                    'Batch Size': batch_size,
                    'Expert Count': expert_count,
                    'Time (ms)': time_val,
                    'Type': 'Computation'
                })
            
            # Memory transfer times
            for time_val in self.results['memory_transfer_times'][config_key]:
                plot_data.append({
                    'Configuration': f'B{batch_size}_E{expert_count}',
                    'Batch Size': batch_size,
                    'Expert Count': expert_count,
                    'Time (ms)': time_val,
                    'Type': 'Memory Transfer'
                })
            
            # Total times
            for time_val in self.results['total_times'][config_key]:
                plot_data.append({
                    'Configuration': f'B{batch_size}_E{expert_count}',
                    'Batch Size': batch_size,
                    'Expert Count': expert_count,
                    'Time (ms)': time_val,
                    'Type': 'Total'
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Box plot by configuration
        sns.boxplot(data=df, x='Configuration', y='Time (ms)', hue='Type', ax=axes[0, 0])
        axes[0, 0].set_title('Computation vs Memory Transfer by Configuration')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot by batch size
        sns.boxplot(data=df, x='Batch Size', y='Time (ms)', hue='Type', ax=axes[0, 1])
        axes[0, 1].set_title('Performance by Batch Size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box plot by expert count
        sns.boxplot(data=df, x='Expert Count', y='Time (ms)', hue='Type', ax=axes[1, 0])
        axes[1, 0].set_title('Performance by Expert Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Ratio analysis
        ratio_data = []
        for config_key in self.results['computation_times'].keys():
            batch_size = int(config_key.split('_')[1])
            expert_count = int(config_key.split('_')[3])
            
            comp_times = self.results['computation_times'][config_key]
            mem_times = self.results['memory_transfer_times'][config_key]
            
            # Calculate ratios
            for comp_time, mem_time in zip(comp_times, mem_times):
                ratio = comp_time / mem_time if mem_time > 0 else 0
                ratio_data.append({
                    'Configuration': f'B{batch_size}_E{expert_count}',
                    'Computation/Memory Ratio': ratio
                })
        
        ratio_df = pd.DataFrame(ratio_data)
        sns.boxplot(data=ratio_df, x='Configuration', y='Computation/Memory Ratio', ax=axes[1, 1])
        axes[1, 1].set_title('Computation vs Memory Transfer Ratio')
        axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal time')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmarks/computation_vs_memory_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_statistical_analysis(self):
        """Create detailed statistical analysis"""
        print("\nPerforming statistical analysis...")
        
        analysis_results = {}
        
        for config_key in self.results['computation_times'].keys():
            batch_size = int(config_key.split('_')[1])
            expert_count = int(config_key.split('_')[3])
            
            comp_times = np.array(self.results['computation_times'][config_key])
            mem_times = np.array(self.results['memory_transfer_times'][config_key])
            total_times = np.array(self.results['total_times'][config_key])
            
            # Statistical tests
            # T-test to compare computation vs memory transfer
            t_stat, p_value = stats.ttest_ind(comp_times, mem_times)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(comp_times) - 1) * np.var(comp_times) + 
                                 (len(mem_times) - 1) * np.var(mem_times)) / 
                                (len(comp_times) + len(mem_times) - 2))
            cohens_d = (np.mean(comp_times) - np.mean(mem_times)) / pooled_std
            
            # Confidence intervals
            comp_ci = stats.t.interval(0.95, len(comp_times)-1, 
                                      loc=np.mean(comp_times), 
                                      scale=stats.sem(comp_times))
            mem_ci = stats.t.interval(0.95, len(mem_times)-1, 
                                     loc=np.mean(mem_times), 
                                     scale=stats.sem(mem_times))
            
            analysis_results[config_key] = {
                'batch_size': int(batch_size),
                'expert_count': int(expert_count),
                'computation': {
                    'mean': float(np.mean(comp_times)),
                    'std': float(np.std(comp_times)),
                    'median': float(np.median(comp_times)),
                    'ci_lower': float(comp_ci[0]),
                    'ci_upper': float(comp_ci[1]),
                    'min': float(np.min(comp_times)),
                    'max': float(np.max(comp_times))
                },
                'memory_transfer': {
                    'mean': float(np.mean(mem_times)),
                    'std': float(np.std(mem_times)),
                    'median': float(np.median(mem_times)),
                    'ci_lower': float(mem_ci[0]),
                    'ci_upper': float(mem_ci[1]),
                    'min': float(np.min(mem_times)),
                    'max': float(np.max(mem_times))
                },
                'total': {
                    'mean': float(np.mean(total_times)),
                    'std': float(np.std(total_times)),
                    'median': float(np.median(total_times))
                },
                'statistical_tests': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'cohens_d': float(cohens_d),
                    'significant': bool(p_value < 0.05)
                },
                'dominance': 'computation' if np.mean(comp_times) > np.mean(mem_times) else 'memory_transfer',
                'dominance_ratio': float(max(np.mean(comp_times), np.mean(mem_times)) / min(np.mean(comp_times), np.mean(mem_times)))
            }
        
        # Save statistical analysis
        with open('benchmarks/statistical_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("=" * 60)
        
        for config_key, analysis in analysis_results.items():
            print(f"\nConfiguration: {config_key}")
            print(f"  Batch size: {analysis['batch_size']}, Expert count: {analysis['expert_count']}")
            print(f"  Computation: {analysis['computation']['mean']:.2f}±{analysis['computation']['std']:.2f} ms")
            print(f"  Memory transfer: {analysis['memory_transfer']['mean']:.2f}±{analysis['memory_transfer']['std']:.2f} ms")
            print(f"  Dominance: {analysis['dominance']} ({analysis['dominance_ratio']:.2f}× faster)")
            print(f"  Statistical significance: {'Yes' if analysis['statistical_tests']['significant'] else 'No'} (p={analysis['statistical_tests']['p_value']:.4f})")
            print(f"  Effect size (Cohen's d): {analysis['statistical_tests']['cohens_d']:.2f}")
        
        return analysis_results
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        print(f"Configuration:")
        print(f"  Model dimension: {self.config.model_dim}")
        print(f"  FF dimension: {self.config.ff_dim}")
        print(f"  Sequence length: {self.config.seq_length}")
        print(f"  Expert size: {self.expert_size_mb:.2f} MB")
        print(f"  Trials per configuration: {self.config.num_trials}")
        
        print(f"\nKey Findings:")
        
        # Analyze overall trends
        total_comp_times = []
        total_mem_times = []
        
        for config_key in self.results['computation_times'].keys():
            total_comp_times.extend(self.results['computation_times'][config_key])
            total_mem_times.extend(self.results['memory_transfer_times'][config_key])
        
        overall_comp_mean = np.mean(total_comp_times)
        overall_mem_mean = np.mean(total_mem_times)
        
        if overall_comp_mean > overall_mem_mean:
            print(f"  • Computation dominates overall ({overall_comp_mean/overall_mem_mean:.2f}× slower than memory)")
        else:
            print(f"  • Memory transfer dominates overall ({overall_mem_mean/overall_comp_mean:.2f}× slower than computation)")
        
        print(f"  • Overall computation time: {overall_comp_mean:.2f}±{np.std(total_comp_times):.2f} ms")
        print(f"  • Overall memory transfer time: {overall_mem_mean:.2f}±{np.std(total_mem_times):.2f} ms")
        
        # Batch size impact
        print(f"\nBatch Size Impact:")
        for batch_size in self.config.batch_sizes:
            batch_comp_times = []
            batch_mem_times = []
            
            for config_key in self.results['computation_times'].keys():
                if f'batch_{batch_size}_' in config_key:
                    batch_comp_times.extend(self.results['computation_times'][config_key])
                    batch_mem_times.extend(self.results['memory_transfer_times'][config_key])
            
            if batch_comp_times:
                comp_mean = np.mean(batch_comp_times)
                mem_mean = np.mean(batch_mem_times)
                print(f"  • Batch {batch_size}: Comp={comp_mean:.2f}ms, Mem={mem_mean:.2f}ms")
        
        # Expert count impact
        print(f"\nExpert Count Impact:")
        for expert_count in self.config.expert_counts:
            expert_comp_times = []
            expert_mem_times = []
            
            for config_key in self.results['computation_times'].keys():
                if f'_experts_{expert_count}' in config_key:
                    expert_comp_times.extend(self.results['computation_times'][config_key])
                    expert_mem_times.extend(self.results['memory_transfer_times'][config_key])
            
            if expert_comp_times:
                comp_mean = np.mean(expert_comp_times)
                mem_mean = np.mean(expert_mem_times)
                print(f"  • {expert_count} experts: Comp={comp_mean:.2f}ms, Mem={mem_mean:.2f}ms")

def main():
    """Run computation vs memory benchmark"""
    print("=" * 80)
    print("COMPUTATION VS MEMORY TRANSFER BENCHMARK")
    print("=" * 80)
    
    # Configuration
    config = BenchmarkConfig(
        model_dim=768,
        ff_dim=3072,
        num_experts=128,
        batch_sizes=[1, 2, 4],
        expert_counts=[1, 10],
        num_trials=100,
        seq_length=512
    )
    
    # Run benchmark
    benchmark = ComputationMemoryBenchmark(config)
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    benchmark.save_results()
    
    # Create visualizations
    benchmark.create_box_plots()
    
    # Statistical analysis
    benchmark.create_statistical_analysis()
    
    # Print summary
    benchmark.print_summary()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()