#!/usr/bin/env python3
"""
Memory vs Compute Latency Analysis with Box-Whisker Plots
Creates publication-quality plots showing latency distributions for different configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("pastel")

@dataclass
class BenchmarkConfig:
    """Configuration for latency benchmarking"""
    model_dim: int = 768
    ff_dim: int = 3072
    num_experts: int = 128
    batch_sizes: List[int] = None
    expert_counts: List[int] = None
    num_trials: int = 200  # Increased for better distribution
    warmup_trials: int = 20
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
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
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
        
        # Route to specified experts
        outputs = []
        for i in range(batch_size):
            expert_idx = expert_indices[i].item()
            expert_output = self.experts[expert_idx](x[i:i+1])
            outputs.append(expert_output)
        
        output = torch.cat(outputs, dim=0)
        return output

class LatencyBenchmark:
    """Comprehensive latency benchmark with detailed statistics"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Running latency benchmark on: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Initialize MoE layer
        self.moe_layer = MoELayer(
            config.model_dim, 
            config.ff_dim, 
            config.num_experts
        ).to(self.device)
        
        # Calculate expert size
        self.expert_size_mb = self._calculate_expert_size()
        print(f"Expert size: {self.expert_size_mb:.2f} MB")
        
        # Results storage
        self.results = {
            'computation': {},
            'memory_transfer': {},
            'metadata': {
                'expert_size_mb': self.expert_size_mb,
                'device': str(self.device),
                'num_trials': config.num_trials
            }
        }
    
    def _calculate_expert_size(self) -> float:
        """Calculate expert size in MB"""
        expert = self.moe_layer.experts[0]
        total_params = sum(p.numel() for p in expert.parameters())
        bytes_size = total_params * 4  # float32 = 4 bytes
        return bytes_size / (1024 * 1024)
    
    def _create_expert_weights(self, expert_count: int) -> torch.Tensor:
        """Create dummy expert weights for memory transfer"""
        total_params = sum(p.numel() for p in self.moe_layer.experts[0].parameters())
        return torch.randn(expert_count, total_params, device='cpu', dtype=torch.float32)
    
    def _get_unique_experts(self, batch_size: int, expert_count: int) -> Tuple[List[int], int]:
        """Generate expert indices with deduplication"""
        all_experts = []
        
        for batch_idx in range(batch_size):
            batch_experts = np.random.choice(
                self.config.num_experts, 
                size=expert_count, 
                replace=False
            ).tolist()
            all_experts.extend(batch_experts)
        
        # Remove duplicates
        unique_experts = []
        seen = set()
        for expert in all_experts:
            if expert not in seen:
                unique_experts.append(expert)
                seen.add(expert)
        
        return unique_experts, len(unique_experts)
    
    def benchmark_computation_latency(self, batch_size: int, expert_count: int) -> List[float]:
        """Benchmark computation latency with statistical rigor"""
        print(f"  Computing: BS={batch_size}, E={expert_count}")
        
        # Create input tensor
        input_tensor = torch.randn(
            batch_size, 
            self.config.seq_length, 
            self.config.model_dim,
            device=self.device
        )
        
        # Generate expert indices
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
        
        # Benchmark with high precision timing
        computation_times = []
        for _ in range(self.config.num_trials):
            torch.cuda.synchronize()
            
            # Use multiple measurements for higher precision
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                output = self.moe_layer(input_tensor, expert_indices)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # Returns milliseconds
            computation_times.append(elapsed_time)
        
        return computation_times
    
    def benchmark_memory_transfer_latency(self, batch_size: int, expert_count: int) -> List[float]:
        """Benchmark memory transfer latency with statistical rigor"""
        print(f"  Memory Transfer: BS={batch_size}, E={expert_count}")
        
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
        
        # Benchmark with high precision timing
        transfer_times = []
        for _ in range(self.config.num_trials):
            torch.cuda.synchronize()
            
            # Use CUDA events for precise timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            gpu_buffer.copy_(expert_weights, non_blocking=True)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # Returns milliseconds
            transfer_times.append(elapsed_time)
        
        return transfer_times
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive latency benchmark"""
        print("=" * 60)
        print("COMPREHENSIVE LATENCY BENCHMARK")
        print("=" * 60)
        
        total_configs = len(self.config.batch_sizes) * len(self.config.expert_counts)
        config_count = 0
        
        for batch_size in self.config.batch_sizes:
            for expert_count in self.config.expert_counts:
                config_count += 1
                print(f"\nConfiguration {config_count}/{total_configs}:")
                print(f"  Batch size: {batch_size}, Expert count: {expert_count}")
                
                # Benchmark computation
                comp_times = self.benchmark_computation_latency(batch_size, expert_count)
                
                # Benchmark memory transfer
                mem_times = self.benchmark_memory_transfer_latency(batch_size, expert_count)
                
                # Store results
                config_key = f"BS{batch_size}_E{expert_count}"
                self.results['computation'][config_key] = comp_times
                self.results['memory_transfer'][config_key] = mem_times
                
                # Print summary statistics
                print(f"    Computation: {np.mean(comp_times):.3f}±{np.std(comp_times):.3f} ms")
                print(f"    Memory Transfer: {np.mean(mem_times):.3f}±{np.std(mem_times):.3f} ms")
                print(f"    Ratio (M/C): {np.mean(mem_times)/np.mean(comp_times):.2f}×")
        
        return self.results
    
    def create_publication_plots(self, output_dir: str = "plots"):
        """Create publication-quality box-whisker plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nCreating publication plots in: {output_path}")
        
        # Prepare data for plotting
        plot_data = []
        
        for config_key in self.results['computation'].keys():
            # Parse configuration
            parts = config_key.split('_')
            batch_size = int(parts[0][2:])  # Remove 'BS' prefix
            expert_count = int(parts[1][1:])  # Remove 'E' prefix
            
            # Add computation data
            for time_val in self.results['computation'][config_key]:
                plot_data.append({
                    'Configuration': f'BS{batch_size}_E{expert_count}',
                    'Batch Size': batch_size,
                    'Expert Count': expert_count,
                    'Latency (ms)': time_val,
                    'Type': 'Computation'
                })
            
            # Add memory transfer data
            for time_val in self.results['memory_transfer'][config_key]:
                plot_data.append({
                    'Configuration': f'BS{batch_size}_E{expert_count}',
                    'Batch Size': batch_size,
                    'Expert Count': expert_count,
                    'Latency (ms)': time_val,
                    'Type': 'Memory Transfer'
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create individual plots
        self._create_main_comparison_plot(df, output_path)
        self._create_batch_size_analysis(df, output_path)
        self._create_expert_count_analysis(df, output_path)
        self._create_ratio_analysis(df, output_path)
        
        print("All plots saved successfully!")
    
    def _create_main_comparison_plot(self, df: pd.DataFrame, output_path: Path):
        """Create main comparison plot"""
        plt.figure(figsize=(12, 8))
        
        # Use pastel colors
        colors = ['#FFB3BA', '#BAFFC9']  # Light pink, light green
        
        # Create box plot
        box_plot = sns.boxplot(
            data=df, 
            x='Configuration', 
            y='Latency (ms)', 
            hue='Type',
            palette=colors,
            width=0.6,
            linewidth=1.5
        )
        
        # Customize plot
        plt.title('Memory Transfer vs Computation Latency by Configuration', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Configuration (Batch Size, Expert Count)', fontsize=14, fontweight='bold')
        plt.ylabel('Latency (milliseconds)', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Customize legend
        plt.legend(title='Operation Type', fontsize=12, title_fontsize=13, 
                  frameon=True, fancybox=True, shadow=True)
        
        # Add statistical annotations
        configs = df['Configuration'].unique()
        for i, config in enumerate(configs):
            comp_data = df[(df['Configuration'] == config) & (df['Type'] == 'Computation')]['Latency (ms)']
            mem_data = df[(df['Configuration'] == config) & (df['Type'] == 'Memory Transfer')]['Latency (ms)']
            
            ratio = np.mean(mem_data) / np.mean(comp_data)
            plt.text(i, max(mem_data) * 1.1, f'{ratio:.1f}×', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save as PDF
        plt.savefig(output_path / 'memory_vs_computation_latency.pdf', 
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        print("  ✓ Main comparison plot saved")
    
    def _create_batch_size_analysis(self, df: pd.DataFrame, output_path: Path):
        """Create batch size analysis plot"""
        plt.figure(figsize=(10, 6))
        
        # Filter for expert count = 10 (more interesting case)
        df_filtered = df[df['Expert Count'] == 10]
        
        colors = ['#FFB3BA', '#BAFFC9']
        
        sns.boxplot(
            data=df_filtered, 
            x='Batch Size', 
            y='Latency (ms)', 
            hue='Type',
            palette=colors,
            width=0.6,
            linewidth=1.5
        )
        
        plt.title('Latency Scaling with Batch Size (10 Experts)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
        plt.ylabel('Latency (milliseconds)', fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(title='Operation Type', fontsize=12, title_fontsize=13)
        
        plt.tight_layout()
        plt.savefig(output_path / 'batch_size_scaling_analysis.pdf', 
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        print("  ✓ Batch size analysis plot saved")
    
    def _create_expert_count_analysis(self, df: pd.DataFrame, output_path: Path):
        """Create expert count analysis plot"""
        plt.figure(figsize=(10, 6))
        
        # Filter for batch size = 4 (middle ground)
        df_filtered = df[df['Batch Size'] == 4]
        
        colors = ['#FFB3BA', '#BAFFC9']
        
        sns.boxplot(
            data=df_filtered, 
            x='Expert Count', 
            y='Latency (ms)', 
            hue='Type',
            palette=colors,
            width=0.6,
            linewidth=1.5
        )
        
        plt.title('Latency Scaling with Expert Count (Batch Size 4)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Experts Prefetched', fontsize=14, fontweight='bold')
        plt.ylabel('Latency (milliseconds)', fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(title='Operation Type', fontsize=12, title_fontsize=13)
        
        plt.tight_layout()
        plt.savefig(output_path / 'expert_count_scaling_analysis.pdf', 
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        print("  ✓ Expert count analysis plot saved")
    
    def _create_ratio_analysis(self, df: pd.DataFrame, output_path: Path):
        """Create ratio analysis plot"""
        plt.figure(figsize=(10, 6))
        
        # Calculate ratios
        ratio_data = []
        configs = df['Configuration'].unique()
        
        for config in configs:
            comp_data = df[(df['Configuration'] == config) & (df['Type'] == 'Computation')]['Latency (ms)']
            mem_data = df[(df['Configuration'] == config) & (df['Type'] == 'Memory Transfer')]['Latency (ms)']
            
            # Calculate ratio for each trial
            min_len = min(len(comp_data), len(mem_data))
            for i in range(min_len):
                ratio = mem_data.iloc[i] / comp_data.iloc[i]
                
                parts = config.split('_')
                batch_size = int(parts[0][2:])
                expert_count = int(parts[1][1:])
                
                ratio_data.append({
                    'Configuration': config,
                    'Batch Size': batch_size,
                    'Expert Count': expert_count,
                    'Memory/Computation Ratio': ratio
                })
        
        ratio_df = pd.DataFrame(ratio_data)
        
        # Create box plot
        sns.boxplot(
            data=ratio_df, 
            x='Configuration', 
            y='Memory/Computation Ratio',
            color='#BAFFC9',  # Light green
            width=0.6,
            linewidth=1.5
        )
        
        plt.title('Memory Transfer to Computation Latency Ratio', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Configuration (Batch Size, Expert Count)', fontsize=14, fontweight='bold')
        plt.ylabel('Latency Ratio (Memory ÷ Computation)', fontsize=14, fontweight='bold')
        
        # Add horizontal line at ratio = 1
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, 
                   label='Equal latency (ratio = 1)')
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path / 'memory_computation_ratio_analysis.pdf', 
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        print("  ✓ Ratio analysis plot saved")
    
    def save_results(self, output_dir: str = "benchmark_results"):
        """Save detailed benchmark results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_results = {
            'metadata': self.results['metadata'],
            'computation': {},
            'memory_transfer': {}
        }
        
        for config_key in self.results['computation'].keys():
            json_results['computation'][config_key] = {
                'times_ms': self.results['computation'][config_key],
                'mean_ms': float(np.mean(self.results['computation'][config_key])),
                'std_ms': float(np.std(self.results['computation'][config_key])),
                'median_ms': float(np.median(self.results['computation'][config_key]))
            }
            
            json_results['memory_transfer'][config_key] = {
                'times_ms': self.results['memory_transfer'][config_key],
                'mean_ms': float(np.mean(self.results['memory_transfer'][config_key])),
                'std_ms': float(np.std(self.results['memory_transfer'][config_key])),
                'median_ms': float(np.median(self.results['memory_transfer'][config_key]))
            }
        
        # Save results
        import json
        with open(output_path / 'detailed_latency_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed results saved to: {output_path}")

def main():
    """Run comprehensive latency benchmark with publication plots"""
    print("=" * 80)
    print("MEMORY VS COMPUTATION LATENCY ANALYSIS")
    print("Publication-Quality Box-Whisker Plots")
    print("=" * 80)
    
    # Configuration
    config = BenchmarkConfig(
        model_dim=768,
        ff_dim=3072,
        num_experts=128,
        batch_sizes=[1, 2, 4],
        expert_counts=[1, 10],
        num_trials=200,  # High number for good distributions
        seq_length=512
    )
    
    # Run benchmark
    benchmark = LatencyBenchmark(config)
    results = benchmark.run_comprehensive_benchmark()
    
    # Create plots
    benchmark.create_publication_plots("plots")
    
    # Save detailed results
    benchmark.save_results("benchmark_results")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("All plots saved as PDFs in ./plots/")
    print("Detailed results saved in ./benchmark_results/")
    print("=" * 80)

if __name__ == "__main__":
    main()