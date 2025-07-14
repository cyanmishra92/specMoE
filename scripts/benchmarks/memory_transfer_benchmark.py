#!/usr/bin/env python3
"""
Memory Transfer Benchmark for MoE Expert Prefetching
Measures GPU-to-GPU and CPU-to-GPU memory transfer speeds for expert weights
"""

import torch
import torch.nn as nn
import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ExpertWeightSimulator:
    """Simulates expert weight tensors for benchmarking"""
    
    def __init__(self, model_dim: int = 768, ff_dim: int = 3072, num_experts: int = 128):
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.num_experts = num_experts
        
        # Calculate expert size (typical MoE expert)
        # Each expert has: up_proj (model_dim -> ff_dim), down_proj (ff_dim -> model_dim), gate_proj (model_dim -> ff_dim)
        self.expert_size_mb = self._calculate_expert_size()
        
    def _calculate_expert_size(self) -> float:
        """Calculate expert size in MB"""
        # 3 linear layers per expert (up, down, gate projections)
        params_per_expert = (self.model_dim * self.ff_dim) * 3
        bytes_per_expert = params_per_expert * 4  # float32 = 4 bytes
        return bytes_per_expert / (1024 * 1024)  # Convert to MB
    
    def create_expert_weights(self, device: str = 'cpu') -> torch.Tensor:
        """Create a single expert's weights"""
        # Simulate expert weights as a single tensor
        total_params = (self.model_dim * self.ff_dim) * 3
        return torch.randn(total_params, device=device, dtype=torch.float32)
    
    def create_expert_batch(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """Create a batch of expert weights"""
        total_params = (self.model_dim * self.ff_dim) * 3
        return torch.randn(batch_size, total_params, device=device, dtype=torch.float32)

class MemoryTransferBenchmark:
    """Benchmark memory transfer speeds for different scenarios"""
    
    def __init__(self, model_dim: int = 768, ff_dim: int = 3072, num_experts: int = 128):
        self.simulator = ExpertWeightSimulator(model_dim, ff_dim, num_experts)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        print(f"Benchmarking on device: {self.device}")
        print(f"Expert size: {self.simulator.expert_size_mb:.2f} MB")
        
    def benchmark_cpu_to_gpu(self, batch_sizes: List[int] = [1, 3, 5, 10], 
                           num_trials: int = 100) -> Dict[str, float]:
        """Benchmark CPU -> GPU transfer speeds"""
        print("\n=== CPU -> GPU Transfer Benchmark ===")
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size: {batch_size}")
            
            # Pre-allocate tensors
            cpu_experts = self.simulator.create_expert_batch(batch_size, device='cpu')
            gpu_slot = torch.empty_like(cpu_experts, device=self.device)
            
            # Warm up
            for _ in range(10):
                gpu_slot.copy_(cpu_experts, non_blocking=True)
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(num_trials):
                torch.cuda.synchronize()
                start = time.perf_counter()
                gpu_slot.copy_(cpu_experts, non_blocking=True)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            total_mb = batch_size * self.simulator.expert_size_mb
            throughput = total_mb / (avg_time / 1000)  # MB/s
            
            results[f'batch_{batch_size}'] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'throughput_mb_s': throughput,
                'size_mb': total_mb
            }
            
            print(f"  Batch {batch_size}: {avg_time:.2f}±{std_time:.2f} ms, "
                  f"{throughput:.0f} MB/s, {total_mb:.1f} MB")
        
        return results
    
    def benchmark_gpu_to_gpu(self, batch_sizes: List[int] = [1, 3, 5, 10], 
                           num_trials: int = 100) -> Dict[str, float]:
        """Benchmark GPU -> GPU transfer speeds (different memory regions)"""
        print("\n=== GPU -> GPU Transfer Benchmark ===")
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size: {batch_size}")
            
            # Pre-allocate tensors in different GPU memory regions
            gpu_source = self.simulator.create_expert_batch(batch_size, device=self.device)
            gpu_dest = torch.empty_like(gpu_source, device=self.device)
            
            # Warm up
            for _ in range(10):
                gpu_dest.copy_(gpu_source, non_blocking=True)
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(num_trials):
                torch.cuda.synchronize()
                start = time.perf_counter()
                gpu_dest.copy_(gpu_source, non_blocking=True)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            total_mb = batch_size * self.simulator.expert_size_mb
            throughput = total_mb / (avg_time / 1000)  # MB/s
            
            results[f'batch_{batch_size}'] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'throughput_mb_s': throughput,
                'size_mb': total_mb
            }
            
            print(f"  Batch {batch_size}: {avg_time:.2f}±{std_time:.2f} ms, "
                  f"{throughput:.0f} MB/s, {total_mb:.1f} MB")
        
        return results
    
    def benchmark_memory_allocation(self, batch_sizes: List[int] = [1, 3, 5, 10], 
                                  num_trials: int = 100) -> Dict[str, float]:
        """Benchmark GPU memory allocation speeds"""
        print("\n=== GPU Memory Allocation Benchmark ===")
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size: {batch_size}")
            
            # Warm up
            for _ in range(10):
                temp = self.simulator.create_expert_batch(batch_size, device=self.device)
                del temp
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(num_trials):
                torch.cuda.synchronize()
                start = time.perf_counter()
                temp = self.simulator.create_expert_batch(batch_size, device=self.device)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
                del temp
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            total_mb = batch_size * self.simulator.expert_size_mb
            
            results[f'batch_{batch_size}'] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'size_mb': total_mb
            }
            
            print(f"  Batch {batch_size}: {avg_time:.2f}±{std_time:.2f} ms, {total_mb:.1f} MB")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Dict]:
        """Run all benchmarks and return results"""
        batch_sizes = [1, 3, 5, 10, 20]
        
        results = {
            'cpu_to_gpu': self.benchmark_cpu_to_gpu(batch_sizes),
            'gpu_to_gpu': self.benchmark_gpu_to_gpu(batch_sizes),
            'gpu_allocation': self.benchmark_memory_allocation(batch_sizes),
            'system_info': {
                'device': str(self.device),
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
                'expert_size_mb': self.simulator.expert_size_mb,
                'model_dim': self.simulator.model_dim,
                'ff_dim': self.simulator.ff_dim
            }
        }
        
        self.results = results
        return results
    
    def save_results(self, filename: str = "memory_benchmark_results.json"):
        """Save benchmark results to JSON"""
        output_path = Path("benchmarks") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def plot_results(self):
        """Create visualizations of benchmark results"""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU to GPU transfer times
        cpu_gpu_data = self.results['cpu_to_gpu']
        batch_sizes = [int(k.split('_')[1]) for k in cpu_gpu_data.keys()]
        cpu_gpu_times = [cpu_gpu_data[f'batch_{b}']['avg_time_ms'] for b in batch_sizes]
        
        axes[0, 0].plot(batch_sizes, cpu_gpu_times, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('CPU → GPU Transfer Time')
        axes[0, 0].set_xlabel('Batch Size (# experts)')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # GPU to GPU transfer times
        gpu_gpu_data = self.results['gpu_to_gpu']
        gpu_gpu_times = [gpu_gpu_data[f'batch_{b}']['avg_time_ms'] for b in batch_sizes]
        
        axes[0, 1].plot(batch_sizes, gpu_gpu_times, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('GPU → GPU Transfer Time')
        axes[0, 1].set_xlabel('Batch Size (# experts)')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput comparison
        cpu_gpu_throughput = [cpu_gpu_data[f'batch_{b}']['throughput_mb_s'] for b in batch_sizes]
        gpu_gpu_throughput = [gpu_gpu_data[f'batch_{b}']['throughput_mb_s'] for b in batch_sizes]
        
        axes[1, 0].plot(batch_sizes, cpu_gpu_throughput, 'bo-', label='CPU → GPU', linewidth=2, markersize=8)
        axes[1, 0].plot(batch_sizes, gpu_gpu_throughput, 'ro-', label='GPU → GPU', linewidth=2, markersize=8)
        axes[1, 0].set_title('Memory Throughput')
        axes[1, 0].set_xlabel('Batch Size (# experts)')
        axes[1, 0].set_ylabel('Throughput (MB/s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Memory allocation times
        alloc_data = self.results['gpu_allocation']
        alloc_times = [alloc_data[f'batch_{b}']['avg_time_ms'] for b in batch_sizes]
        
        axes[1, 1].plot(batch_sizes, alloc_times, 'go-', linewidth=2, markersize=8)
        axes[1, 1].set_title('GPU Memory Allocation Time')
        axes[1, 1].set_xlabel('Batch Size (# experts)')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmarks/memory_transfer_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run memory transfer benchmarks"""
    print("=== MoE Expert Memory Transfer Benchmark ===")
    print("Benchmarking memory transfer speeds for expert prefetching")
    
    # Initialize benchmark
    benchmark = MemoryTransferBenchmark(
        model_dim=768,      # Standard transformer dimension
        ff_dim=3072,        # 4x model dimension
        num_experts=128     # Switch transformer configuration
    )
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    benchmark.save_results("memory_benchmark_results.json")
    
    # Create visualizations
    benchmark.plot_results()
    
    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"GPU: {results['system_info']['gpu_name']}")
    print(f"Expert size: {results['system_info']['expert_size_mb']:.2f} MB")
    
    # Key metrics for single expert
    cpu_gpu_1 = results['cpu_to_gpu']['batch_1']['avg_time_ms']
    gpu_gpu_1 = results['gpu_to_gpu']['batch_1']['avg_time_ms']
    alloc_1 = results['gpu_allocation']['batch_1']['avg_time_ms']
    
    print(f"\nSingle Expert Transfer Times:")
    print(f"  CPU → GPU: {cpu_gpu_1:.2f} ms")
    print(f"  GPU → GPU: {gpu_gpu_1:.2f} ms")
    print(f"  Allocation: {alloc_1:.2f} ms")
    
    # Key metrics for top-10 prefetch
    cpu_gpu_10 = results['cpu_to_gpu']['batch_10']['avg_time_ms']
    gpu_gpu_10 = results['gpu_to_gpu']['batch_10']['avg_time_ms']
    
    print(f"\nTop-10 Expert Prefetch Times:")
    print(f"  CPU → GPU: {cpu_gpu_10:.2f} ms")
    print(f"  GPU → GPU: {gpu_gpu_10:.2f} ms")
    
    # Calculate practical implications
    print(f"\n=== PRACTICAL IMPLICATIONS ===")
    print(f"For 33.86% accuracy (66.14% miss rate):")
    print(f"  Expected miss penalty: {cpu_gpu_1 * 0.6614:.2f} ms")
    print(f"  Prefetch overhead: {gpu_gpu_10:.2f} ms")
    print(f"  Net benefit: {cpu_gpu_1 * 0.6614 - gpu_gpu_10:.2f} ms per layer")

if __name__ == "__main__":
    main()