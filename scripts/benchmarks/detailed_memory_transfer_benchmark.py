#!/usr/bin/env python3
"""
Detailed Memory Transfer Benchmarks for MoE Expert Caching
Measures precise timing for all memory hierarchy levels:
- Host RAM → GPU RAM (PCIe transfer)
- GPU RAM → GPU Cache (L2 cache)  
- GPU Cache → Compute Units (L1/register access)
"""

import torch
import time
import numpy as np
import statistics
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import gc

class DetailedMemoryBenchmark:
    def __init__(self, device: str = "cuda:0", warmup_runs: int = 10, benchmark_runs: int = 100):
        self.device = torch.device(device)
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        # Expert size from Qwen2-MoE-A2.7B (from previous analysis)
        self.expert_params = 14_680_064  # parameters per expert
        self.param_size_bytes = 2  # FP16
        self.expert_size_mb = (self.expert_params * self.param_size_bytes) / (1024 * 1024)
        
        print(f"🔧 Benchmark Config:")
        print(f"   Device: {self.device}")
        print(f"   Expert size: {self.expert_size_mb:.2f} MB")
        print(f"   Warmup runs: {warmup_runs}, Benchmark runs: {benchmark_runs}")
        
    def create_expert_tensor(self, num_experts: int = 1) -> torch.Tensor:
        """Create tensor representing expert weights"""
        total_params = self.expert_params * num_experts
        return torch.randn(total_params, dtype=torch.float16)
    
    def benchmark_host_to_gpu(self) -> Dict[str, float]:
        """Benchmark Host RAM → GPU RAM transfer (PCIe)"""
        print(f"\n🔄 Benchmarking Host → GPU transfers...")
        
        results = {}
        expert_counts = [1, 4, 10, 20]  # Single, top-4, top-10, top-20
        
        for num_experts in expert_counts:
            # Create data on CPU
            cpu_tensor = self.create_expert_tensor(num_experts)
            size_mb = (cpu_tensor.numel() * cpu_tensor.element_size()) / (1024 * 1024)
            
            # Warmup
            for _ in range(self.warmup_runs):
                gpu_tensor = cpu_tensor.to(self.device)
                torch.cuda.synchronize()
                del gpu_tensor
                torch.cuda.empty_cache()
            
            # Benchmark
            times = []
            for _ in range(self.benchmark_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                gpu_tensor = cpu_tensor.to(self.device, non_blocking=False)
                torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
                
                del gpu_tensor
                torch.cuda.empty_cache()
            
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times)
            bandwidth_gbps = (size_mb / avg_time) * 1000 / 1024  # GB/s
            
            results[f"host_to_gpu_{num_experts}_experts"] = {
                "time_ms": avg_time,
                "std_ms": std_time,
                "size_mb": size_mb,
                "bandwidth_gbps": bandwidth_gbps
            }
            
            print(f"   {num_experts:2d} experts: {avg_time:6.2f}ms ± {std_time:4.2f}ms, "
                  f"{bandwidth_gbps:5.1f} GB/s ({size_mb:5.1f} MB)")
        
        return results
    
    def benchmark_gpu_cache_access(self) -> Dict[str, float]:
        """Benchmark GPU RAM → GPU Cache access patterns"""
        print(f"\n🏃 Benchmarking GPU Cache access patterns...")
        
        results = {}
        expert_counts = [1, 4, 10, 20]
        
        for num_experts in expert_counts:
            # Create data directly on GPU
            gpu_tensor = self.create_expert_tensor(num_experts).to(self.device)
            size_mb = (gpu_tensor.numel() * gpu_tensor.element_size()) / (1024 * 1024)
            
            # Warmup - access pattern to load into cache
            for _ in range(self.warmup_runs):
                _ = gpu_tensor.sum()
                torch.cuda.synchronize()
            
            # Benchmark sequential access (cache-friendly)
            times_sequential = []
            for _ in range(self.benchmark_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Sequential memory access
                result = gpu_tensor.sum()
                torch.cuda.synchronize()
                
                end = time.perf_counter()
                times_sequential.append((end - start) * 1000)
            
            # Benchmark random access (cache-unfriendly)
            indices = torch.randperm(gpu_tensor.size(0), device=self.device)
            times_random = []
            for _ in range(self.benchmark_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Random memory access
                result = gpu_tensor[indices[:gpu_tensor.size(0)//4]].sum()
                torch.cuda.synchronize()
                
                end = time.perf_counter()
                times_random.append((end - start) * 1000)
            
            seq_time = statistics.mean(times_sequential)
            rand_time = statistics.mean(times_random)
            cache_benefit = rand_time / seq_time
            
            results[f"gpu_cache_{num_experts}_experts"] = {
                "sequential_time_ms": seq_time,
                "random_time_ms": rand_time,
                "cache_benefit_ratio": cache_benefit,
                "size_mb": size_mb
            }
            
            print(f"   {num_experts:2d} experts: Sequential {seq_time:6.3f}ms, "
                  f"Random {rand_time:6.3f}ms, Cache benefit: {cache_benefit:.2f}x")
            
            del gpu_tensor
            torch.cuda.empty_cache()
        
        return results
    
    def benchmark_compute_access(self) -> Dict[str, float]:
        """Benchmark GPU Cache → Compute Units (actual computation)"""
        print(f"\n🧮 Benchmarking Compute access patterns...")
        
        results = {}
        expert_counts = [1, 4, 10, 20]
        
        # Simulate expert computation (matrix multiplication)
        batch_size = 32
        hidden_dim = 2048
        intermediate_dim = 8192
        
        for num_experts in expert_counts:
            # Create expert weights (simplified MLP layer)
            w1 = torch.randn(num_experts, hidden_dim, intermediate_dim, 
                           dtype=torch.float16, device=self.device)
            w2 = torch.randn(num_experts, intermediate_dim, hidden_dim,
                           dtype=torch.float16, device=self.device)
            
            # Input data
            input_tensor = torch.randn(batch_size, hidden_dim, 
                                     dtype=torch.float16, device=self.device)
            
            size_mb = ((w1.numel() + w2.numel()) * w1.element_size()) / (1024 * 1024)
            
            # Warmup
            for _ in range(self.warmup_runs):
                # Simulate MoE expert computation
                intermediate = torch.matmul(input_tensor, w1[0])  # First expert
                output = torch.matmul(intermediate, w2[0])
                torch.cuda.synchronize()
            
            # Benchmark single expert computation
            times_single = []
            for _ in range(self.benchmark_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Single expert forward pass
                intermediate = torch.matmul(input_tensor, w1[0])
                output = torch.matmul(intermediate, w2[0])
                torch.cuda.synchronize()
                
                end = time.perf_counter()
                times_single.append((end - start) * 1000)
            
            # Benchmark multi-expert computation (simulating top-k)
            times_multi = []
            k = min(4, num_experts)  # Top-4 or available experts
            for _ in range(self.benchmark_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Multi-expert computation
                outputs = []
                for i in range(k):
                    intermediate = torch.matmul(input_tensor, w1[i])
                    output = torch.matmul(intermediate, w2[i])
                    outputs.append(output)
                
                # Combine outputs (weighted sum simulation)
                combined = torch.stack(outputs).mean(dim=0)
                torch.cuda.synchronize()
                
                end = time.perf_counter()
                times_multi.append((end - start) * 1000)
            
            single_time = statistics.mean(times_single)
            multi_time = statistics.mean(times_multi)
            
            results[f"compute_{num_experts}_experts"] = {
                "single_expert_time_ms": single_time,
                "multi_expert_time_ms": multi_time,
                "experts_computed": k,
                "size_mb": size_mb,
                "compute_efficiency": single_time * k / multi_time
            }
            
            print(f"   {num_experts:2d} experts: Single {single_time:.3f}ms, "
                  f"Top-{k} {multi_time:.3f}ms, Efficiency: {single_time * k / multi_time:.2f}x")
            
            del w1, w2, input_tensor
            torch.cuda.empty_cache()
        
        return results
    
    def benchmark_memory_allocation(self) -> Dict[str, float]:
        """Benchmark GPU memory allocation overhead"""
        print(f"\n💾 Benchmarking Memory allocation overhead...")
        
        results = {}
        expert_counts = [1, 4, 10, 20]
        
        for num_experts in expert_counts:
            size_mb = self.expert_size_mb * num_experts
            total_params = self.expert_params * num_experts
            
            # Benchmark allocation
            alloc_times = []
            for _ in range(self.benchmark_runs):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                tensor = torch.empty(total_params, dtype=torch.float16, device=self.device)
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                alloc_times.append((end - start) * 1000)
                del tensor
            
            # Benchmark deallocation
            dealloc_times = []
            for _ in range(self.benchmark_runs):
                tensor = torch.empty(total_params, dtype=torch.float16, device=self.device)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                del tensor
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                dealloc_times.append((end - start) * 1000)
            
            alloc_time = statistics.mean(alloc_times)
            dealloc_time = statistics.mean(dealloc_times)
            
            results[f"allocation_{num_experts}_experts"] = {
                "allocation_time_ms": alloc_time,
                "deallocation_time_ms": dealloc_time,
                "total_overhead_ms": alloc_time + dealloc_time,
                "size_mb": size_mb
            }
            
            print(f"   {num_experts:2d} experts: Alloc {alloc_time:.3f}ms, "
                  f"Dealloc {dealloc_time:.3f}ms, Total {alloc_time + dealloc_time:.3f}ms")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, any]:
        """Run all benchmarks and compile results"""
        print(f"🚀 Starting Comprehensive Memory Transfer Benchmarks")
        print(f"=" * 60)
        
        results = {
            "device": str(self.device),
            "expert_size_mb": self.expert_size_mb,
            "expert_params": self.expert_params,
            "benchmark_runs": self.benchmark_runs,
            "timestamp": time.time()
        }
        
        # Run all benchmark categories
        results["host_to_gpu"] = self.benchmark_host_to_gpu()
        results["gpu_cache"] = self.benchmark_gpu_cache_access()  
        results["compute"] = self.benchmark_compute_access()
        results["allocation"] = self.benchmark_memory_allocation()
        
        # Summary analysis
        print(f"\n📊 SUMMARY ANALYSIS")
        print(f"=" * 60)
        
        # Host → GPU transfer analysis
        single_host_gpu = results["host_to_gpu"]["host_to_gpu_1_experts"]["time_ms"]
        multi_host_gpu = results["host_to_gpu"]["host_to_gpu_20_experts"]["time_ms"]
        print(f"Host → GPU: {single_host_gpu:.2f}ms (1 expert) → {multi_host_gpu:.2f}ms (20 experts)")
        
        # Cache efficiency analysis  
        single_cache = results["gpu_cache"]["gpu_cache_1_experts"]["cache_benefit_ratio"]
        multi_cache = results["gpu_cache"]["gpu_cache_20_experts"]["cache_benefit_ratio"]
        print(f"Cache benefit: {single_cache:.2f}x (1 expert) → {multi_cache:.2f}x (20 experts)")
        
        # Compute efficiency
        single_compute = results["compute"]["compute_1_experts"]["single_expert_time_ms"]
        multi_compute = results["compute"]["compute_20_experts"]["multi_expert_time_ms"]
        print(f"Compute: {single_compute:.3f}ms (1 expert) → {multi_compute:.3f}ms (4 experts)")
        
        # Memory hierarchy comparison
        transfer_dominant = single_host_gpu / single_compute
        print(f"Transfer vs Compute bottleneck: {transfer_dominant:.1f}x")
        
        return results
    
    def save_results(self, results: Dict[str, any], output_file: Path):
        """Save benchmark results to JSON file"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Detailed Memory Transfer Benchmarks")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark runs")
    parser.add_argument("--output", type=str, default="qwen15_moe_a27b/results/detailed_memory_benchmark.json",
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    benchmark = DetailedMemoryBenchmark(
        device=args.device,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs
    )
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    output_path = Path(args.output)
    benchmark.save_results(results, output_path)
    
    print(f"\n✅ Detailed memory benchmarking completed!")
    print(f"Use these results to calibrate the memory management simulator.")

if __name__ == "__main__":
    main()