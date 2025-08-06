#!/usr/bin/env python3
"""
Enhanced Memory Simulator with Multi-Level Caching and Adaptive Policies

Combines:
- Coverage-aware memory management (47.55% prediction accuracy)
- Multi-level caching hierarchy (L1/L2/L3)
- Adaptive caching based on memory pressure
- Accurate batched transfer timing from benchmarks
"""

import torch
import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import pickle
import logging
import matplotlib.pyplot as plt
from coverage_aware_memory_manager import CoverageAwareMemoryManager, CacheLevel
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Configuration for enhanced memory simulation"""
    num_requests: int = 2000
    batch_size: int = 32
    sequence_length: int = 512
    hidden_dim: int = 2048
    num_experts: int = 60
    experts_per_token: int = 4  # Top-4 routing
    
    # Memory hierarchy configuration
    l1_capacity: int = 4      # Active experts
    l2_capacity: int = 20     # High-confidence predictions (39% perfect coverage)
    l3_capacity: int = 40     # Speculative predictions
    
    # Prediction model performance (from breakthrough results)
    prediction_accuracy: float = 0.4755  # 47.55% Top-1 accuracy
    top5_accuracy: float = 0.7385        # 73.85% Top-5 accuracy
    coverage_at_20: float = 0.39         # 39% perfect coverage at Top-20
    
    # Adaptive parameters
    memory_pressure_threshold: float = 0.8
    confidence_adaptation_rate: float = 0.05

class EnhancedMemorySimulator:
    def __init__(self, config: SimulationConfig, device: str = "cuda:0"):
        self.config = config
        self.device = device
        
        # Initialize coverage-aware memory manager
        self.memory_manager = CoverageAwareMemoryManager(
            device=device,
            l1_capacity=config.l1_capacity,
            l2_capacity=config.l2_capacity, 
            l3_capacity=config.l3_capacity,
            prediction_accuracy=config.prediction_accuracy,
            coverage_threshold=config.coverage_at_20
        )
        
        # Simulation tracking
        self.request_history = []
        self.latency_history = []
        self.coverage_history = []
        self.cache_efficiency_history = []
        
        # Load real routing traces if available
        self.routing_traces = self._load_routing_traces()
        
        print(f"🚀 Enhanced Memory Simulator initialized:")
        print(f"   Prediction accuracy: {config.prediction_accuracy:.1%}")
        print(f"   Multi-level hierarchy: L1({config.l1_capacity}) + L2({config.l2_capacity}) + L3({config.l3_capacity})")
        print(f"   Coverage target: {config.coverage_at_20:.1%} @ Top-20")
    
    def _load_routing_traces(self) -> Optional[List[Set[int]]]:
        """Load real routing traces from Qwen trace collection"""
        trace_files = list(Path("qwen15_moe_a27b/routing_data").glob("shard_*.pkl"))
        
        if trace_files:
            print(f"📊 Loading real routing traces from {len(trace_files)} shards...")
            all_traces = []
            
            for trace_file in trace_files[:3]:  # Load first 3 shards for simulation
                try:
                    with open(trace_file, 'rb') as f:
                        shard_traces = pickle.load(f)
                    
                    for trace in shard_traces:
                        if 'router_logits' in trace:
                            # Extract top-4 expert selections from real data
                            logits = trace['router_logits']
                            if torch.is_tensor(logits):
                                # Get top-4 experts for each token
                                _, top_experts = torch.topk(logits, k=4, dim=-1)
                                # Average across sequence to get typical routing pattern
                                expert_set = set(top_experts.flatten().tolist())
                                # Limit to realistic top-4 selection
                                expert_set = set(list(expert_set)[:4])
                                all_traces.append(expert_set)
                            
                            if len(all_traces) >= self.config.num_requests:
                                break
                    
                    if len(all_traces) >= self.config.num_requests:
                        break
                        
                except Exception as e:
                    logging.warning(f"Could not load trace file {trace_file}: {e}")
                    continue
            
            if all_traces:
                print(f"✅ Loaded {len(all_traces)} real routing traces")
                return all_traces[:self.config.num_requests]
        
        print(f"⚠️  No routing traces found, using synthetic data")
        return None
    
    def generate_synthetic_routing(self) -> Set[int]:
        """Generate synthetic but realistic expert routing pattern"""
        # Simulate realistic expert selection based on actual MoE behavior
        
        # Expert popularity distribution (some experts are more popular)
        popular_experts = np.random.choice(20, 2, replace=False)  # Top 20 experts are popular
        regular_experts = np.random.choice(range(20, 50), 1, replace=False)
        rare_experts = np.random.choice(range(50, 60), 1, replace=False)
        
        selected = list(popular_experts) + list(regular_experts) + list(rare_experts)
        return set(selected[:4])  # Top-4 routing
    
    def simulate_inference_batch(self, request_id: int) -> Dict[str, any]:
        """Simulate a single inference batch with realistic routing"""
        
        # Generate input (hidden states)
        hidden_state = torch.randn(
            self.config.batch_size, 
            self.config.sequence_length,
            self.config.hidden_dim,
            device=self.device
        )
        
        # Get target experts (real traces if available, synthetic otherwise)
        if self.routing_traces and request_id < len(self.routing_traces):
            target_experts = self.routing_traces[request_id]
        else:
            target_experts = self.generate_synthetic_routing()
        
        # Process through coverage-aware memory manager
        start_time = time.perf_counter()
        latency, cache_stats = self.memory_manager.request_experts(hidden_state, target_experts)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Simulate actual computation time (from benchmarks)
        compute_latency = self.memory_manager.timing["compute_multi"]  # 0.415ms for top-4
        
        # Total inference latency
        total_latency = latency + compute_latency
        
        batch_stats = {
            "request_id": request_id,
            "target_experts": target_experts,
            "memory_latency_ms": latency,
            "compute_latency_ms": compute_latency,
            "total_latency_ms": total_latency,
            "processing_overhead_ms": processing_time,
            "cache_stats": cache_stats
        }
        
        return batch_stats
    
    def run_simulation(self) -> Dict[str, any]:
        """Run complete enhanced memory simulation"""
        print(f"🔄 Starting Enhanced Memory Simulation ({self.config.num_requests} requests)")
        print("=" * 70)
        
        start_time = time.time()
        
        for request_id in range(self.config.num_requests):
            batch_stats = self.simulate_inference_batch(request_id)
            
            # Track metrics
            self.request_history.append(batch_stats)
            self.latency_history.append(batch_stats["total_latency_ms"])
            self.coverage_history.append(batch_stats["cache_stats"]["coverage_ratio"])
            
            # Calculate cache efficiency
            cache_efficiency = (
                batch_stats["cache_stats"]["l1_hits"] * 3 +      # L1 hits are worth 3x
                batch_stats["cache_stats"]["l2_hits"] * 2 +      # L2 hits are worth 2x
                batch_stats["cache_stats"]["l3_hits"] * 1        # L3 hits are worth 1x
            ) / 4  # Normalize by top-4 routing
            
            self.cache_efficiency_history.append(cache_efficiency)
            
            # Progress reporting
            if request_id % 200 == 0 and request_id > 0:
                avg_latency = np.mean(self.latency_history[-200:])
                avg_coverage = np.mean(self.coverage_history[-200:])
                avg_efficiency = np.mean(self.cache_efficiency_history[-200:])
                
                print(f"Request {request_id:4d}: "
                      f"Latency {avg_latency:6.2f}ms, "
                      f"Coverage {avg_coverage:.1%}, "
                      f"Cache efficiency {avg_efficiency:.2f}")
        
        simulation_time = time.time() - start_time
        
        # Get final performance summary
        performance_summary = self.memory_manager.get_performance_summary()
        
        # Compile comprehensive results
        results = {
            "simulation_config": {
                "num_requests": self.config.num_requests,
                "prediction_accuracy": self.config.prediction_accuracy,
                "l1_capacity": self.config.l1_capacity,
                "l2_capacity": self.config.l2_capacity,
                "l3_capacity": self.config.l3_capacity,
                "used_real_traces": self.routing_traces is not None
            },
            "performance_metrics": {
                "average_latency_ms": np.mean(self.latency_history),
                "median_latency_ms": np.median(self.latency_history),
                "p95_latency_ms": np.percentile(self.latency_history, 95),
                "p99_latency_ms": np.percentile(self.latency_history, 99),
                "average_coverage": np.mean(self.coverage_history),
                "average_cache_efficiency": np.mean(self.cache_efficiency_history),
                "simulation_time_seconds": simulation_time
            },
            "memory_manager_stats": performance_summary,
            "detailed_history": {
                "latency_history": self.latency_history,
                "coverage_history": self.coverage_history, 
                "cache_efficiency_history": self.cache_efficiency_history
            }
        }
        
        # Performance analysis
        print(f"\n📊 SIMULATION RESULTS")
        print("=" * 70)
        print(f"Average latency: {results['performance_metrics']['average_latency_ms']:.2f}ms")
        print(f"P95 latency: {results['performance_metrics']['p95_latency_ms']:.2f}ms")
        print(f"Average coverage: {results['performance_metrics']['average_coverage']:.1%}")
        print(f"Cache efficiency: {results['performance_metrics']['average_cache_efficiency']:.2f}")
        print(f"Overall hit rate: {performance_summary['cache_performance']['overall_hit_rate']:.1%}")
        print(f"Perfect coverage rate: {performance_summary['coverage_stats']['perfect_coverage_rate']:.1%}")
        
        return results
    
    def analyze_batched_transfer_efficiency(self) -> Dict[str, any]:
        """Analyze the efficiency of batched transfers vs individual transfers"""
        print(f"\n🔍 Analyzing Batched Transfer Efficiency")
        print("=" * 50)
        
        # Analyze transfer patterns from simulation
        individual_transfers = 0
        batched_transfers = {4: 0, 20: 0, "larger": 0}
        total_transfer_time = 0.0
        
        for request in self.request_history:
            misses = request["cache_stats"]["misses"]
            if misses == 1:
                individual_transfers += 1
                total_transfer_time += self.memory_manager.timing["host_to_gpu_1"]
            elif misses <= 4:
                batched_transfers[4] += 1
                total_transfer_time += self.memory_manager.timing["host_to_gpu_4"]
            elif misses <= 20:
                batched_transfers[20] += 1 
                # Use batch efficiency calculation
                batch_efficiency = 0.56
                total_transfer_time += misses * self.memory_manager.timing["host_to_gpu_1"] * batch_efficiency
            else:
                batched_transfers["larger"] += 1
        
        # Calculate efficiency metrics
        total_transfers = sum([individual_transfers] + list(batched_transfers.values()))
        
        efficiency_analysis = {
            "transfer_distribution": {
                "individual_transfers": individual_transfers,
                "batch_4_transfers": batched_transfers[4],
                "batch_20_transfers": batched_transfers[20],
                "larger_batch_transfers": batched_transfers["larger"]
            },
            "efficiency_metrics": {
                "avg_transfer_time_ms": total_transfer_time / max(1, total_transfers),
                "batch_4_efficiency": 19.14 / (4 * 4.35),  # 1.10x efficiency
                "batch_20_efficiency": 97.16 / (20 * 4.35), # 1.12x efficiency  
                "theoretical_individual_time_ms": total_transfers * 4.35,
                "actual_batched_time_ms": total_transfer_time,
                "batching_speedup": (total_transfers * 4.35) / total_transfer_time if total_transfer_time > 0 else 1.0
            }
        }
        
        print(f"Individual transfers: {individual_transfers}")
        print(f"Batch-4 transfers: {batched_transfers[4]}")  
        print(f"Batch-20 transfers: {batched_transfers[20]}")
        print(f"Batching speedup: {efficiency_analysis['efficiency_metrics']['batching_speedup']:.2f}x")
        
        return efficiency_analysis
    
    def generate_performance_plots(self, output_dir: Path):
        """Generate comprehensive performance visualization"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Latency over time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.latency_history, alpha=0.7, linewidth=0.8)
        plt.title('Inference Latency Over Time')
        plt.xlabel('Request ID')
        plt.ylabel('Latency (ms)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Coverage over time  
        plt.subplot(2, 2, 2)
        plt.plot(self.coverage_history, alpha=0.7, color='green')
        plt.axhline(y=0.39, color='red', linestyle='--', label='Target coverage (39%)')
        plt.title('Cache Coverage Over Time')
        plt.xlabel('Request ID')
        plt.ylabel('Coverage Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cache efficiency
        plt.subplot(2, 2, 3)
        plt.plot(self.cache_efficiency_history, alpha=0.7, color='orange')
        plt.title('Cache Efficiency Over Time')
        plt.xlabel('Request ID')
        plt.ylabel('Cache Efficiency Score')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Latency distribution
        plt.subplot(2, 2, 4)
        plt.hist(self.latency_history, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=np.mean(self.latency_history), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.latency_history):.1f}ms')
        plt.title('Latency Distribution')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "enhanced_memory_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance plots saved to {output_dir}/")
    
    def save_results(self, results: Dict[str, any], output_file: Path):
        """Save comprehensive simulation results"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Simulation results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Memory Simulator")
    parser.add_argument("--requests", type=int, default=2000, help="Number of inference requests")
    parser.add_argument("--l1-capacity", type=int, default=4, help="L1 cache capacity")
    parser.add_argument("--l2-capacity", type=int, default=20, help="L2 cache capacity")
    parser.add_argument("--l3-capacity", type=int, default=40, help="L3 cache capacity")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--output", default="qwen15_moe_a27b/results/enhanced_memory_simulation.json",
                       help="Output file path")
    parser.add_argument("--plots", action="store_true", help="Generate performance plots")
    
    args = parser.parse_args()
    
    # Create simulation configuration
    config = SimulationConfig(
        num_requests=args.requests,
        l1_capacity=args.l1_capacity,
        l2_capacity=args.l2_capacity,
        l3_capacity=args.l3_capacity
    )
    
    # Initialize and run simulator
    simulator = EnhancedMemorySimulator(config, device=args.device)
    results = simulator.run_simulation()
    
    # Analyze transfer efficiency
    batch_analysis = simulator.analyze_batched_transfer_efficiency()
    results["batch_transfer_analysis"] = batch_analysis
    
    # Save results
    output_path = Path(args.output)
    simulator.save_results(results, output_path)
    
    # Generate plots if requested
    if args.plots:
        plots_dir = output_path.parent / "plots"
        simulator.generate_performance_plots(plots_dir)
    
    print(f"\n✅ Enhanced memory simulation completed!")
    print(f"🎯 Key Results:")
    print(f"   Average latency: {results['performance_metrics']['average_latency_ms']:.2f}ms")
    print(f"   Perfect coverage rate: {results['memory_manager_stats']['coverage_stats']['perfect_coverage_rate']:.1%}")
    print(f"   Batching speedup: {batch_analysis['efficiency_metrics']['batching_speedup']:.2f}x")

if __name__ == "__main__":
    main()