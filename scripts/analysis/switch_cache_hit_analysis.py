#!/usr/bin/env python3
"""
Cache Hit Analysis for Switch Transformer Routing Patterns

Analyzes real Switch Transformer routing traces to evaluate:
- Cache hit rates with different cache sizes
- Expert popularity and access patterns  
- Comparison with multi-expert routing (Qwen/Mixtral)
- Optimal caching strategies for single-expert routing
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque, Counter
from dataclasses import dataclass
import argparse
import time

@dataclass
class CacheHitStats:
    """Statistics for cache hit analysis"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    miss_penalty_ms: float = 0.0
    total_latency_ms: float = 0.0

class SwitchCacheSimulator:
    """Simulate caching behavior for Switch Transformer routing"""
    
    def __init__(self, cache_size: int = 16):
        self.cache_size = cache_size
        self.cache = set()  # Set of cached expert IDs
        self.lru_queue = deque()  # LRU ordering
        
        # Performance tracking
        self.stats = CacheHitStats()
        self.access_history = []
        
        # Timing data (from our benchmarks)
        self.cache_hit_time_ms = 0.047  # Sequential cache access
        self.cache_miss_time_ms = 4.35  # Host→GPU transfer for 1 expert
        
    def access_expert(self, expert_id: int) -> bool:
        """
        Access an expert and return True if cache hit
        Updates cache using LRU policy
        """
        self.stats.total_requests += 1
        self.access_history.append(expert_id)
        
        if expert_id in self.cache:
            # Cache hit
            self.stats.cache_hits += 1
            self.stats.total_latency_ms += self.cache_hit_time_ms
            
            # Update LRU - move to front
            self.lru_queue.remove(expert_id)
            self.lru_queue.appendleft(expert_id)
            
            return True
        else:
            # Cache miss
            self.stats.cache_misses += 1
            self.stats.total_latency_ms += self.cache_miss_time_ms
            
            # Add to cache
            if len(self.cache) >= self.cache_size:
                # Evict LRU expert
                evicted = self.lru_queue.pop()
                self.cache.remove(evicted)
            
            self.cache.add(expert_id)
            self.lru_queue.appendleft(expert_id)
            
            return False
    
    def get_hit_rate(self) -> float:
        if self.stats.total_requests == 0:
            return 0.0
        return self.stats.cache_hits / self.stats.total_requests
    
    def get_average_latency(self) -> float:
        if self.stats.total_requests == 0:
            return 0.0
        return self.stats.total_latency_ms / self.stats.total_requests
    
    def reset(self):
        """Reset simulator state"""
        self.cache.clear()
        self.lru_queue.clear()
        self.stats = CacheHitStats()
        self.access_history.clear()

class SwitchRoutingAnalyzer:
    """Analyze Switch Transformer routing patterns"""
    
    def __init__(self, trace_file: Path):
        self.trace_file = trace_file
        self.traces = self._load_traces()
        self.expert_stats = self._analyze_expert_patterns()
        
        print(f"📊 Loaded {len(self.traces)} Switch routing traces")
        print(f"   Model: {self.expert_stats.get('model_name', 'Unknown')}")
        print(f"   Experts: {self.expert_stats.get('num_experts', 'Unknown')}")
    
    def _load_traces(self) -> List[int]:
        """Load Switch Transformer routing traces"""
        if self.trace_file.suffix == '.json':
            with open(self.trace_file, 'r') as f:
                data = json.load(f)
            
            # Check if we have a trace sequence (preferred)
            if 'trace_sequence' in data:
                print(f"✅ Using trace sequence ({len(data['trace_sequence'])} traces)")
                return data['trace_sequence']
            
            # Fallback: extract expert sequence from distribution data
            elif 'expert_distribution' in data:
                traces = []
                for expert_id, count in data['expert_distribution'].items():
                    traces.extend([int(expert_id)] * count)
                
                # Shuffle to simulate realistic temporal access pattern
                np.random.shuffle(traces)
                print(f"✅ Using shuffled distribution data ({len(traces)} traces)")
                return traces
        
        elif self.trace_file.suffix == '.pkl':
            with open(self.trace_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract traces from pickle data
            if isinstance(data, list):
                # Assume each item has expert routing information
                traces = []
                for item in data:
                    if 'expert_id' in item:
                        traces.append(item['expert_id'])
                    elif 'router_logits' in item:
                        # Get top expert from logits
                        logits = item['router_logits']
                        if hasattr(logits, 'argmax'):
                            traces.append(int(logits.argmax()))
                return traces
        
        print(f"⚠️  Could not parse trace file format, generating synthetic traces")
        return self._generate_synthetic_switch_traces()
    
    def _generate_synthetic_switch_traces(self, num_traces: int = 10000) -> List[int]:
        """Generate synthetic but realistic Switch routing traces"""
        # Switch Transformer expert usage follows power law distribution
        num_experts = 128
        
        # Create realistic expert popularity (some experts much more popular)
        popularities = np.random.power(0.5, num_experts)  # Power law
        popularities = popularities / np.sum(popularities)  # Normalize
        
        # Generate traces based on popularity
        expert_ids = np.arange(num_experts)
        traces = np.random.choice(expert_ids, size=num_traces, p=popularities)
        
        print(f"🔧 Generated {num_traces} synthetic Switch traces")
        return traces.tolist()
    
    def _analyze_expert_patterns(self) -> Dict[str, any]:
        """Analyze expert access patterns from traces"""
        if not self.traces:
            return {}
        
        expert_counts = Counter(self.traces)
        total_accesses = len(self.traces)
        
        stats = {
            'total_traces': total_accesses,
            'unique_experts': len(expert_counts),
            'most_popular_expert': expert_counts.most_common(1)[0],
            'expert_distribution': dict(expert_counts),
            'access_entropy': self._calculate_entropy(expert_counts),
            'temporal_locality': self._analyze_temporal_locality()
        }
        
        return stats
    
    def _calculate_entropy(self, counts: Counter) -> float:
        """Calculate entropy of expert access distribution"""
        total = sum(counts.values())
        probabilities = [count / total for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def _analyze_temporal_locality(self) -> Dict[str, float]:
        """Analyze temporal locality in expert accesses"""
        if len(self.traces) < 100:
            return {"locality_score": 0.0}
        
        # Calculate how often the same expert is accessed within a window
        window_sizes = [5, 10, 20, 50]
        locality_scores = {}
        
        for window_size in window_sizes:
            repeats = 0
            total_windows = len(self.traces) - window_size + 1
            
            for i in range(total_windows):
                window = self.traces[i:i + window_size]
                if len(set(window)) < len(window):  # Repeated experts in window
                    repeats += 1
            
            locality_scores[f"window_{window_size}"] = repeats / total_windows
        
        return locality_scores
    
    def run_cache_analysis(self, cache_sizes: List[int] = None) -> Dict[str, any]:
        """Run comprehensive cache hit analysis"""
        if cache_sizes is None:
            cache_sizes = [4, 8, 16, 32, 64, 128]
        
        print(f"🔄 Running cache analysis with sizes: {cache_sizes}")
        
        results = {}
        
        for cache_size in cache_sizes:
            print(f"   Testing cache size: {cache_size}")
            
            # Run simulation
            simulator = SwitchCacheSimulator(cache_size=cache_size)
            
            start_time = time.time()
            for expert_id in self.traces:
                simulator.access_expert(expert_id)
            simulation_time = time.time() - start_time
            
            # Collect results
            results[cache_size] = {
                "hit_rate": simulator.get_hit_rate(),
                "average_latency_ms": simulator.get_average_latency(),
                "total_requests": simulator.stats.total_requests,
                "cache_hits": simulator.stats.cache_hits,
                "cache_misses": simulator.stats.cache_misses,
                "simulation_time_s": simulation_time
            }
            
            print(f"     Hit rate: {simulator.get_hit_rate():.2%}, "
                  f"Avg latency: {simulator.get_average_latency():.3f}ms")
        
        return results
    
    def analyze_optimal_cache_size(self, cache_results: Dict[str, any]) -> Dict[str, any]:
        """Find optimal cache size based on hit rate vs memory trade-off"""
        cache_sizes = list(cache_results.keys())
        hit_rates = [cache_results[size]["hit_rate"] for size in cache_sizes]
        latencies = [cache_results[size]["average_latency_ms"] for size in cache_sizes]
        
        # Calculate diminishing returns - where hit rate improvement slows
        improvements = []
        for i in range(1, len(hit_rates)):
            improvement = hit_rates[i] - hit_rates[i-1]
            improvements.append(improvement)
        
        # Find elbow point (diminishing returns threshold)
        if improvements:
            # Optimal size is where improvement drops below threshold
            threshold = max(improvements) * 0.1  # 10% of max improvement
            optimal_idx = 0
            for i, improvement in enumerate(improvements):
                if improvement < threshold:
                    optimal_idx = i
                    break
            
            optimal_size = cache_sizes[optimal_idx + 1]
        else:
            optimal_size = cache_sizes[0]
        
        return {
            "optimal_cache_size": optimal_size,
            "optimal_hit_rate": cache_results[optimal_size]["hit_rate"],
            "optimal_latency_ms": cache_results[optimal_size]["average_latency_ms"],
            "improvement_curve": list(zip(cache_sizes[1:], improvements))
        }
    
    def compare_with_multi_expert(self, multi_expert_hit_rate: float = 0.808) -> Dict[str, any]:
        """Compare Switch single-expert vs multi-expert routing cache performance"""
        
        # Get best Switch cache performance
        cache_results = self.run_cache_analysis([16, 32, 64])  # Reasonable cache sizes
        best_switch_size = max(cache_results.keys(), key=lambda k: cache_results[k]["hit_rate"])
        best_switch_hit_rate = cache_results[best_switch_size]["hit_rate"]
        best_switch_latency = cache_results[best_switch_size]["average_latency_ms"]
        
        # Multi-expert system stats (from our Qwen analysis)
        multi_expert_latency = 7.08  # ms from enhanced memory simulation
        
        comparison = {
            "switch_transformer": {
                "optimal_cache_size": best_switch_size,
                "hit_rate": best_switch_hit_rate,
                "average_latency_ms": best_switch_latency,
                "experts_per_token": 1
            },
            "multi_expert_system": {
                "cache_size": 64,  # L1(4) + L2(20) + L3(40) 
                "hit_rate": multi_expert_hit_rate,
                "average_latency_ms": multi_expert_latency,
                "experts_per_token": 4
            },
            "performance_ratio": {
                "hit_rate_ratio": best_switch_hit_rate / multi_expert_hit_rate,
                "latency_ratio": best_switch_latency / multi_expert_latency,
                "complexity_factor": 4.0  # Multi-expert is 4x more complex
            }
        }
        
        print(f"\n📊 SWITCH vs MULTI-EXPERT COMPARISON:")
        print(f"Switch (1 expert):     {best_switch_hit_rate:.1%} hit rate, {best_switch_latency:.2f}ms")
        print(f"Multi-expert (4 exp):  {multi_expert_hit_rate:.1%} hit rate, {multi_expert_latency:.2f}ms")
        print(f"Complexity trade-off:   {comparison['performance_ratio']['complexity_factor']:.1f}×")
        
        return comparison
    
    def generate_analysis_plots(self, cache_results: Dict[str, any], output_dir: Path):
        """Generate comprehensive analysis plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Cache hit rate vs cache size
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        cache_sizes = list(cache_results.keys())
        hit_rates = [cache_results[size]["hit_rate"] for size in cache_sizes]
        
        plt.plot(cache_sizes, hit_rates, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Cache Size')
        plt.ylabel('Hit Rate')
        plt.title('Switch Transformer Cache Hit Rate')
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        
        # Plot 2: Average latency vs cache size
        plt.subplot(2, 3, 2)
        latencies = [cache_results[size]["average_latency_ms"] for size in cache_sizes]
        
        plt.plot(cache_sizes, latencies, 'ro-', linewidth=2, markersize=6)
        plt.xlabel('Cache Size')
        plt.ylabel('Average Latency (ms)')
        plt.title('Average Access Latency')
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        
        # Plot 3: Expert popularity distribution
        plt.subplot(2, 3, 3)
        expert_counts = Counter(self.traces)
        popularities = list(expert_counts.values())
        
        plt.hist(popularities, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Access Count')
        plt.ylabel('Number of Experts')
        plt.title('Expert Popularity Distribution')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Hit rate improvement curve
        plt.subplot(2, 3, 4)
        improvements = []
        for i in range(1, len(hit_rates)):
            improvements.append(hit_rates[i] - hit_rates[i-1])
        
        plt.plot(cache_sizes[1:], improvements, 'go-', linewidth=2, markersize=6)
        plt.xlabel('Cache Size')
        plt.ylabel('Hit Rate Improvement')
        plt.title('Diminishing Returns Analysis')
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        
        # Plot 5: Cumulative hit rate distribution
        plt.subplot(2, 3, 5)
        sorted_popularity = sorted(expert_counts.values(), reverse=True)
        cumulative = np.cumsum(sorted_popularity) / sum(sorted_popularity)
        
        plt.plot(range(1, len(cumulative) + 1), cumulative, 'mo-', linewidth=2)
        plt.xlabel('Expert Rank')
        plt.ylabel('Cumulative Access Fraction')
        plt.title('Expert Access Pareto Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Memory efficiency
        plt.subplot(2, 3, 6)
        expert_size_mb = 28  # From benchmarks
        memory_usage = [size * expert_size_mb for size in cache_sizes]
        efficiency = [hit_rates[i] / (memory_usage[i] / 1000) for i in range(len(cache_sizes))]
        
        plt.plot(cache_sizes, efficiency, 'co-', linewidth=2, markersize=6)
        plt.xlabel('Cache Size')
        plt.ylabel('Hit Rate per GB Memory')
        plt.title('Memory Efficiency')
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(output_dir / "switch_cache_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Analysis plots saved to {output_dir}/")
    
    def save_results(self, results: Dict[str, any], output_file: Path):
        """Save comprehensive analysis results"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Switch Transformer Cache Hit Analysis")
    parser.add_argument("--traces", type=str, 
                       default="routing_data/maximum_real_traces.json",
                       help="Path to routing trace file")
    parser.add_argument("--cache-sizes", nargs='+', type=int,
                       default=[4, 8, 16, 32, 64, 128],
                       help="Cache sizes to test")
    parser.add_argument("--output", type=str,
                       default="qwen15_moe_a27b/results/switch_cache_analysis.json",
                       help="Output file path") 
    parser.add_argument("--plots", action="store_true",
                       help="Generate analysis plots")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    trace_file = Path(args.traces)
    if not trace_file.exists():
        print(f"❌ Trace file not found: {trace_file}")
        print("🔧 Using synthetic traces instead")
    
    analyzer = SwitchRoutingAnalyzer(trace_file)
    
    # Run cache analysis
    print(f"\n🚀 Running Switch Cache Hit Analysis")
    print("=" * 50)
    
    cache_results = analyzer.run_cache_analysis(args.cache_sizes)
    
    # Find optimal cache size
    optimal_analysis = analyzer.analyze_optimal_cache_size(cache_results)
    print(f"\n🎯 Optimal cache size: {optimal_analysis['optimal_cache_size']} experts")
    print(f"   Hit rate: {optimal_analysis['optimal_hit_rate']:.2%}")
    print(f"   Latency: {optimal_analysis['optimal_latency_ms']:.3f}ms")
    
    # Compare with multi-expert system
    comparison = analyzer.compare_with_multi_expert()
    
    # Compile results
    results = {
        "expert_statistics": analyzer.expert_stats,
        "cache_analysis": cache_results,
        "optimal_analysis": optimal_analysis,
        "multi_expert_comparison": comparison,
        "analysis_timestamp": time.time()
    }
    
    # Save results
    output_path = Path(args.output)
    analyzer.save_results(results, output_path)
    
    # Generate plots
    if args.plots:
        plots_dir = output_path.parent / "plots"
        analyzer.generate_analysis_plots(cache_results, plots_dir)
    
    print(f"\n✅ Switch cache hit analysis completed!")

if __name__ == "__main__":
    main()