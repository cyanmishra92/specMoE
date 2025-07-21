#!/usr/bin/env python3
"""
Run Single Switch Transformer Prefetching Experiment

Tests one strategy + batch size combination with multiple runs for statistical significance.
"""

import json
import time
import numpy as np
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import asdict
import pickle
import logging
from experiment_types import ExperimentResult

class SwitchInferenceSimulator:
    """Simulate Switch Transformer inference with different prefetching strategies"""
    
    def __init__(self, config: Dict, batch_size: int = 1):
        self.config = config
        self.batch_size = batch_size
        self.num_layers = 12  # Switch-Base has 12 layers
        self.num_experts = 128
        self.expert_size_mb = 28.0
        
        # Load timing parameters
        self.timing = config["timing_config"]
        
        # Initialize cache
        self.cache = set()
        self.cache_hits = 0
        self.cache_misses = 0
        self.experts_loaded = 0
        
        # Load Switch routing trace
        self.routing_trace = self._load_routing_trace()
        
        logging.info(f"🔧 Simulator initialized:")
        logging.info(f"   Strategy: {config['strategy_name']}")
        logging.info(f"   Batch size: {batch_size}")
        logging.info(f"   Layers: {self.num_layers}")
        
    def _load_routing_trace(self) -> List[List[int]]:
        """Load routing trace for each layer"""
        # Try multiple possible paths
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent.parent / "routing_data" / "small_switch_trace.json",
            script_dir.parent / "routing_data" / "small_switch_trace.json",
            Path("../routing_data/small_switch_trace.json"),
            Path("routing_data/small_switch_trace.json")
        ]
        
        trace_file = None
        for path in possible_paths:
            if path.exists():
                trace_file = path
                break
        
        if trace_file:
            with open(trace_file) as f:
                data = json.load(f)
            
            # Create layer-wise traces
            base_sequence = data["trace_sequence"]
            layer_traces = []
            
            for layer in range(self.num_layers):
                # Different layers have different expert preferences (simulate reality)
                layer_offset = layer * 7  # Prime offset for variety
                layer_trace = [(expert + layer_offset) % self.num_experts 
                              for expert in base_sequence]
                layer_traces.append(layer_trace)
            
            return layer_traces
        else:
            # Generate synthetic traces
            return self._generate_synthetic_traces()
    
    def _generate_synthetic_traces(self) -> List[List[int]]:
        """Generate synthetic routing traces"""
        traces = []
        for layer in range(self.num_layers):
            # Each layer has different expert preferences
            layer_popularities = np.random.power(0.8, self.num_experts)
            layer_popularities = layer_popularities / np.sum(layer_popularities)
            
            layer_trace = np.random.choice(
                self.num_experts, 
                size=2000, 
                p=layer_popularities
            ).tolist()
            
            traces.append(layer_trace)
        
        return traces
    
    def _predict_next_expert(self, layer: int, position: int, 
                           lookahead: int = 1) -> Tuple[int, float]:
        """Simulate expert prediction with configured accuracy"""
        if position + lookahead >= len(self.routing_trace[layer]):
            # Fallback prediction
            return np.random.randint(0, self.num_experts), 0.1
        
        true_expert = self.routing_trace[layer][position + lookahead]
        prediction_accuracy = self.config["prefetching"].get("accuracy", 0.4755)
        
        if np.random.random() < prediction_accuracy:
            # Correct prediction
            return true_expert, 0.9
        else:
            # Incorrect prediction
            wrong_expert = (true_expert + np.random.randint(1, self.num_experts)) % self.num_experts
            return wrong_expert, 0.3
    
    def _predict_topk_experts(self, layer: int, position: int, 
                            k: int = 10) -> List[Tuple[int, float]]:
        """Predict top-k experts for a layer"""
        predictions = []
        
        # Get base prediction
        base_expert, base_conf = self._predict_next_expert(layer, position)
        predictions.append((base_expert, base_conf))
        
        # Add additional predictions with decreasing confidence
        for i in range(1, k):
            expert_id = (base_expert + i * 3) % self.num_experts  # Spread out predictions
            confidence = base_conf * (1.0 - 0.1 * i)  # Decreasing confidence
            predictions.append((expert_id, max(0.1, confidence)))
        
        return predictions
    
    def _should_prefetch(self, expert_id: int, confidence: float) -> bool:
        """Decide whether to prefetch an expert"""
        if not self.config["prefetching"]["enabled"]:
            return False
        
        threshold = self.config["prefetching"]["confidence_threshold"]
        return confidence >= threshold and expert_id not in self.cache
    
    def _calculate_transfer_time(self, num_experts: int) -> float:
        """Calculate transfer time based on number of experts"""
        if num_experts == 0:
            return 0.0
        elif num_experts == 1:
            return self.timing["cpu_to_gpu_ms"]
        elif num_experts <= 4:
            return self.timing.get("cpu_to_gpu_4_ms", 
                                 self.timing["cpu_to_gpu_ms"] * num_experts * 0.85)
        elif num_experts <= 10:
            return self.timing.get("cpu_to_gpu_10_ms",
                                 self.timing["cpu_to_gpu_ms"] * num_experts * 0.75)
        else:
            # Scale for larger batches
            base_time = self.timing.get("cpu_to_gpu_10_ms", 
                                      self.timing["cpu_to_gpu_ms"] * 10 * 0.75)
            return base_time * (num_experts / 10) * 0.9  # Continued efficiency
    
    def _update_cache(self, expert_id: int):
        """Update cache with LRU policy"""
        cache_size = self.config["caching"]["cache_size"]
        
        if expert_id in self.cache:
            # Cache hit - no update needed for LRU
            self.cache_hits += 1
            return True
        else:
            # Cache miss
            self.cache_misses += 1
            
            if len(self.cache) >= cache_size:
                # Evict random expert (simplified LRU)
                if self.cache:
                    evicted = self.cache.pop()
            
            self.cache.add(expert_id)
            return False
    
    def run_inference(self, sequence_length: int = 512) -> ExperimentResult:
        """Run complete inference simulation"""
        start_time = time.time()
        total_latency = 0.0
        total_transfer_time = 0.0
        prefetch_correct = 0
        prefetch_attempts = 0
        
        # Reset metrics
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.experts_loaded = 0
        
        # Simulate inference through all layers and positions
        for position in range(min(sequence_length, len(self.routing_trace[0]))):
            layer_latencies = []
            
            for layer in range(self.num_layers):
                layer_start = time.perf_counter()
                
                # Get current expert for this layer/position
                current_expert = self.routing_trace[layer][position]
                
                # Check cache and handle loading
                cache_hit = self._update_cache(current_expert)
                
                if cache_hit:
                    # Cache hit - fast access
                    access_time = self.timing["gpu_cache_hit_ms"]
                else:
                    # Cache miss - need to load from CPU
                    access_time = self._calculate_transfer_time(1)
                    self.experts_loaded += 1
                
                # Add computation time
                compute_time = self.timing["compute_ms"]
                layer_latency = access_time + compute_time
                
                # Handle prefetching for future layers
                if self.config["prefetching"]["enabled"]:
                    lookahead = self.config["prefetching"]["lookahead_layers"]
                    
                    if self.config["strategy_name"] == "B_Oracle":
                        # Oracle prefetching - perfect prediction
                        if layer + lookahead < self.num_layers and position < len(self.routing_trace[layer + lookahead]):
                            future_expert = self.routing_trace[layer + lookahead][position]
                            if self._should_prefetch(future_expert, 1.0):
                                # Prefetch with overlap
                                prefetch_time = self._calculate_transfer_time(1)
                                overlap_factor = self.timing.get("prefetch_overlap_factor", 0.8)
                                layer_latency += prefetch_time * (1 - overlap_factor)
                                self.experts_loaded += 1
                                self.cache.add(future_expert)
                                prefetch_correct += 1
                            prefetch_attempts += 1
                    
                    elif self.config["strategy_name"] in ["C_MultiLook", "D_TopK", "E_Intelligent"]:
                        # Prediction-based prefetching
                        k = self.config["prefetching"]["top_k_predictions"]
                        topk_predictions = self._predict_topk_experts(layer, position, k)
                        
                        experts_to_prefetch = []
                        for expert_id, confidence in topk_predictions:
                            if self._should_prefetch(expert_id, confidence):
                                experts_to_prefetch.append(expert_id)
                        
                        if experts_to_prefetch:
                            # Batch prefetch
                            prefetch_time = self._calculate_transfer_time(len(experts_to_prefetch))
                            overlap_factor = self.timing.get("prefetch_overlap_factor", 0.6)
                            layer_latency += prefetch_time * (1 - overlap_factor)
                            
                            # Add to cache
                            for expert_id in experts_to_prefetch:
                                self.cache.add(expert_id)
                                self.experts_loaded += 1
                            
                            # Check prefetch accuracy
                            if layer + lookahead < self.num_layers and position < len(self.routing_trace[layer + lookahead]):
                                true_future = self.routing_trace[layer + lookahead][position]
                                if true_future in experts_to_prefetch:
                                    prefetch_correct += 1
                            prefetch_attempts += 1
                
                # Add prediction overhead
                if self.config["prefetching"]["enabled"]:
                    layer_latency += self.timing.get("prediction_overhead_ms", 0.0)
                
                layer_latencies.append(layer_latency)
                layer_end = time.perf_counter()
            
            # Max latency across layers for this position (parallel processing)
            position_latency = max(layer_latencies) * self.batch_size  # Scale for batch
            total_latency += position_latency
        
        # Calculate metrics
        total_time = time.time() - start_time
        cache_hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        prefetch_accuracy = prefetch_correct / max(1, prefetch_attempts)
        memory_usage = len(self.cache) * self.expert_size_mb
        
        return ExperimentResult(
            strategy=self.config["strategy_name"],
            batch_size=self.batch_size,
            run_id=0,  # Will be set by caller
            inference_latency_ms=total_latency,
            memory_usage_mb=memory_usage,
            cache_hit_rate=cache_hit_rate,
            expert_transfer_time_ms=total_transfer_time,
            gpu_utilization=0.85,  # Simulated
            prefetch_accuracy=prefetch_accuracy,
            total_experts_loaded=self.experts_loaded,
            cache_misses=self.cache_misses,
            timestamp=time.time()
        )

def load_config(strategy: str) -> Dict:
    """Load configuration for a strategy"""
    # Try relative to script location first
    script_dir = Path(__file__).parent
    config_file = script_dir.parent / "configs" / f"strategy_{strategy.lower()}_{'ondemand' if strategy.lower() == 'a' else strategy.lower()}.json"
    
    # Handle specific strategy naming
    strategy_files = {
        'a': 'strategy_a_ondemand.json',
        'b': 'strategy_b_oracle.json', 
        'c': 'strategy_c_multilook.json',
        'd': 'strategy_d_topk.json',
        'e': 'strategy_e_intelligent.json'
    }
    
    config_file = script_dir.parent / "configs" / strategy_files[strategy.lower()]
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file) as f:
        return json.load(f)

def run_experiment(strategy: str, batch_size: int, num_runs: int = 10) -> List[ExperimentResult]:
    """Run experiment with multiple runs for statistics"""
    config = load_config(strategy)
    results = []
    
    print(f"🚀 Running experiment: Strategy {strategy}, Batch size {batch_size}")
    print(f"   Configuration: {config['description']}")
    print(f"   Runs: {num_runs}")
    
    for run_id in range(num_runs):
        print(f"   Run {run_id + 1}/{num_runs}...", end=" ")
        
        simulator = SwitchInferenceSimulator(config, batch_size)
        result = simulator.run_inference()
        result.run_id = run_id
        results.append(result)
        
        print(f"Latency: {result.inference_latency_ms:.2f}ms, "
              f"Hit rate: {result.cache_hit_rate:.2%}")
    
    return results

def save_results(results: List[ExperimentResult], output_file: Path):
    """Save experiment results"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle for analysis
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save as JSON for readability
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"📁 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run Switch Prefetching Experiment")
    parser.add_argument("--strategy", choices=["A", "B", "C", "D", "E"], required=True,
                       help="Prefetching strategy")
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4, 8, 16], required=True,
                       help="Batch size")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of runs for statistics")
    parser.add_argument("--output-dir", type=str, default="../results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(args.strategy, args.batch_size, args.runs)
    
    # Save results
    output_file = Path(args.output_dir) / f"strategy_{args.strategy}_batch_{args.batch_size}.pkl"
    save_results(results, output_file)
    
    # Print summary statistics
    latencies = [r.inference_latency_ms for r in results]
    hit_rates = [r.cache_hit_rate for r in results]
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Latency: {np.mean(latencies):.2f} ± {np.std(latencies):.2f} ms")
    print(f"   Hit Rate: {np.mean(hit_rates):.2%} ± {np.std(hit_rates):.2%}")
    print(f"   Memory Usage: {np.mean([r.memory_usage_mb for r in results]):.1f} MB")

if __name__ == "__main__":
    main()