#!/usr/bin/env python3
"""
Virtual Memory Manager for MoE Expert Prefetching
Simulates memory management with prediction tracking and timing
"""

import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import time
from pathlib import Path
import matplotlib.pyplot as plt

@dataclass
class ExpertInfo:
    """Information about an expert"""
    expert_id: int
    size_mb: float
    last_access_time: float
    access_count: int
    is_prefetched: bool = False
    is_cached: bool = False
    load_time_ms: float = 0.0

@dataclass
class PredictionResult:
    """Result of expert prediction"""
    predicted_experts: List[int]
    actual_expert: int
    confidence_scores: List[float]
    layer_id: int
    token_id: int
    timestamp: float

@dataclass
class MemoryEvent:
    """Memory management event"""
    timestamp: float
    event_type: str  # 'prefetch', 'hit', 'miss', 'evict', 'cache_hit'
    expert_id: int
    layer_id: int
    token_id: int
    latency_ms: float
    memory_usage_mb: float

class VirtualMemoryManager:
    """Manages virtual memory for MoE expert prefetching"""
    
    def __init__(self, 
                 num_experts: int = 128,
                 expert_size_mb: float = 18.0,
                 gpu_memory_limit_mb: float = 2048,
                 prefetch_k: int = 10,
                 enable_caching: bool = True):
        
        self.num_experts = num_experts
        self.expert_size_mb = expert_size_mb
        self.gpu_memory_limit_mb = gpu_memory_limit_mb
        self.prefetch_k = prefetch_k
        self.enable_caching = enable_caching
        
        # Memory state
        self.experts_info: Dict[int, ExpertInfo] = {}
        self.gpu_memory_usage = 0.0
        self.prefetched_experts: Set[int] = set()
        self.cached_experts: Set[int] = set()
        self.lru_cache: deque = deque()
        
        # Performance tracking
        self.prediction_history: List[PredictionResult] = []
        self.memory_events: List[MemoryEvent] = []
        self.timing_stats = {
            'total_prefetch_time': 0.0,
            'total_miss_penalty': 0.0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'total_predictions': 0,
            'correct_predictions': 0
        }
        
        # Transfer benchmarks (will be loaded from benchmark results)
        self.transfer_times = {
            'cpu_to_gpu_ms': 2.4,    # Default values
            'gpu_to_gpu_ms': 0.12,
            'allocation_ms': 0.05
        }
        
        # Initialize expert info
        self._initialize_experts()
        
    def _initialize_experts(self):
        """Initialize expert information"""
        for i in range(self.num_experts):
            self.experts_info[i] = ExpertInfo(
                expert_id=i,
                size_mb=self.expert_size_mb,
                last_access_time=0.0,
                access_count=0
            )
    
    def load_benchmark_results(self, benchmark_file: str = "benchmarks/memory_benchmark_results.json"):
        """Load memory transfer benchmarks"""
        try:
            with open(benchmark_file, 'r') as f:
                results = json.load(f)
            
            # Extract single expert transfer times
            self.transfer_times['cpu_to_gpu_ms'] = results['cpu_to_gpu']['batch_1']['avg_time_ms']
            self.transfer_times['gpu_to_gpu_ms'] = results['gpu_to_gpu']['batch_1']['avg_time_ms']
            self.transfer_times['allocation_ms'] = results['gpu_allocation']['batch_1']['avg_time_ms']
            
            print(f"Loaded benchmark results: {self.transfer_times}")
            
        except FileNotFoundError:
            print(f"Benchmark file not found: {benchmark_file}")
            print("Using default transfer times")
    
    def prefetch_experts(self, predicted_experts: List[int], confidence_scores: List[float],
                        layer_id: int, token_id: int, timestamp: float) -> float:
        """Prefetch top-k predicted experts"""
        prefetch_time = 0.0
        
        # Sort by confidence and take top-k
        expert_confidence = list(zip(predicted_experts, confidence_scores))
        expert_confidence.sort(key=lambda x: x[1], reverse=True)
        top_experts = [exp for exp, conf in expert_confidence[:self.prefetch_k]]
        
        for expert_id in top_experts:
            if expert_id not in self.prefetched_experts and expert_id not in self.cached_experts:
                # Check if we need to evict
                if self.gpu_memory_usage + self.expert_size_mb > self.gpu_memory_limit_mb:
                    self._evict_expert(timestamp)
                
                # Prefetch expert
                transfer_time = self.transfer_times['cpu_to_gpu_ms']
                prefetch_time += transfer_time
                
                self.prefetched_experts.add(expert_id)
                self.gpu_memory_usage += self.expert_size_mb
                
                # Log event
                self.memory_events.append(MemoryEvent(
                    timestamp=timestamp,
                    event_type='prefetch',
                    expert_id=expert_id,
                    layer_id=layer_id,
                    token_id=token_id,
                    latency_ms=transfer_time,
                    memory_usage_mb=self.gpu_memory_usage
                ))
        
        self.timing_stats['total_prefetch_time'] += prefetch_time
        return prefetch_time
    
    def access_expert(self, expert_id: int, layer_id: int, token_id: int, 
                     timestamp: float) -> Tuple[float, str]:
        """Access an expert and return access time and hit/miss status"""
        expert_info = self.experts_info[expert_id]
        expert_info.last_access_time = timestamp
        expert_info.access_count += 1
        
        # Check cache first
        if expert_id in self.cached_experts:
            access_time = self.transfer_times['gpu_to_gpu_ms']
            status = 'cache_hit'
            self.timing_stats['total_cache_hits'] += 1
            
            # Update LRU
            if expert_id in self.lru_cache:
                self.lru_cache.remove(expert_id)
            self.lru_cache.appendleft(expert_id)
            
        # Check if prefetched
        elif expert_id in self.prefetched_experts:
            access_time = self.transfer_times['gpu_to_gpu_ms']
            status = 'hit'
            
            # Move to cache if caching is enabled
            if self.enable_caching:
                self.cached_experts.add(expert_id)
                self.lru_cache.appendleft(expert_id)
            
            self.prefetched_experts.remove(expert_id)
            
        else:
            # Cache miss - need to load from CPU
            access_time = self.transfer_times['cpu_to_gpu_ms']
            status = 'miss'
            self.timing_stats['total_cache_misses'] += 1
            self.timing_stats['total_miss_penalty'] += access_time
            
            # Check if we need to evict
            if self.gpu_memory_usage + self.expert_size_mb > self.gpu_memory_limit_mb:
                self._evict_expert(timestamp)
            
            # Load expert to GPU
            self.gpu_memory_usage += self.expert_size_mb
            
            # Add to cache if caching is enabled
            if self.enable_caching:
                self.cached_experts.add(expert_id)
                self.lru_cache.appendleft(expert_id)
        
        # Log event
        self.memory_events.append(MemoryEvent(
            timestamp=timestamp,
            event_type=status,
            expert_id=expert_id,
            layer_id=layer_id,
            token_id=token_id,
            latency_ms=access_time,
            memory_usage_mb=self.gpu_memory_usage
        ))
        
        return access_time, status
    
    def _evict_expert(self, timestamp: float):
        """Evict least recently used expert"""
        if not self.lru_cache and not self.prefetched_experts:
            return
        
        # Prefer evicting prefetched experts over cached ones
        if self.prefetched_experts:
            expert_to_evict = next(iter(self.prefetched_experts))
            self.prefetched_experts.remove(expert_to_evict)
        else:
            expert_to_evict = self.lru_cache.pop()
            self.cached_experts.remove(expert_to_evict)
        
        self.gpu_memory_usage -= self.expert_size_mb
        
        # Log eviction
        self.memory_events.append(MemoryEvent(
            timestamp=timestamp,
            event_type='evict',
            expert_id=expert_to_evict,
            layer_id=-1,
            token_id=-1,
            latency_ms=0.0,
            memory_usage_mb=self.gpu_memory_usage
        ))
    
    def record_prediction(self, predicted_experts: List[int], actual_expert: int,
                         confidence_scores: List[float], layer_id: int, 
                         token_id: int, timestamp: float):
        """Record prediction result"""
        prediction = PredictionResult(
            predicted_experts=predicted_experts,
            actual_expert=actual_expert,
            confidence_scores=confidence_scores,
            layer_id=layer_id,
            token_id=token_id,
            timestamp=timestamp
        )
        
        self.prediction_history.append(prediction)
        self.timing_stats['total_predictions'] += 1
        
        # Check if prediction was correct
        if actual_expert in predicted_experts[:self.prefetch_k]:
            self.timing_stats['correct_predictions'] += 1
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        total_events = len(self.memory_events)
        if total_events == 0:
            return {}
        
        # Count event types
        event_counts = defaultdict(int)
        total_latency = 0.0
        
        for event in self.memory_events:
            event_counts[event.event_type] += 1
            total_latency += event.latency_ms
        
        # Calculate hit rates
        total_accesses = event_counts['hit'] + event_counts['miss'] + event_counts['cache_hit']
        hit_rate = (event_counts['hit'] + event_counts['cache_hit']) / total_accesses if total_accesses > 0 else 0
        
        # Calculate prediction accuracy
        prediction_accuracy = (self.timing_stats['correct_predictions'] / 
                             self.timing_stats['total_predictions']) if self.timing_stats['total_predictions'] > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'cache_hit_rate': event_counts['cache_hit'] / total_accesses if total_accesses > 0 else 0,
            'miss_rate': event_counts['miss'] / total_accesses if total_accesses > 0 else 0,
            'prediction_accuracy': prediction_accuracy,
            'avg_latency_ms': total_latency / total_events,
            'total_prefetch_time_ms': self.timing_stats['total_prefetch_time'],
            'total_miss_penalty_ms': self.timing_stats['total_miss_penalty'],
            'memory_utilization': self.gpu_memory_usage / self.gpu_memory_limit_mb,
            'event_counts': dict(event_counts),
            'total_events': total_events
        }
    
    def simulate_inference(self, routing_traces: List[List[int]], predictions: List[List[int]],
                          confidence_scores: List[List[float]]) -> Dict:
        """Simulate inference with memory management"""
        print("Starting inference simulation...")
        
        total_inference_time = 0.0
        layer_times = []
        
        for token_id, (trace, preds, confs) in enumerate(zip(routing_traces, predictions, confidence_scores)):
            token_time = 0.0
            
            for layer_id in range(len(trace)):
                timestamp = time.time()
                actual_expert = trace[layer_id]
                
                # Get predictions for this layer (if available)
                if layer_id < len(preds) and layer_id < len(confs):
                    predicted_experts = preds[layer_id]
                    layer_confidences = confs[layer_id]
                    
                    # Record prediction
                    self.record_prediction(predicted_experts, actual_expert, 
                                         layer_confidences, layer_id, token_id, timestamp)
                    
                    # Prefetch experts
                    prefetch_time = self.prefetch_experts(predicted_experts, layer_confidences,
                                                        layer_id, token_id, timestamp)
                    token_time += prefetch_time
                
                # Access actual expert
                access_time, status = self.access_expert(actual_expert, layer_id, token_id, timestamp)
                token_time += access_time
            
            layer_times.append(token_time)
            total_inference_time += token_time
        
        # Get final metrics
        metrics = self.get_performance_metrics()
        metrics['total_inference_time_ms'] = total_inference_time
        metrics['avg_token_time_ms'] = total_inference_time / len(routing_traces)
        
        print(f"Simulation complete: {total_inference_time:.2f} ms total")
        return metrics
    
    def save_simulation_results(self, filename: str = "memory_simulation_results.json"):
        """Save simulation results"""
        output_path = Path("simulation_results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        results = {
            'metrics': self.get_performance_metrics(),
            'config': {
                'num_experts': self.num_experts,
                'expert_size_mb': self.expert_size_mb,
                'gpu_memory_limit_mb': self.gpu_memory_limit_mb,
                'prefetch_k': self.prefetch_k,
                'enable_caching': self.enable_caching
            },
            'transfer_times': self.transfer_times,
            'timing_stats': self.timing_stats
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Simulation results saved to: {output_path}")
    
    def plot_memory_usage(self):
        """Plot memory usage over time"""
        timestamps = [event.timestamp for event in self.memory_events]
        memory_usage = [event.memory_usage_mb for event in self.memory_events]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, memory_usage, linewidth=2)
        plt.axhline(y=self.gpu_memory_limit_mb, color='r', linestyle='--', label='Memory Limit')
        plt.xlabel('Time')
        plt.ylabel('GPU Memory Usage (MB)')
        plt.title('GPU Memory Usage Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('simulation_results/memory_usage.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_latency_distribution(self):
        """Plot latency distribution by event type"""
        event_types = ['hit', 'miss', 'cache_hit']
        latencies = {event_type: [] for event_type in event_types}
        
        for event in self.memory_events:
            if event.event_type in latencies:
                latencies[event.event_type].append(event.latency_ms)
        
        plt.figure(figsize=(10, 6))
        for i, event_type in enumerate(event_types):
            if latencies[event_type]:
                plt.hist(latencies[event_type], bins=20, alpha=0.7, 
                        label=f'{event_type} (avg: {np.mean(latencies[event_type]):.2f}ms)')
        
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Memory Access Latency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('simulation_results/latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Test the virtual memory manager"""
    print("=== Virtual Memory Manager Test ===")
    
    # Initialize memory manager
    memory_manager = VirtualMemoryManager(
        num_experts=128,
        expert_size_mb=18.0,
        gpu_memory_limit_mb=2048,
        prefetch_k=10,
        enable_caching=True
    )
    
    # Load benchmark results if available
    memory_manager.load_benchmark_results()
    
    # Simulate some routing traces
    num_tokens = 100
    num_layers = 12
    
    routing_traces = []
    predictions = []
    confidence_scores = []
    
    for _ in range(num_tokens):
        # Generate random routing trace
        trace = [np.random.randint(0, 128) for _ in range(num_layers)]
        routing_traces.append(trace)
        
        # Generate predictions (with some accuracy)
        token_predictions = []
        token_confidences = []
        
        for layer_id in range(num_layers):
            # Generate top-10 predictions with some correlation to actual
            actual = trace[layer_id]
            preds = [actual]  # Include actual with some probability
            
            # Add random predictions
            while len(preds) < 10:
                pred = np.random.randint(0, 128)
                if pred not in preds:
                    preds.append(pred)
            
            # Generate confidence scores
            confs = np.random.uniform(0.1, 0.9, 10)
            confs[0] = np.random.uniform(0.6, 0.9)  # Higher confidence for actual
            
            token_predictions.append(preds)
            token_confidences.append(confs.tolist())
        
        predictions.append(token_predictions)
        confidence_scores.append(token_confidences)
    
    # Run simulation
    metrics = memory_manager.simulate_inference(routing_traces, predictions, confidence_scores)
    
    # Print results
    print("\n=== SIMULATION RESULTS ===")
    print(f"Hit Rate: {metrics['hit_rate']:.2%}")
    print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
    print(f"Miss Rate: {metrics['miss_rate']:.2%}")
    print(f"Prediction Accuracy: {metrics['prediction_accuracy']:.2%}")
    print(f"Average Latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"Total Prefetch Time: {metrics['total_prefetch_time_ms']:.2f} ms")
    print(f"Total Miss Penalty: {metrics['total_miss_penalty_ms']:.2f} ms")
    print(f"Memory Utilization: {metrics['memory_utilization']:.2%}")
    
    # Save results
    memory_manager.save_simulation_results("test_simulation_results.json")
    
    # Create visualizations
    memory_manager.plot_memory_usage()
    memory_manager.plot_latency_distribution()

if __name__ == "__main__":
    main()