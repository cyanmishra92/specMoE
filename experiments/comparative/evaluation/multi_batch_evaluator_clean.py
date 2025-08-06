#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
import json
import os
from pathlib import Path

from iso_cache_framework import IsoCacheFramework, BatchSizeAwareCacheFramework

class MultiBatchEvaluator:
    """
    Comprehensive multi-batch size evaluation framework for MoE expert prefetching strategies.
    
    Provides iso-cache fairness, statistical rigor, and comprehensive performance analysis
    across different batch sizes, sequence lengths, and model architectures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config.get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation parameters
        self.batch_sizes = config.get('batch_sizes', [1, 2, 4, 8, 16, 32, 64])
        self.sequence_lengths = config.get('sequence_lengths', [512, 1024, 2048])
        self.cache_sizes_mb = config.get('cache_sizes_mb', [50, 100, 200])
        self.num_replications = config.get('num_replications', 5)
        
        # Model configurations
        self.model_configs = {
            'switch_transformer': {
                'num_experts': 128,
                'num_layers': 12,
                'top_k': 1,
                'expert_size_mb': 2.5
            },
            'qwen_moe': {
                'num_experts': 64,
                'num_layers': 28,
                'top_k': 8,
                'expert_size_mb': 2.5
            }
        }
        
        # Results storage
        self.comprehensive_results = []
        self.strategy_comparisons = {}
        
    def generate_routing_trace(self, model_config: Dict, batch_size: int, sequence_length: int) -> List[List[List[int]]]:
        """
        Generate realistic routing trace for evaluation.
        Returns: [sequence_step][layer][experts_per_batch_item]
        """
        num_experts = model_config['num_experts']
        num_layers = model_config['num_layers']
        top_k = model_config['top_k']
        
        routing_trace = []
        
        # Generate routing for each sequence step
        for step in range(sequence_length):
            layer_routing = []
            
            for layer in range(num_layers):
                batch_routing = []
                
                for batch_item in range(batch_size):
                    # Expert selection with realistic patterns
                    if step > 0 and np.random.random() < 0.3:  # 30% chance of reusing recent experts
                        prev_experts = routing_trace[-1][layer] if layer < len(routing_trace[-1]) else []
                        if prev_experts and len(prev_experts[batch_item]) > 0:
                            # Reuse some experts from previous step
                            reuse_count = min(top_k//2, len(prev_experts[batch_item]))
                            reused = np.random.choice(prev_experts[batch_item], size=reuse_count, replace=False).tolist()
                            remaining = top_k - reuse_count
                            if remaining > 0:
                                available = [e for e in range(num_experts) if e not in reused]
                                new_experts = np.random.choice(available, size=remaining, replace=False).tolist()
                                selected_experts = reused + new_experts
                            else:
                                selected_experts = reused
                        else:
                            selected_experts = np.random.choice(num_experts, size=top_k, replace=False).tolist()
                    else:
                        selected_experts = np.random.choice(num_experts, size=top_k, replace=False).tolist()
                    
                    batch_routing.append(selected_experts)
                
                layer_routing.append(batch_routing)
            
            routing_trace.append(layer_routing)
        
        return routing_trace
        
    def run_simple_evaluation(self, strategies: Dict) -> pd.DataFrame:
        """Run a simplified evaluation for testing"""
        print("Running simple evaluation...")
        
        all_results = []
        
        # Use only switch_transformer for testing
        model_name = 'switch_transformer'
        model_config = self.model_configs[model_name]
        
        # Simplified configuration
        cache_size = 100  # MB
        sequence_length = 20
        batch_size = 4
        
        for strategy_name, strategy in strategies.items():
            print(f"  Testing {strategy_name}...")
            
            try:
                # Reset strategy
                if hasattr(strategy, 'reset_strategy'):
                    strategy.reset_strategy()
                
                # Generate routing trace
                routing_trace = self.generate_routing_trace(model_config, batch_size, sequence_length)
                
                # Evaluate strategy
                total_latency = 0.0
                
                for step_idx, step_routing in enumerate(routing_trace):
                    for layer_idx, layer_routing in enumerate(step_routing):
                        # Flatten batch routing to get all required experts
                        required_experts = []
                        for batch_item_experts in layer_routing:
                            required_experts.extend(batch_item_experts)
                        
                        # Remove duplicates
                        seen = set()
                        unique_experts = []
                        for expert in required_experts:
                            if expert not in seen:
                                unique_experts.append(expert)
                                seen.add(expert)
                        
                        # Process layer with strategy
                        if hasattr(strategy, 'process_layer'):
                            layer_latency, access_details = strategy.process_layer(layer_idx, unique_experts)
                        else:
                            # Simple fallback
                            layer_latency = 0.0
                            for expert_id in unique_experts:
                                latency, level = strategy.cache.access_expert(expert_id)
                                layer_latency += latency
                        
                        total_latency += layer_latency
                
                # Collect metrics
                cache_metrics = strategy.cache.get_performance_metrics()
                
                if hasattr(strategy, 'get_strategy_metrics'):
                    strategy_metrics = strategy.get_strategy_metrics()
                else:
                    strategy_metrics = {'strategy_name': strategy.__class__.__name__}
                
                result_record = {
                    'model': model_name,
                    'strategy': strategy_name,
                    'batch_size': batch_size,
                    'sequence_length': sequence_length,
                    'cache_size_mb': cache_size,
                    'total_latency': total_latency,
                    'l1_hit_rate': cache_metrics['l1_hit_rate'],
                    'l2_hit_rate': cache_metrics['l2_hit_rate'],
                    'l3_hit_rate': cache_metrics['l3_hit_rate'],
                    'overall_hit_rate': cache_metrics['overall_hit_rate'],
                    'miss_rate': cache_metrics['miss_rate'],
                    'average_cache_latency': cache_metrics['average_latency']
                }
                
                all_results.append(result_record)
                print(f"    Latency: {total_latency:.2f}ms, Hit rate: {cache_metrics['overall_hit_rate']:.3f}")
                
            except Exception as e:
                print(f"    Error evaluating {strategy_name}: {e}")
                continue
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)
        
        if not results_df.empty:
            results_file = self.results_dir / 'simple_evaluation_results.csv'
            results_df.to_csv(results_file, index=False)
            print(f"\nSaved results to {results_file}")
        
        return results_df

# Simple strategy adapters for testing
class SimpleOnDemandStrategy:
    def __init__(self, cache_framework):
        self.cache = cache_framework
    
    def reset_strategy(self):
        self.cache.reset_metrics()
        self.cache.clear_cache()

class SimpleTopKStrategy:
    def __init__(self, cache_framework, num_experts):
        self.cache = cache_framework
        self.num_experts = num_experts
        self.top_experts = list(range(min(20, num_experts)))  # Cache top 20 experts
        
        # Pre-populate cache with top experts
        for expert_id in self.top_experts:
            self.cache.prefetch_expert(expert_id, 'L3')
    
    def reset_strategy(self):
        self.cache.reset_metrics()
        self.cache.clear_cache()
        # Re-populate cache
        for expert_id in self.top_experts:
            self.cache.prefetch_expert(expert_id, 'L3')

if __name__ == "__main__":
    # Simple test configuration
    config = {
        'batch_sizes': [4],
        'sequence_lengths': [20],
        'cache_sizes_mb': [100],
        'num_replications': 1,
        'results_dir': 'results'
    }
    
    evaluator = MultiBatchEvaluator(config)
    
    # Create simple test strategies
    from iso_cache_framework import IsoCacheFramework
    
    cache1 = IsoCacheFramework(100.0, 2.5)
    cache2 = IsoCacheFramework(100.0, 2.5)
    
    strategies = {
        'on_demand': SimpleOnDemandStrategy(cache1),
        'top_k': SimpleTopKStrategy(cache2, 128)
    }
    
    results_df = evaluator.run_simple_evaluation(strategies)
    print("\nSimple evaluation completed!")
    print(results_df)