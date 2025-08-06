#!/usr/bin/env python3
"""
Fix the Intelligent Strategy Implementation for Qwen MoE

The current Intelligent strategy has several critical bugs:
1. Numerical overflow in recency weights
2. Poor prediction logic for top-8 routing
3. Insufficient expert predictions

This script provides a corrected implementation and re-runs the experiments.
"""

import numpy as np
import pandas as pd
import json
import pickle
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import sys
import os

# Add the current directory to path to import from main evaluation
sys.path.append('.')
from qwen_comprehensive_evaluation import *

class QwenIntelligentStrategyFixed(QwenPrefetchingStrategy):
    """FIXED: Intelligent adaptive prefetching with learning"""
    
    def __init__(self, config: QwenExperimentConfig):
        super().__init__("Intelligent", config)
        self.transition_matrix = np.zeros((config.num_experts, config.num_experts))
        self.expert_frequency = np.zeros(config.num_experts)
        self.expert_recency = np.zeros(config.num_experts)
        self.adaptation_rate = 0.1
        self.decay_rate = 0.99
        self.time_step = 0
        
    def predict_experts(self, routing_history: List[List[int]], 
                       current_token: int) -> List[int]:
        if len(routing_history) < 1:
            # Cold start: predict most frequent experts from global distribution
            # For Qwen, use a reasonable default based on Zipfian distribution
            return list(range(min(24, self.config.num_experts)))
            
        self.time_step += 1
        
        # Update learning from recent history
        self._update_learning(routing_history[-10:])  # Use last 10 tokens
        
        # Multi-strategy prediction
        predicted = set()
        
        # 1. Transition-based prediction
        if len(routing_history) >= 2:
            current_experts = routing_history[-1]
            for expert_id in current_experts:
                # Get top transitions for this expert
                transitions = self.transition_matrix[expert_id]
                if transitions.sum() > 0:
                    # Normalize and get top predictions
                    normalized = transitions / transitions.sum()
                    top_indices = np.argsort(normalized)[-6:]  # Top 6 transitions
                    predicted.update(top_indices.tolist())
        
        # 2. Frequency-based prediction (recently popular experts)
        freq_scores = self.expert_frequency.copy()
        top_frequent = np.argsort(freq_scores)[-12:]  # Top 12 frequent
        predicted.update(top_frequent.tolist())
        
        # 3. Recency-based prediction (recently used experts)
        recency_scores = self.expert_recency.copy()
        top_recent = np.argsort(recency_scores)[-8:]  # Top 8 recent
        predicted.update(top_recent.tolist())
        
        # 4. Pattern continuation (if we see repeating patterns)
        if len(routing_history) >= 3:
            # Look for patterns in last 3 tokens
            recent_pattern = tuple(tuple(sorted(experts)) for experts in routing_history[-3:])
            # Simple pattern matching - if we've seen this before, predict continuation
            pattern_experts = self._predict_from_pattern(routing_history[-3:])
            predicted.update(pattern_experts)
        
        # Convert to list and limit size (Qwen uses top-8, so cache more for safety)
        predicted_list = list(predicted)[:32]  # Cache 32 experts (4x the routing requirement)
        
        return predicted_list
        
    def _update_learning(self, recent_history: List[List[int]]):
        """Update learning models from recent routing history"""
        
        # Update transition matrix
        for i in range(len(recent_history) - 1):
            current_experts = recent_history[i]
            next_experts = recent_history[i + 1]
            
            for curr_expert in current_experts:
                for next_expert in next_experts:
                    self.transition_matrix[curr_expert][next_expert] += self.adaptation_rate
        
        # Update frequency tracking
        for token_experts in recent_history:
            for expert_id in token_experts:
                self.expert_frequency[expert_id] += 1.0
        
        # Update recency with time decay
        self.expert_recency *= self.decay_rate  # Decay all
        if recent_history:
            for expert_id in recent_history[-1]:  # Boost recent
                self.expert_recency[expert_id] = self.time_step
                
        # Apply decay to transition matrix to prevent unlimited growth
        self.transition_matrix *= self.decay_rate
        self.expert_frequency *= self.decay_rate
        
    def _predict_from_pattern(self, recent_tokens: List[List[int]]) -> List[int]:
        """Simple pattern-based prediction"""
        # Look for expert co-occurrence patterns
        if len(recent_tokens) < 2:
            return []
            
        # Find experts that often appear together
        co_occurring = set()
        current_experts = set(recent_tokens[-1])
        
        # Simple co-occurrence: if expert A is active and expert B often follows A
        for expert_a in current_experts:
            # Find experts that have high transition probability from expert_a
            transitions = self.transition_matrix[expert_a]
            if transitions.sum() > 0:
                normalized = transitions / transitions.sum()
                # Add experts with >10% transition probability
                high_prob = np.where(normalized > 0.1)[0]
                co_occurring.update(high_prob.tolist())
        
        return list(co_occurring)[:8]
        
    def get_complexity_score(self) -> float:
        return 7.0  # High complexity

def run_fixed_intelligent_experiment():
    """Run experiments with fixed Intelligent strategy"""
    
    print("🔧 Running Fixed Intelligent Strategy Experiments...")
    
    config = QwenExperimentConfig()
    routing_sim = QwenRoutingSimulator(config)
    cache = QwenExpertCache(config)
    performance_sim = QwenPerformanceSimulator(config)
    
    # Use fixed strategy
    strategy = QwenIntelligentStrategyFixed(config)
    
    batch_sizes = [1, 2, 4, 8, 16]
    num_runs = 10
    
    all_results = []
    
    for batch_size in batch_sizes:
        print(f"  Testing batch size {batch_size}...")
        
        batch_results = []
        
        for run_idx in range(num_runs):
            # Generate routing sequence
            routing_sequence = routing_sim.generate_routing_sequence(
                config.sequence_length // batch_size,
                batch_size
            )
            
            # Flatten routing
            flattened_routing = []
            for batch in routing_sequence:
                flattened_routing.extend(batch)
                
            # Run simulation
            result = performance_sim.simulate_inference(flattened_routing, cache, strategy)
            
            # Add metadata
            result.update({
                'strategy': 'E',
                'strategy_name': strategy.name,
                'batch_size': batch_size,
                'run': run_idx,
                'sequence_length': len(flattened_routing),
                'complexity_score': strategy.get_complexity_score(),
                'timestamp': datetime.now().isoformat()
            })
            
            batch_results.append(result)
            
        all_results.extend(batch_results)
        
        # Print summary for this batch size
        latencies = [r['total_latency_ms'] for r in batch_results]
        hit_rates = [r['cache_hit_rate'] for r in batch_results]
        
        print(f"    Latency: {np.mean(latencies):.1f} ± {np.std(latencies):.1f} ms")
        print(f"    Hit Rate: {np.mean(hit_rates)*100:.1f}%")
        
    return all_results

def compare_strategies():
    """Compare original vs fixed Intelligent strategy"""
    
    print("📊 Comparing Original vs Fixed Intelligent Strategy")
    print("="*60)
    
    # Load original results
    original_df = pd.read_csv('results/qwen_comprehensive_summary.csv')
    original_intelligent = original_df[original_df['strategy'] == 'E']
    
    # Run fixed experiments
    fixed_results = run_fixed_intelligent_experiment()
    
    # Create comparison
    print("\nBatch Size 1 Comparison:")
    print("-" * 40)
    
    original_b1 = original_intelligent[original_intelligent['batch_size'] == 1].iloc[0]
    fixed_b1_results = [r for r in fixed_results if r['batch_size'] == 1]
    
    fixed_latency = np.mean([r['total_latency_ms'] for r in fixed_b1_results])
    fixed_hit_rate = np.mean([r['cache_hit_rate'] for r in fixed_b1_results])
    
    print(f"Original Intelligent:")
    print(f"  Latency: {original_b1['total_latency_ms_mean']:.1f} ms")
    print(f"  Hit Rate: {original_b1['cache_hit_rate_mean']*100:.1f}%")
    print()
    print(f"Fixed Intelligent:")
    print(f"  Latency: {fixed_latency:.1f} ms")
    print(f"  Hit Rate: {fixed_hit_rate*100:.1f}%")
    print()
    print(f"Improvement:")
    print(f"  Latency: {(original_b1['total_latency_ms_mean'] / fixed_latency):.2f}× better")
    print(f"  Hit Rate: {(fixed_hit_rate - original_b1['cache_hit_rate_mean'])*100:.1f}% points better")
    
    # Compare with other strategies
    print(f"\nComparison with Other Strategies (Batch Size 1):")
    print("-" * 50)
    
    other_strategies = original_df[original_df['batch_size'] == 1]
    for _, row in other_strategies.iterrows():
        if row['strategy'] != 'E':
            speedup_vs_fixed = row['total_latency_ms_mean'] / fixed_latency
            print(f"{row['strategy_name']:12}: {row['total_latency_ms_mean']:6.1f}ms ({speedup_vs_fixed:.2f}× vs Fixed Intelligent)")
    
    print(f"{'Fixed Intel':12}: {fixed_latency:6.1f}ms (1.00× baseline)")
    
    return fixed_results

if __name__ == "__main__":
    results = compare_strategies()
    
    print("\n" + "="*60)
    print("🎯 INTELLIGENT STRATEGY FIX COMPLETE")
    print("="*60)
    print("The original Intelligent strategy had critical bugs:")
    print("1. Numerical overflow in recency weights")
    print("2. Poor prediction logic for top-8 routing")
    print("3. Insufficient expert caching")
    print()
    print("Fixed version should show significant improvements!")
    print("="*60)