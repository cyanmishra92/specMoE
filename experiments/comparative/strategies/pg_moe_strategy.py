#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
import time
from ..evaluation.iso_cache_framework import IsoCacheFramework

class PreGatedMoEStrategy:
    """
    Pre-gated MoE Strategy based on arXiv:2308.12066
    
    Key Features:
    - Predictive expert migration with algorithm-system co-design
    - Cross-layer expert activation prediction
    - Memory-efficient subset prefetching
    - Overlapped computation-communication
    - Misprediction penalty handling
    """
    
    def __init__(self, cache_framework: IsoCacheFramework, num_experts: int, num_layers: int, top_k: int = 2):
        self.cache = cache_framework
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.top_k = top_k
        
        # Cross-layer routing history for prediction
        self.layer_routing_history = defaultdict(list)  # layer_id -> [routing_patterns]
        self.cross_layer_patterns = defaultdict(lambda: defaultdict(int))  # layer_i -> {experts_j: count}
        
        # Expert activation prediction models
        self.expert_transition_matrix = np.zeros((num_experts, num_experts))  # P(expert_j | expert_i)
        self.layer_expert_affinity = np.zeros((num_layers, num_experts))  # Layer-expert activation patterns
        
        # Prefetching configuration
        self.prediction_window = 3  # Predict 3 layers ahead
        self.prediction_confidence_threshold = 0.6
        self.prefetch_subset_ratio = 0.7  # Prefetch top 70% predicted experts
        
        # Performance tracking
        self.predictions_made = 0
        self.predictions_correct = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        
        # Memory management
        self.max_concurrent_prefetch = 8  # Limit concurrent prefetching
        self.prefetch_queue = deque()
        
        # Time step tracking
        self.current_time_step = 0
        
    def predict_next_layer_experts(self, current_layer: int, current_experts: List[int]) -> List[int]:
        """
        Predict experts likely to be activated in subsequent layers using
        cross-layer routing pattern analysis.
        """
        if current_layer >= self.num_layers - 1:
            return []
            
        next_layer = current_layer + 1
        predicted_experts = []
        
        # Method 1: Cross-layer pattern analysis
        layer_predictions = defaultdict(float)
        for expert_id in current_experts:
            if expert_id in self.cross_layer_patterns[current_layer]:
                for next_expert, count in self.cross_layer_patterns[current_layer].items():
                    if next_expert < self.num_experts:
                        layer_predictions[next_expert] += count / len(current_experts)
        
        # Method 2: Expert transition probability
        transition_predictions = defaultdict(float)
        for current_expert in current_experts:
            for next_expert in range(self.num_experts):
                transition_prob = self.expert_transition_matrix[current_expert][next_expert]
                if transition_prob > 0:
                    transition_predictions[next_expert] += transition_prob
        
        # Method 3: Layer-expert affinity
        affinity_predictions = {}
        if next_layer < self.num_layers:
            for expert_id in range(self.num_experts):
                affinity_predictions[expert_id] = self.layer_expert_affinity[next_layer][expert_id]
        
        # Combine prediction methods with weighted scoring
        combined_scores = defaultdict(float)
        for expert_id in range(self.num_experts):
            layer_score = layer_predictions.get(expert_id, 0.0) * 0.4
            transition_score = transition_predictions.get(expert_id, 0.0) * 0.4
            affinity_score = affinity_predictions.get(expert_id, 0.0) * 0.2
            
            combined_scores[expert_id] = layer_score + transition_score + affinity_score
        
        # Select top predicted experts above confidence threshold
        sorted_predictions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        for expert_id, score in sorted_predictions:
            if score >= self.prediction_confidence_threshold and len(predicted_experts) < self.top_k * 2:
                predicted_experts.append(expert_id)
        
        self.predictions_made += len(predicted_experts)
        return predicted_experts
    
    def execute_prefetching(self, predicted_experts: List[int], current_layer: int):
        """
        Execute memory-efficient subset prefetching with overlapped communication.
        Only prefetch subset of predicted experts to balance memory and accuracy.
        """
        if not predicted_experts:
            return
            
        # Apply subset ratio to limit memory usage
        subset_size = max(1, int(len(predicted_experts) * self.prefetch_subset_ratio))
        experts_to_prefetch = predicted_experts[:subset_size]
        
        # Limit concurrent prefetching to avoid memory contention
        available_prefetch_slots = self.max_concurrent_prefetch - len(self.prefetch_queue)
        experts_to_prefetch = experts_to_prefetch[:available_prefetch_slots]
        
        # Prefetch experts with priority-based cache level assignment
        for i, expert_id in enumerate(experts_to_prefetch):
            # Higher confidence predictions go to higher cache levels
            if i < len(experts_to_prefetch) * 0.3:  # Top 30% to L2
                target_level = 'L2'
            else:  # Remaining to L3
                target_level = 'L3'
                
            success = self.cache.prefetch_expert(expert_id, target_level)
            if success:
                self.prefetch_queue.append({
                    'expert_id': expert_id,
                    'layer': current_layer + 1,
                    'timestamp': self.current_time_step,
                    'level': target_level
                })
    
    def process_layer(self, layer_id: int, required_experts: List[int]) -> Tuple[float, Dict]:
        """
        Process a layer with Pre-gated MoE strategy:
        1. Access required experts for current layer
        2. Update routing patterns and predictions
        3. Predict and prefetch experts for future layers
        """
        self.current_time_step += 1
        
        # Phase 1: Access required experts for current computation
        total_latency = 0.0
        access_details = {
            'L1': [],
            'L2': [],
            'L3': [],
            'MEMORY': []
        }
        
        for expert_id in required_experts:
            latency, level = self.cache.access_expert(expert_id)
            total_latency += latency
            access_details[level].append(expert_id)
            
            # Check if this was a successful prefetch
            self._check_prefetch_success(expert_id, layer_id)
        
        # Phase 2: Update routing patterns and learning models
        self._update_routing_patterns(layer_id, required_experts)
        self._update_prediction_models(layer_id, required_experts)
        
        # Phase 3: Predict experts for next layer(s) and execute prefetching
        if layer_id < self.num_layers - 1:
            predicted_experts = self.predict_next_layer_experts(layer_id, required_experts)
            
            # Execute prefetching with overlap (simulated asynchronous)
            self.execute_prefetching(predicted_experts, layer_id)
        
        # Clean up old prefetch entries
        self._cleanup_prefetch_queue()
        
        return total_latency, access_details
    
    def _check_prefetch_success(self, expert_id: int, layer_id: int):
        """Check if expert access was a successful prefetch hit"""
        for prefetch_entry in self.prefetch_queue:
            if (prefetch_entry['expert_id'] == expert_id and 
                prefetch_entry['layer'] == layer_id):
                self.prefetch_hits += 1
                self.predictions_correct += 1
                return
        
        # If not found in prefetch queue, it's a prefetch miss
        self.prefetch_misses += 1
    
    def _update_routing_patterns(self, layer_id: int, experts: List[int]):
        """Update cross-layer routing pattern analysis"""
        # Store routing history for this layer
        self.layer_routing_history[layer_id].append(experts.copy())
        
        # Update cross-layer patterns (current layer -> next layer correlation)
        if layer_id > 0 and len(self.layer_routing_history[layer_id - 1]) > 0:
            prev_experts = self.layer_routing_history[layer_id - 1][-1]
            
            # Update cross-layer correlation
            for prev_expert in prev_experts:
                for curr_expert in experts:
                    self.cross_layer_patterns[layer_id - 1][curr_expert] += 1
    
    def _update_prediction_models(self, layer_id: int, experts: List[int]):
        """Update expert prediction models with new observations"""
        # Update expert transition matrix
        if layer_id > 0 and len(self.layer_routing_history[layer_id - 1]) > 0:
            prev_experts = self.layer_routing_history[layer_id - 1][-1]
            
            for prev_expert in prev_experts:
                for curr_expert in experts:
                    if prev_expert < self.num_experts and curr_expert < self.num_experts:
                        self.expert_transition_matrix[prev_expert][curr_expert] += 0.1
        
        # Update layer-expert affinity with exponential moving average
        decay_rate = 0.95
        for expert_id in experts:
            if expert_id < self.num_experts:
                self.layer_expert_affinity[layer_id][expert_id] = (
                    decay_rate * self.layer_expert_affinity[layer_id][expert_id] + 
                    (1 - decay_rate) * 1.0
                )
        
        # Decay unused expert affinities
        for expert_id in range(self.num_experts):
            if expert_id not in experts:
                self.layer_expert_affinity[layer_id][expert_id] *= decay_rate
    
    def _cleanup_prefetch_queue(self):
        """Remove old prefetch entries to prevent memory leaks"""
        current_time = self.current_time_step
        cleanup_threshold = 10  # Remove entries older than 10 time steps
        
        self.prefetch_queue = deque([
            entry for entry in self.prefetch_queue
            if current_time - entry['timestamp'] <= cleanup_threshold
        ])
    
    def get_strategy_metrics(self) -> Dict:
        """Return comprehensive strategy performance metrics"""
        base_metrics = self.cache.get_performance_metrics()
        
        prediction_accuracy = (
            self.predictions_correct / self.predictions_made 
            if self.predictions_made > 0 else 0.0
        )
        
        prefetch_efficiency = (
            self.prefetch_hits / (self.prefetch_hits + self.prefetch_misses)
            if (self.prefetch_hits + self.prefetch_misses) > 0 else 0.0
        )
        
        strategy_metrics = {
            'strategy_name': 'Pre-gated MoE',
            'prediction_accuracy': prediction_accuracy,
            'prefetch_efficiency': prefetch_efficiency,
            'predictions_made': self.predictions_made,
            'predictions_correct': self.predictions_correct,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_misses': self.prefetch_misses,
            'prefetch_queue_size': len(self.prefetch_queue),
            'cross_layer_patterns_learned': sum(
                len(patterns) for patterns in self.cross_layer_patterns.values()
            )
        }
        
        return {**base_metrics, **strategy_metrics}
    
    def reset_strategy(self):
        """Reset strategy state for new evaluation"""
        self.cache.reset_metrics()
        self.cache.clear_cache()
        
        self.layer_routing_history.clear()
        self.cross_layer_patterns.clear()
        
        self.expert_transition_matrix.fill(0)
        self.layer_expert_affinity.fill(0)
        
        self.predictions_made = 0
        self.predictions_correct = 0
        self.prefetch_hits = 0 
        self.prefetch_misses = 0
        
        self.prefetch_queue.clear()
        self.current_time_step = 0
        
    def get_configuration(self) -> Dict:
        """Return strategy configuration for reproducibility"""
        return {
            'strategy': 'Pre-gated MoE',
            'prediction_window': self.prediction_window,
            'confidence_threshold': self.prediction_confidence_threshold,
            'prefetch_subset_ratio': self.prefetch_subset_ratio,
            'max_concurrent_prefetch': self.max_concurrent_prefetch,
            'cache_config': self.cache.get_cache_info()
        }