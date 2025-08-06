#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
import time
import heapq
from ..evaluation.iso_cache_framework import IsoCacheFramework

class ExpertFlowPLECStrategy:
    """
    ExpertFlow PLEC (Predictive Locality-aware Expert Caching) Strategy
    Based on arXiv:2410.17954
    
    Key Features:
    - Predictive Locality-aware Expert Caching (PLEC)
    - Routing path information utilization for prediction
    - Dynamic locality-aware prefetching
    - Asynchronous expert loading
    - Adaptive cache replacement policies
    """
    
    def __init__(self, cache_framework: IsoCacheFramework, num_experts: int, num_layers: int, top_k: int = 2):
        self.cache = cache_framework
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.top_k = top_k
        
        # Routing path information storage
        self.routing_paths = []  # Complete routing paths for prediction
        self.path_length_limit = 50  # Limit stored path length for memory efficiency
        
        # Locality analysis components
        self.spatial_locality_matrix = np.zeros((num_experts, num_experts))  # Expert co-occurrence
        self.temporal_locality_tracker = defaultdict(list)  # Expert access timestamps
        self.locality_window_size = 20
        
        # Predictive components
        self.route_predictor = RoutingPathPredictor(num_experts, num_layers)
        self.locality_analyzer = LocalityAnalyzer(num_experts)
        
        # Cache replacement policy (locality-aware)
        self.cache_priority_queue = []  # Priority queue for replacement decisions
        self.expert_access_history = defaultdict(list)
        
        # Asynchronous prefetching simulation
        self.prefetch_scheduler = PrefetchScheduler()
        self.prefetch_operations = deque()
        
        # Performance tracking
        self.locality_hits = 0
        self.prediction_hits = 0
        self.total_predictions = 0
        self.cache_replacements = 0
        
        # Configuration parameters
        self.locality_weight = 0.6  # Weight for locality in prediction
        self.prediction_weight = 0.4  # Weight for path prediction
        self.prefetch_aggressiveness = 0.8  # How aggressively to prefetch
        
        # Time tracking
        self.current_sequence_step = 0
        
    def predict_experts_with_locality(self, current_experts: List[int], layer_id: int) -> List[Tuple[int, float]]:
        """
        Predict future expert activations using routing path information
        and locality analysis.
        """
        predictions = defaultdict(float)
        
        # Method 1: Routing path prediction
        if len(self.routing_paths) > 0:
            path_predictions = self.route_predictor.predict_next_experts(
                self.routing_paths, current_experts, layer_id
            )
            for expert_id, confidence in path_predictions:
                predictions[expert_id] += confidence * self.prediction_weight
        
        # Method 2: Spatial locality analysis
        spatial_predictions = self._analyze_spatial_locality(current_experts)
        for expert_id, score in spatial_predictions:
            predictions[expert_id] += score * self.locality_weight * 0.6
        
        # Method 3: Temporal locality analysis
        temporal_predictions = self._analyze_temporal_locality()
        for expert_id, score in temporal_predictions:
            predictions[expert_id] += score * self.locality_weight * 0.4
        
        # Convert to sorted list of (expert_id, confidence) tuples
        sorted_predictions = sorted(
            [(expert_id, confidence) for expert_id, confidence in predictions.items()],
            key=lambda x: x[1], reverse=True
        )
        
        return sorted_predictions[:self.top_k * 3]  # Return top candidates
    
    def _analyze_spatial_locality(self, current_experts: List[int]) -> List[Tuple[int, float]]:
        """Analyze spatial locality patterns between experts"""
        locality_scores = defaultdict(float)
        
        for current_expert in current_experts:
            if current_expert < self.num_experts:
                # Find experts with high co-occurrence
                for expert_id in range(self.num_experts):
                    cooccurrence = self.spatial_locality_matrix[current_expert][expert_id]
                    if cooccurrence > 0:
                        locality_scores[expert_id] += cooccurrence
        
        # Normalize scores
        if locality_scores:
            max_score = max(locality_scores.values())
            if max_score > 0:
                locality_scores = {
                    expert_id: score / max_score
                    for expert_id, score in locality_scores.items()
                }
        
        return sorted(locality_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _analyze_temporal_locality(self) -> List[Tuple[int, float]]:
        """Analyze temporal locality patterns in expert access"""
        current_time = self.current_sequence_step
        temporal_scores = {}
        
        for expert_id, access_times in self.temporal_locality_tracker.items():
            if access_times:
                # Calculate recency score with exponential decay
                recency_score = sum(
                    np.exp(-(current_time - access_time) / 10.0)
                    for access_time in access_times[-self.locality_window_size:]
                )
                temporal_scores[expert_id] = recency_score
        
        # Normalize scores
        if temporal_scores:
            max_score = max(temporal_scores.values())
            if max_score > 0:
                temporal_scores = {
                    expert_id: score / max_score
                    for expert_id, score in temporal_scores.items()
                }
        
        return sorted(temporal_scores.items(), key=lambda x: x[1], reverse=True)
    
    def execute_locality_aware_prefetching(self, predictions: List[Tuple[int, float]], layer_id: int):
        """
        Execute locality-aware prefetching with asynchronous loading simulation.
        """
        # Filter predictions by confidence and prefetch aggressiveness
        confidence_threshold = 1.0 - self.prefetch_aggressiveness
        experts_to_prefetch = [
            expert_id for expert_id, confidence in predictions
            if confidence >= confidence_threshold
        ]
        
        # Schedule asynchronous prefetching operations
        for i, expert_id in enumerate(experts_to_prefetch):
            # Determine target cache level based on prediction confidence
            confidence = predictions[i][1] if i < len(predictions) else 0.5
            
            if confidence > 0.8:
                target_level = 'L1'
            elif confidence > 0.6:
                target_level = 'L2'
            else:
                target_level = 'L3'
            
            # Schedule prefetch operation
            prefetch_op = {
                'expert_id': expert_id,
                'target_level': target_level,
                'confidence': confidence,
                'layer_id': layer_id + 1,
                'timestamp': self.current_sequence_step,
                'priority': confidence
            }
            
            self.prefetch_scheduler.schedule_prefetch(prefetch_op)
    
    def process_layer(self, layer_id: int, required_experts: List[int]) -> Tuple[float, Dict]:
        """
        Process layer with ExpertFlow PLEC strategy:
        1. Execute scheduled prefetch operations
        2. Access required experts with locality tracking
        3. Update routing paths and locality information
        4. Predict and schedule future prefetching
        """
        self.current_sequence_step += 1
        
        # Phase 1: Execute scheduled prefetch operations (asynchronous simulation)
        self.prefetch_scheduler.execute_scheduled_prefetches(self.cache)
        
        # Phase 2: Access required experts for current layer
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
            
            # Track locality hits
            if self._was_locality_prediction(expert_id):
                self.locality_hits += 1
            
            # Update temporal locality tracking
            self.temporal_locality_tracker[expert_id].append(self.current_sequence_step)
            
            # Limit temporal history to prevent memory growth
            if len(self.temporal_locality_tracker[expert_id]) > self.locality_window_size:
                self.temporal_locality_tracker[expert_id].pop(0)
        
        # Phase 3: Update routing path and locality information
        self._update_routing_path(layer_id, required_experts)
        self._update_spatial_locality(required_experts)
        
        # Phase 4: Predict future experts and schedule prefetching
        if layer_id < self.num_layers - 1:
            predictions = self.predict_experts_with_locality(required_experts, layer_id)
            self.execute_locality_aware_prefetching(predictions, layer_id)
            self.total_predictions += len(predictions)
        
        return total_latency, access_details
    
    def _update_routing_path(self, layer_id: int, experts: List[int]):
        """Update routing path information for prediction"""
        path_entry = {
            'layer_id': layer_id,
            'experts': experts.copy(),
            'timestamp': self.current_sequence_step
        }
        
        self.routing_paths.append(path_entry)
        
        # Limit path history to prevent memory growth
        if len(self.routing_paths) > self.path_length_limit:
            self.routing_paths.pop(0)
    
    def _update_spatial_locality(self, experts: List[int]):
        """Update spatial locality matrix with expert co-occurrences"""
        for i, expert_i in enumerate(experts):
            for j, expert_j in enumerate(experts):
                if i != j and expert_i < self.num_experts and expert_j < self.num_experts:
                    self.spatial_locality_matrix[expert_i][expert_j] += 1.0
                    # Apply decay to prevent unbounded growth
                    self.spatial_locality_matrix[expert_i][expert_j] *= 0.99
    
    def _was_locality_prediction(self, expert_id: int) -> bool:
        """Check if expert access was predicted by locality analysis"""
        # Check recent prefetch operations for this expert
        return self.prefetch_scheduler.was_recently_prefetched(expert_id)
    
    def get_strategy_metrics(self) -> Dict:
        """Return comprehensive strategy performance metrics"""
        base_metrics = self.cache.get_performance_metrics()
        
        locality_hit_rate = (
            self.locality_hits / base_metrics['total_accesses']
            if base_metrics['total_accesses'] > 0 else 0.0
        )
        
        prediction_accuracy = (
            self.prediction_hits / self.total_predictions
            if self.total_predictions > 0 else 0.0
        )
        
        strategy_metrics = {
            'strategy_name': 'ExpertFlow PLEC',
            'locality_hit_rate': locality_hit_rate,
            'prediction_accuracy': prediction_accuracy,
            'locality_hits': self.locality_hits,
            'prediction_hits': self.prediction_hits,
            'total_predictions': self.total_predictions,
            'cache_replacements': self.cache_replacements,
            'routing_paths_stored': len(self.routing_paths),
            'prefetch_operations_completed': self.prefetch_scheduler.completed_operations,
            'spatial_locality_entries': np.count_nonzero(self.spatial_locality_matrix),
            'temporal_locality_experts': len(self.temporal_locality_tracker)
        }
        
        return {**base_metrics, **strategy_metrics}
    
    def reset_strategy(self):
        """Reset strategy state for new evaluation"""
        self.cache.reset_metrics()
        self.cache.clear_cache()
        
        self.routing_paths.clear()
        self.spatial_locality_matrix.fill(0)
        self.temporal_locality_tracker.clear()
        
        self.route_predictor.reset()
        self.locality_analyzer.reset()
        self.prefetch_scheduler.reset()
        
        self.locality_hits = 0
        self.prediction_hits = 0
        self.total_predictions = 0
        self.cache_replacements = 0
        
        self.prefetch_operations.clear()
        self.expert_access_history.clear()
        self.cache_priority_queue.clear()
        
        self.current_sequence_step = 0
    
    def get_configuration(self) -> Dict:
        """Return strategy configuration for reproducibility"""
        return {
            'strategy': 'ExpertFlow PLEC',
            'locality_weight': self.locality_weight,
            'prediction_weight': self.prediction_weight,
            'prefetch_aggressiveness': self.prefetch_aggressiveness,
            'locality_window_size': self.locality_window_size,
            'path_length_limit': self.path_length_limit,
            'cache_config': self.cache.get_cache_info()
        }

class RoutingPathPredictor:
    """Routing path prediction component for ExpertFlow PLEC"""
    
    def __init__(self, num_experts: int, num_layers: int):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.path_patterns = defaultdict(lambda: defaultdict(int))
        
    def predict_next_experts(self, routing_paths: List[Dict], current_experts: List[int], layer_id: int) -> List[Tuple[int, float]]:
        """Predict next experts based on routing path patterns"""
        predictions = defaultdict(float)
        
        # Analyze recent paths for pattern matching
        recent_paths = routing_paths[-10:] if len(routing_paths) >= 10 else routing_paths
        
        for path in recent_paths:
            if path['layer_id'] == layer_id:
                # Look for matching expert patterns
                overlap = set(current_experts) & set(path['experts'])
                if overlap:
                    # Find subsequent layer experts in path
                    next_layer_paths = [p for p in routing_paths if p['layer_id'] == layer_id + 1]
                    for next_path in next_layer_paths:
                        for expert_id in next_path['experts']:
                            predictions[expert_id] += len(overlap) / len(current_experts)
        
        # Normalize predictions
        if predictions:
            max_score = max(predictions.values())
            if max_score > 0:
                predictions = {
                    expert_id: score / max_score
                    for expert_id, score in predictions.items()
                }
        
        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    def reset(self):
        """Reset predictor state"""
        self.path_patterns.clear()

class LocalityAnalyzer:
    """Locality analysis component for ExpertFlow PLEC"""
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.locality_patterns = defaultdict(list)
        
    def analyze_locality(self, experts: List[int]) -> Dict[int, float]:
        """Analyze locality patterns in expert access"""
        # Implementation of locality analysis
        return {}
        
    def reset(self):
        """Reset analyzer state"""
        self.locality_patterns.clear()

class PrefetchScheduler:
    """Asynchronous prefetch operation scheduler"""
    
    def __init__(self):
        self.scheduled_operations = []
        self.completed_operations = 0
        self.recent_prefetches = deque(maxlen=100)
        
    def schedule_prefetch(self, prefetch_op: Dict):
        """Schedule a prefetch operation"""
        heapq.heappush(self.scheduled_operations, 
                      (-prefetch_op['priority'], prefetch_op))
    
    def execute_scheduled_prefetches(self, cache_framework):
        """Execute scheduled prefetch operations"""
        operations_to_execute = []
        
        # Execute high-priority operations first
        while self.scheduled_operations and len(operations_to_execute) < 5:
            priority, operation = heapq.heappop(self.scheduled_operations)
            operations_to_execute.append(operation)
        
        for operation in operations_to_execute:
            success = cache_framework.prefetch_expert(
                operation['expert_id'], 
                operation['target_level']
            )
            if success:
                self.completed_operations += 1
                self.recent_prefetches.append({
                    'expert_id': operation['expert_id'],
                    'timestamp': operation['timestamp']
                })
    
    def was_recently_prefetched(self, expert_id: int) -> bool:
        """Check if expert was recently prefetched"""
        return any(
            prefetch['expert_id'] == expert_id 
            for prefetch in self.recent_prefetches
        )
    
    def reset(self):
        """Reset scheduler state"""
        self.scheduled_operations.clear()
        self.completed_operations = 0
        self.recent_prefetches.clear()