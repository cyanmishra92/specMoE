#!/usr/bin/env python3
"""
Adaptive Caching Policies for MoE Expert Memory Management

Implements intelligent caching policies that adapt based on:
- Memory pressure and resource constraints
- Prediction confidence and accuracy feedback
- Expert usage patterns and temporal locality
- Coverage optimization for perfect 4-expert matches
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import torch

class AdaptationStrategy(Enum):
    CONSERVATIVE = "conservative"    # Slow adaptation, stable performance
    AGGRESSIVE = "aggressive"       # Fast adaptation, higher risk
    BALANCED = "balanced"           # Medium adaptation rate
    DYNAMIC = "dynamic"             # Changes strategy based on conditions

@dataclass
class AdaptiveParameters:
    """Parameters that adapt based on system conditions"""
    l2_confidence_threshold: float = 0.7
    l3_confidence_threshold: float = 0.3
    memory_pressure_threshold: float = 0.8
    adaptation_rate: float = 0.05
    coverage_target: float = 0.39
    
    # Adaptation bounds
    min_l2_threshold: float = 0.5
    max_l2_threshold: float = 0.9
    min_l3_threshold: float = 0.1
    max_l3_threshold: float = 0.6

@dataclass
class CachePerformanceMetrics:
    """Performance metrics for adaptive feedback"""
    hit_rates: Dict[str, float] = field(default_factory=dict)
    coverage_rates: List[float] = field(default_factory=list)
    latency_samples: List[float] = field(default_factory=list)
    prediction_accuracies: List[float] = field(default_factory=list)
    memory_pressures: List[float] = field(default_factory=list)
    
    def add_sample(self, hit_rate: float, coverage: float, latency: float, 
                   pred_accuracy: float, memory_pressure: float):
        self.coverage_rates.append(coverage)
        self.latency_samples.append(latency)
        self.prediction_accuracies.append(pred_accuracy)
        self.memory_pressures.append(memory_pressure)
        
        # Keep only recent samples for adaptation
        max_samples = 100
        if len(self.coverage_rates) > max_samples:
            self.coverage_rates = self.coverage_rates[-max_samples:]
            self.latency_samples = self.latency_samples[-max_samples:]
            self.prediction_accuracies = self.prediction_accuracies[-max_samples:]
            self.memory_pressures = self.memory_pressures[-max_samples:]

class AdaptiveCachePolicy:
    def __init__(self, 
                 strategy: AdaptationStrategy = AdaptationStrategy.BALANCED,
                 base_params: Optional[AdaptiveParameters] = None):
        
        self.strategy = strategy
        self.params = base_params or AdaptiveParameters()
        self.metrics = CachePerformanceMetrics()
        
        # Adaptation tracking
        self.adaptation_history = []
        self.last_adaptation_time = time.time()
        self.adaptation_interval = 10.0  # seconds
        
        # Expert popularity tracking
        self.expert_popularity = defaultdict(int)
        self.expert_recent_access = defaultdict(float)
        self.expert_prediction_success = defaultdict(list)
        
        # Temporal pattern detection
        self.access_patterns = defaultdict(list)
        self.pattern_window = 20  # Track last N accesses
        
        logging.info(f"🧠 Adaptive Cache Policy initialized:")
        logging.info(f"   Strategy: {strategy.value}")
        logging.info(f"   L2 threshold: {self.params.l2_confidence_threshold:.2f}")
        logging.info(f"   L3 threshold: {self.params.l3_confidence_threshold:.2f}")
    
    def should_cache_expert(self, 
                           expert_id: int, 
                           confidence: float,
                           cache_level: str,
                           current_memory_pressure: float) -> bool:
        """
        Decide whether to cache an expert based on adaptive criteria
        """
        current_time = time.time()
        
        # Base confidence thresholds (adaptive)
        if cache_level == "L2":
            base_threshold = self.params.l2_confidence_threshold
        elif cache_level == "L3":
            base_threshold = self.params.l3_confidence_threshold
        else:
            return True  # L1 always caches active experts
        
        # Adjust threshold based on memory pressure
        pressure_adjustment = self._calculate_pressure_adjustment(current_memory_pressure)
        adjusted_threshold = base_threshold + pressure_adjustment
        
        # Expert-specific adjustments
        expert_adjustment = self._calculate_expert_adjustment(expert_id, current_time)
        final_threshold = adjusted_threshold + expert_adjustment
        
        # Ensure within bounds
        if cache_level == "L2":
            final_threshold = np.clip(final_threshold, 
                                    self.params.min_l2_threshold,
                                    self.params.max_l2_threshold)
        else:
            final_threshold = np.clip(final_threshold,
                                    self.params.min_l3_threshold, 
                                    self.params.max_l3_threshold)
        
        return confidence >= final_threshold
    
    def _calculate_pressure_adjustment(self, memory_pressure: float) -> float:
        """Calculate threshold adjustment based on memory pressure"""
        if memory_pressure > self.params.memory_pressure_threshold:
            # High pressure: raise thresholds (be more selective)
            pressure_factor = (memory_pressure - self.params.memory_pressure_threshold) / 0.2
            return min(0.2, pressure_factor * 0.15)  # Max +0.15 adjustment
        elif memory_pressure < 0.3:
            # Low pressure: lower thresholds (be more permissive)
            pressure_factor = (0.3 - memory_pressure) / 0.3
            return max(-0.1, -pressure_factor * 0.1)  # Max -0.1 adjustment
        else:
            return 0.0  # No adjustment in normal range
    
    def _calculate_expert_adjustment(self, expert_id: int, current_time: float) -> float:
        """Calculate expert-specific threshold adjustment"""
        adjustment = 0.0
        
        # Popularity bonus: frequently accessed experts get lower threshold
        popularity = self.expert_popularity.get(expert_id, 0)
        if popularity > 10:  # Popular expert
            adjustment -= 0.05
        elif popularity > 50:  # Very popular expert
            adjustment -= 0.1
        
        # Recency bonus: recently accessed experts get lower threshold
        last_access = self.expert_recent_access.get(expert_id, 0)
        if current_time - last_access < 60:  # Accessed within last minute
            adjustment -= 0.03
        elif current_time - last_access < 10:  # Very recently accessed
            adjustment -= 0.06
        
        # Prediction success bonus: experts with good prediction history
        success_history = self.expert_prediction_success.get(expert_id, [])
        if len(success_history) >= 5:
            success_rate = np.mean(success_history[-10:])  # Last 10 predictions
            if success_rate > 0.6:  # Good prediction success
                adjustment -= 0.04
            elif success_rate < 0.2:  # Poor prediction success
                adjustment += 0.04
        
        return adjustment
    
    def update_expert_access(self, 
                            expert_id: int, 
                            was_predicted: bool,
                            prediction_confidence: float,
                            was_cache_hit: bool):
        """Update expert access statistics for adaptation"""
        current_time = time.time()
        
        # Update popularity
        self.expert_popularity[expert_id] += 1
        
        # Update recency
        self.expert_recent_access[expert_id] = current_time
        
        # Update prediction success
        self.expert_prediction_success[expert_id].append(float(was_predicted))
        if len(self.expert_prediction_success[expert_id]) > 20:
            self.expert_prediction_success[expert_id] = \
                self.expert_prediction_success[expert_id][-20:]  # Keep only recent history
        
        # Update access patterns
        self.access_patterns[expert_id].append(current_time)
        if len(self.access_patterns[expert_id]) > self.pattern_window:
            self.access_patterns[expert_id] = self.access_patterns[expert_id][-self.pattern_window:]
    
    def adapt_parameters(self, 
                        recent_performance: CachePerformanceMetrics) -> bool:
        """
        Adapt caching parameters based on recent performance
        Returns True if parameters were changed
        """
        current_time = time.time()
        
        # Check if it's time to adapt
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return False
        
        if not recent_performance.coverage_rates:
            return False
        
        # Calculate recent performance metrics
        recent_coverage = np.mean(recent_performance.coverage_rates[-20:])
        recent_latency = np.mean(recent_performance.latency_samples[-20:])
        recent_accuracy = np.mean(recent_performance.prediction_accuracies[-20:])
        recent_pressure = np.mean(recent_performance.memory_pressures[-20:])
        
        # Determine adaptation direction
        coverage_delta = recent_coverage - self.params.coverage_target
        performance_score = self._calculate_performance_score(
            recent_coverage, recent_latency, recent_accuracy, recent_pressure)
        
        # Store current parameters for history
        old_l2_threshold = self.params.l2_confidence_threshold
        old_l3_threshold = self.params.l3_confidence_threshold
        
        adapted = False
        
        # Strategy-specific adaptation
        if self.strategy == AdaptationStrategy.CONSERVATIVE:
            adapted = self._conservative_adaptation(coverage_delta, performance_score)
        elif self.strategy == AdaptationStrategy.AGGRESSIVE:
            adapted = self._aggressive_adaptation(coverage_delta, performance_score)
        elif self.strategy == AdaptationStrategy.BALANCED:
            adapted = self._balanced_adaptation(coverage_delta, performance_score)
        elif self.strategy == AdaptationStrategy.DYNAMIC:
            adapted = self._dynamic_adaptation(coverage_delta, performance_score, recent_pressure)
        
        if adapted:
            # Record adaptation
            self.adaptation_history.append({
                "timestamp": current_time,
                "old_l2_threshold": old_l2_threshold,
                "new_l2_threshold": self.params.l2_confidence_threshold,
                "old_l3_threshold": old_l3_threshold,
                "new_l3_threshold": self.params.l3_confidence_threshold,
                "coverage_delta": coverage_delta,
                "performance_score": performance_score,
                "strategy": self.strategy.value
            })
            
            self.last_adaptation_time = current_time
            
            logging.info(f"🔄 Adapted cache parameters:")
            logging.info(f"   L2: {old_l2_threshold:.3f} → {self.params.l2_confidence_threshold:.3f}")
            logging.info(f"   L3: {old_l3_threshold:.3f} → {self.params.l3_confidence_threshold:.3f}")
            logging.info(f"   Coverage: {recent_coverage:.1%}, Target: {self.params.coverage_target:.1%}")
        
        return adapted
    
    def _calculate_performance_score(self, coverage: float, latency: float, 
                                   accuracy: float, pressure: float) -> float:
        """Calculate overall performance score (0-1, higher is better)"""
        # Weighted combination of metrics
        coverage_score = min(1.0, coverage / 0.5)  # Target 50% coverage
        latency_score = max(0.0, 1.0 - (latency - 5.0) / 10.0)  # Target < 5ms
        accuracy_score = accuracy / 0.5  # Relative to 50% accuracy
        pressure_score = max(0.0, 1.0 - pressure)  # Lower pressure is better
        
        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1]  # Coverage most important
        scores = [coverage_score, latency_score, accuracy_score, pressure_score]
        
        return np.average(scores, weights=weights)
    
    def _conservative_adaptation(self, coverage_delta: float, performance_score: float) -> bool:
        """Conservative adaptation: small, careful adjustments"""
        adaptation_rate = self.params.adaptation_rate * 0.5  # Half normal rate
        
        if coverage_delta < -0.05 and performance_score < 0.7:  # Significantly under target
            # Lower thresholds slightly to allow more caching
            self.params.l2_confidence_threshold -= adaptation_rate
            self.params.l3_confidence_threshold -= adaptation_rate * 0.5
            return True
        elif coverage_delta > 0.05 and performance_score > 0.8:  # Over target with good performance
            # Raise thresholds slightly to be more selective
            self.params.l2_confidence_threshold += adaptation_rate
            self.params.l3_confidence_threshold += adaptation_rate * 0.5
            return True
        
        return False
    
    def _aggressive_adaptation(self, coverage_delta: float, performance_score: float) -> bool:
        """Aggressive adaptation: larger, faster adjustments"""
        adaptation_rate = self.params.adaptation_rate * 2.0  # Double normal rate
        
        if coverage_delta < 0:  # Under target
            # Lower thresholds to allow more caching
            self.params.l2_confidence_threshold -= adaptation_rate
            self.params.l3_confidence_threshold -= adaptation_rate
            return True
        elif coverage_delta > 0:  # Over target
            # Raise thresholds to be more selective
            self.params.l2_confidence_threshold += adaptation_rate
            self.params.l3_confidence_threshold += adaptation_rate
            return True
        
        return False
    
    def _balanced_adaptation(self, coverage_delta: float, performance_score: float) -> bool:
        """Balanced adaptation: moderate adjustments based on multiple factors"""
        adaptation_rate = self.params.adaptation_rate
        
        # Adjust based on both coverage and performance
        if abs(coverage_delta) > 0.02 or performance_score < 0.6:
            if coverage_delta < -0.02:  # Under target
                self.params.l2_confidence_threshold -= adaptation_rate * (1 - performance_score)
                self.params.l3_confidence_threshold -= adaptation_rate * 0.7
            elif coverage_delta > 0.02:  # Over target
                self.params.l2_confidence_threshold += adaptation_rate * performance_score
                self.params.l3_confidence_threshold += adaptation_rate * 0.7
            
            return True
        
        return False
    
    def _dynamic_adaptation(self, coverage_delta: float, performance_score: float, 
                          memory_pressure: float) -> bool:
        """Dynamic adaptation: changes strategy based on conditions"""
        # Switch adaptation aggressiveness based on conditions
        if memory_pressure > 0.9:  # High pressure - be conservative
            return self._conservative_adaptation(coverage_delta, performance_score)
        elif performance_score < 0.4:  # Poor performance - be aggressive
            return self._aggressive_adaptation(coverage_delta, performance_score)
        else:  # Normal conditions - balanced
            return self._balanced_adaptation(coverage_delta, performance_score)
    
    def get_cache_priority(self, expert_id: int, confidence: float) -> float:
        """Calculate caching priority for expert (higher = more important to cache)"""
        base_priority = confidence
        
        # Popularity boost
        popularity = self.expert_popularity.get(expert_id, 0)
        popularity_boost = min(0.2, popularity / 100.0)  # Max +0.2
        
        # Recency boost
        current_time = time.time()
        last_access = self.expert_recent_access.get(expert_id, 0)
        recency_boost = max(0, 0.1 - (current_time - last_access) / 600)  # 10-minute decay
        
        # Pattern regularity boost
        pattern_boost = self._calculate_pattern_boost(expert_id)
        
        return base_priority + popularity_boost + recency_boost + pattern_boost
    
    def _calculate_pattern_boost(self, expert_id: int) -> float:
        """Calculate boost based on access pattern regularity"""
        if expert_id not in self.access_patterns:
            return 0.0
        
        access_times = self.access_patterns[expert_id]
        if len(access_times) < 3:
            return 0.0
        
        # Calculate access intervals
        intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
        
        if len(intervals) < 2:
            return 0.0
        
        # Regular intervals get higher priority
        mean_interval = np.mean(intervals)
        interval_std = np.std(intervals)
        
        if mean_interval > 0:
            regularity = 1.0 / (1.0 + interval_std / mean_interval)  # Lower std = more regular
            return min(0.1, regularity * 0.05)  # Max +0.05 boost for very regular patterns
        
        return 0.0
    
    def get_adaptation_summary(self) -> Dict[str, any]:
        """Get summary of adaptive behavior"""
        return {
            "current_parameters": {
                "l2_confidence_threshold": self.params.l2_confidence_threshold,
                "l3_confidence_threshold": self.params.l3_confidence_threshold,
                "memory_pressure_threshold": self.params.memory_pressure_threshold,
                "coverage_target": self.params.coverage_target
            },
            "expert_statistics": {
                "tracked_experts": len(self.expert_popularity),
                "most_popular_expert": max(self.expert_popularity.items(), 
                                          key=lambda x: x[1], default=(None, 0))[0],
                "total_adaptations": len(self.adaptation_history)
            },
            "adaptation_history": self.adaptation_history[-10:],  # Last 10 adaptations
            "strategy": self.strategy.value
        }

# Integration test
if __name__ == "__main__":
    print("🧠 Testing Adaptive Cache Policy")
    
    policy = AdaptiveCachePolicy(AdaptationStrategy.BALANCED)
    metrics = CachePerformanceMetrics()
    
    # Simulate performance samples
    for i in range(50):
        coverage = np.random.normal(0.4, 0.1)
        latency = np.random.normal(7.0, 2.0)
        accuracy = np.random.normal(0.47, 0.05)
        pressure = np.random.normal(0.6, 0.2)
        
        metrics.add_sample(0.8, coverage, latency, accuracy, pressure)
        
        # Test expert access updates
        expert_id = np.random.randint(0, 60)
        was_predicted = np.random.random() < accuracy
        policy.update_expert_access(expert_id, was_predicted, 0.7, True)
        
        # Test caching decisions
        should_cache_l2 = policy.should_cache_expert(expert_id, 0.75, "L2", pressure)
        should_cache_l3 = policy.should_cache_expert(expert_id, 0.4, "L3", pressure)
        
        if i % 15 == 14:  # Attempt adaptation every 15 steps
            adapted = policy.adapt_parameters(metrics)
            if adapted:
                print(f"Step {i}: Adapted parameters")
        
        if i % 10 == 9:
            priority = policy.get_cache_priority(expert_id, 0.6)
            print(f"Step {i}: Expert {expert_id} priority: {priority:.3f}")
    
    # Print final summary
    summary = policy.get_adaptation_summary()
    print(f"\n📊 Adaptation Summary:")
    print(f"L2 threshold: {summary['current_parameters']['l2_confidence_threshold']:.3f}")
    print(f"L3 threshold: {summary['current_parameters']['l3_confidence_threshold']:.3f}")
    print(f"Total adaptations: {summary['expert_statistics']['total_adaptations']}")
    print(f"Tracked experts: {summary['expert_statistics']['tracked_experts']}")
    print(f"Strategy: {summary['strategy']}")