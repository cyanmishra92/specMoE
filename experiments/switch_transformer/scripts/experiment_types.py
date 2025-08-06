#!/usr/bin/env python3
"""
Shared data types for experiment results
"""

from dataclasses import dataclass

@dataclass
class ExperimentResult:
    """Single experiment run result"""
    strategy: str
    batch_size: int
    run_id: int
    inference_latency_ms: float
    memory_usage_mb: float
    cache_hit_rate: float
    expert_transfer_time_ms: float
    gpu_utilization: float
    prefetch_accuracy: float
    total_experts_loaded: int
    cache_misses: int
    timestamp: float