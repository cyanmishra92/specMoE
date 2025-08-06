"""
SpecMoE: Expert Prefetching for Mixture-of-Experts Models

A comprehensive framework for MoE expert prefetching optimization including:
- Neural prediction models for expert routing
- Batch-aware optimization strategies  
- Multi-architecture evaluation frameworks
- Hardware-aware cost modeling
"""

__version__ = "1.0.0"
__author__ = "SpecMoE Research Team"

from . import models, training, evaluation, data, utils

__all__ = ["models", "training", "evaluation", "data", "utils"]