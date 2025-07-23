"""
Comprehensive MoE Expert Prefetching Comparative Evaluation Framework

This package provides a comprehensive evaluation framework for comparing
MoE expert prefetching strategies, including:

- Iso-cache constraint system for fair comparison
- Multi-batch size evaluation across realistic deployment scenarios
- Hardware-aware cost modeling for practical deployment guidance
- Integration of state-of-the-art strategies from recent research papers
- Statistical significance testing and publication-quality visualizations

Key Components:
- evaluation/: Core evaluation infrastructure
- strategies/: Prefetching strategy implementations
- analysis/: Visualization and analysis tools
- results/: Generated experimental data and reports
"""

__version__ = "1.0.0"
__author__ = "MoE Prefetching Research Team"

from .evaluation.iso_cache_framework import IsoCacheFramework, BatchSizeAwareCacheFramework
from .evaluation.multi_batch_evaluator import MultiBatchEvaluator
from .evaluation.hardware_cost_model import HardwareAwareCostModel, DeviceType

from .strategies.pg_moe_strategy import PreGatedMoEStrategy
from .strategies.expertflow_plec_strategy import ExpertFlowPLECStrategy

__all__ = [
    'IsoCacheFramework',
    'BatchSizeAwareCacheFramework', 
    'MultiBatchEvaluator',
    'HardwareAwareCostModel',
    'DeviceType',
    'PreGatedMoEStrategy',
    'ExpertFlowPLECStrategy'
]