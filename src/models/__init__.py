"""
MoE Expert Prefetching Strategies Module

Contains implementations of various expert prefetching strategies including:
- Pre-gated MoE strategy (based on arXiv:2308.12066)
- ExpertFlow PLEC strategy (based on arXiv:2410.17954)
- Enhanced versions of existing strategies with iso-cache constraints
"""

from .pg_moe_strategy import PreGatedMoEStrategy
from .expertflow_plec_strategy import ExpertFlowPLECStrategy

__all__ = ['PreGatedMoEStrategy', 'ExpertFlowPLECStrategy']