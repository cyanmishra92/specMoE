#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
import json
import os
from pathlib import Path

# Import existing strategies
import sys
sys.path.append('../evalSwitchB8')
sys.path.append('../evalQwenB8')

from iso_cache_framework import IsoCacheFramework, BatchSizeAwareCacheFramework
from ..strategies.pg_moe_strategy import PreGatedMoEStrategy
from ..strategies.expertflow_plec_strategy import ExpertFlowPLECStrategy

class MultiBatchEvaluator:
    """
    Comprehensive multi-batch size evaluation framework for MoE expert prefetching strategies.
    
    Provides iso-cache fairness, statistical rigor, and comprehensive performance analysis
    across different batch sizes, sequence lengths, and model architectures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config.get('results_dir', 'evalComparative/results'))
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
        
    def initialize_strategies(self, model_config: Dict, cache_size_mb: float) -> Dict[str, Any]:
        """Initialize all prefetching strategies with iso-cache constraints"""
        # Create iso-cache framework
        cache_framework = BatchSizeAwareCacheFramework(
            total_cache_size_mb=cache_size_mb,
            expert_size_mb=model_config['expert_size_mb']
        )
        
        strategies = {}
        
        # Original strategies (adapted with iso-cache)
        strategies['on_demand'] = OnDemandStrategy(cache_framework)
        strategies['oracle'] = OracleStrategy(cache_framework, model_config['num_experts'])
        strategies['top_k'] = TopKStrategy(cache_framework, model_config['num_experts'])
        strategies['multi_lookahead'] = MultiLookAheadStrategy(cache_framework, model_config['num_experts'])
        strategies['intelligent'] = IntelligentStrategy(cache_framework, model_config['num_experts'])
        
        # New paper-based strategies
        strategies['pregated_moe'] = PreGatedMoEStrategy(
            cache_framework, 
            model_config['num_experts'], 
            model_config['num_layers'],
            model_config['top_k']
        )
        
        strategies['expertflow_plec'] = ExpertFlowPLECStrategy(
            cache_framework,
            model_config['num_experts'],
            model_config['num_layers'], 
            model_config['top_k']
        )
        
        return strategies
    
    def generate_routing_trace(self, model_config: Dict, batch_size: int, sequence_length: int) -> List[List[List[int]]]:
        """
        Generate realistic routing trace for evaluation.
        Returns: [sequence_step][layer][experts_per_batch_item]
        """
        num_experts = model_config['num_experts']
        num_layers = model_config['num_layers']
        top_k = model_config['top_k']
        
        # Simulate realistic routing patterns
        routing_trace = []
        
        # Generate routing for each sequence step
        for step in range(sequence_length):
            layer_routing = []
            
            for layer in range(num_layers):
                batch_routing = []
                
                for batch_item in range(batch_size):
                    # Expert selection with realistic patterns
                    if model_config.get('routing_pattern') == 'specialized':
                        # Some experts are more likely to be selected
                        expert_probs = np.zeros(num_experts)
                        specialized_experts = np.random.choice(num_experts, size=num_experts//4, replace=False)
                        expert_probs[specialized_experts] = 0.8
                        expert_probs[expert_probs == 0] = 0.2 / (num_experts - len(specialized_experts))
                        expert_probs /= expert_probs.sum()
                        
                        selected_experts = np.random.choice(
                            num_experts, size=top_k, replace=False, p=expert_probs
                        ).tolist()
                    else:
                        # Default: uniform random selection with some locality
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
    
    def evaluate_strategy_on_trace(self, strategy: Any, routing_trace: List[List[List[int]]], 
                                 batch_size: int) -> Dict[str, Any]:
        """Evaluate a single strategy on a routing trace"""
        strategy.reset_strategy()
        strategy.cache.set_batch_size(batch_size)
        
        total_latency = 0.0
        layer_latencies = []
        detailed_access_log = []
        
        # Process each sequence step
        for step_idx, step_routing in enumerate(routing_trace):
            step_latency = 0.0
            
            # Process each layer in sequence
            for layer_idx, layer_routing in enumerate(step_routing):
                # Flatten batch routing to get all required experts for this layer
                required_experts = []
                for batch_item_experts in layer_routing:
                    required_experts.extend(batch_item_experts)
                
                # Remove duplicates while preserving order
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
                    # Fallback for simpler strategies
                    layer_latency = 0.0
                    access_details = {'L1': [], 'L2': [], 'L3': [], 'MEMORY': []}
                    for expert_id in unique_experts:
                        latency, level = strategy.cache.access_expert(expert_id)
                        layer_latency += latency
                        access_details[level].append(expert_id)
                
                step_latency += layer_latency
                
                # Log detailed access information
                detailed_access_log.append({
                    'step': step_idx,
                    'layer': layer_idx,
                    'batch_size': batch_size,
                    'experts_accessed': unique_experts.copy(),
                    'latency': layer_latency,
                    'access_details': access_details.copy()
                })
            
            layer_latencies.append(step_latency)
            total_latency += step_latency
        
        # Collect comprehensive metrics
        cache_metrics = strategy.cache.get_performance_metrics()
        
        if hasattr(strategy, 'get_strategy_metrics'):
            strategy_metrics = strategy.get_strategy_metrics()
        else:
            strategy_metrics = {'strategy_name': strategy.__class__.__name__}
        
        batch_stats = strategy.cache.get_batch_statistics() if hasattr(strategy.cache, 'get_batch_statistics') else {}
        
        return {
            'total_latency': total_latency,
            'average_step_latency': np.mean(layer_latencies),
            'latency_std': np.std(layer_latencies),
            'cache_metrics': cache_metrics,
            'strategy_metrics': strategy_metrics,
            'batch_statistics': batch_stats,
            'detailed_access_log': detailed_access_log
        }
    
    def run_comprehensive_evaluation(self) -> pd.DataFrame:
        """Run comprehensive evaluation across all configurations"""
        print(f"Starting comprehensive evaluation with {len(self.batch_sizes)} batch sizes, " +
              f"{len(self.sequence_lengths)} sequence lengths, {len(self.cache_sizes_mb)} cache sizes, " +
              f"{self.num_replications} replications per configuration...")\n        \n        all_results = []\n        total_configurations = (\n            len(self.model_configs) * len(self.batch_sizes) * \n            len(self.sequence_lengths) * len(self.cache_sizes_mb) * self.num_replications\n        )\n        \n        config_count = 0\n        \n        for model_name, model_config in self.model_configs.items():\n            print(f\"\\nEvaluating {model_name}...\")\n            \n            for cache_size in self.cache_sizes_mb:\n                # Initialize strategies with iso-cache constraints\n                strategies = self.initialize_strategies(model_config, cache_size)\n                \n                for sequence_length in self.sequence_lengths:\n                    for batch_size in self.batch_sizes:\n                        print(f\"  Configuration: batch_size={batch_size}, seq_len={sequence_length}, cache={cache_size}MB\")\n                        \n                        for replication in range(self.num_replications):\n                            config_count += 1\n                            print(f\"    Replication {replication + 1}/{self.num_replications} \"\n                                  f\"({config_count}/{total_configurations})\")\n                            \n                            # Generate routing trace for this configuration\n                            routing_trace = self.generate_routing_trace(\n                                model_config, batch_size, sequence_length\n                            )\n                            \n                            # Evaluate each strategy\n                            for strategy_name, strategy in strategies.items():\n                                try:\n                                    result = self.evaluate_strategy_on_trace(\n                                        strategy, routing_trace, batch_size\n                                    )\n                                    \n                                    # Compile comprehensive result record\n                                    result_record = {\n                                        'model': model_name,\n                                        'strategy': strategy_name,\n                                        'batch_size': batch_size,\n                                        'sequence_length': sequence_length,\n                                        'cache_size_mb': cache_size,\n                                        'replication': replication,\n                                        'num_experts': model_config['num_experts'],\n                                        'num_layers': model_config['num_layers'],\n                                        'top_k': model_config['top_k'],\n                                        \n                                        # Performance metrics\n                                        'total_latency': result['total_latency'],\n                                        'average_step_latency': result['average_step_latency'],\n                                        'latency_std': result['latency_std'],\n                                        \n                                        # Cache metrics\n                                        'l1_hit_rate': result['cache_metrics']['l1_hit_rate'],\n                                        'l2_hit_rate': result['cache_metrics']['l2_hit_rate'],\n                                        'l3_hit_rate': result['cache_metrics']['l3_hit_rate'],\n                                        'overall_hit_rate': result['cache_metrics']['overall_hit_rate'],\n                                        'miss_rate': result['cache_metrics']['miss_rate'],\n                                        'average_cache_latency': result['cache_metrics']['average_latency'],\n                                        \n                                        # Strategy-specific metrics\n                                        **{f\"strategy_{k}\": v for k, v in result['strategy_metrics'].items() \n                                           if isinstance(v, (int, float, bool))}\n                                    }\n                                    \n                                    all_results.append(result_record)\n                                    \n                                except Exception as e:\n                                    print(f\"    Error evaluating {strategy_name}: {e}\")\n                                    continue\n        \n        # Convert to DataFrame and save\n        results_df = pd.DataFrame(all_results)\n        \n        # Save comprehensive results\n        results_file = self.results_dir / 'comprehensive_comparative_results.csv'\n        results_df.to_csv(results_file, index=False)\n        print(f\"\\nSaved comprehensive results to {results_file}\")\n        \n        # Save detailed configuration\n        config_file = self.results_dir / 'evaluation_config.json'\n        with open(config_file, 'w') as f:\n            json.dump(self.config, f, indent=2)\n        \n        return results_df\n    \n    def compute_statistical_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:\n        \"\"\"Compute statistical analysis across strategies and configurations\"\"\"\n        from scipy import stats\n        \n        analysis = {\n            'strategy_performance': {},\n            'batch_size_scaling': {},\n            'cache_sensitivity': {},\n            'model_comparison': {},\n            'statistical_significance': {}\n        }\n        \n        # Strategy performance analysis\n        for strategy in results_df['strategy'].unique():\n            strategy_data = results_df[results_df['strategy'] == strategy]\n            \n            analysis['strategy_performance'][strategy] = {\n                'mean_latency': strategy_data['total_latency'].mean(),\n                'std_latency': strategy_data['total_latency'].std(),\n                'mean_hit_rate': strategy_data['overall_hit_rate'].mean(),\n                'std_hit_rate': strategy_data['overall_hit_rate'].std(),\n                'configurations_tested': len(strategy_data)\n            }\n        \n        # Batch size scaling analysis\n        for batch_size in results_df['batch_size'].unique():\n            batch_data = results_df[results_df['batch_size'] == batch_size]\n            analysis['batch_size_scaling'][batch_size] = {\n                'mean_latency': batch_data['total_latency'].mean(),\n                'mean_hit_rate': batch_data['overall_hit_rate'].mean(),\n                'strategy_performance': batch_data.groupby('strategy')['total_latency'].mean().to_dict()\n            }\n        \n        # Statistical significance testing (pairwise strategy comparison)\n        strategies = results_df['strategy'].unique()\n        for i, strategy1 in enumerate(strategies):\n            for strategy2 in strategies[i+1:]:\n                data1 = results_df[results_df['strategy'] == strategy1]['total_latency']\n                data2 = results_df[results_df['strategy'] == strategy2]['total_latency']\n                \n                if len(data1) > 0 and len(data2) > 0:\n                    t_stat, p_value = stats.ttest_ind(data1, data2)\n                    effect_size = (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)\n                    \n                    analysis['statistical_significance'][f\"{strategy1}_vs_{strategy2}\"] = {\n                        't_statistic': t_stat,\n                        'p_value': p_value,\n                        'effect_size': effect_size,\n                        'significant': p_value < 0.05\n                    }\n        \n        # Save analysis\n        analysis_file = self.results_dir / 'statistical_analysis.json'\n        with open(analysis_file, 'w') as f:\n            json.dump(analysis, f, indent=2, default=str)\n        \n        return analysis\n    \n    def generate_summary_report(self, results_df: pd.DataFrame, analysis: Dict[str, Any]) -> str:\n        \"\"\"Generate comprehensive evaluation summary report\"\"\"\n        report = []\n        report.append(\"# Comprehensive MoE Expert Prefetching Evaluation Report\")\n        report.append(\"=\"*60)\n        report.append(\"\")\n        \n        # Configuration summary\n        report.append(\"## Evaluation Configuration\")\n        report.append(f\"- Models evaluated: {list(self.model_configs.keys())}\")\n        report.append(f\"- Batch sizes: {self.batch_sizes}\")\n        report.append(f\"- Sequence lengths: {self.sequence_lengths}\")\n        report.append(f\"- Cache sizes (MB): {self.cache_sizes_mb}\")\n        report.append(f\"- Replications per configuration: {self.num_replications}\")\n        report.append(f\"- Total experimental runs: {len(results_df)}\")\n        report.append(\"\")\n        \n        # Strategy performance ranking\n        report.append(\"## Strategy Performance Ranking (by Average Latency)\")\n        strategy_perf = sorted(\n            analysis['strategy_performance'].items(),\n            key=lambda x: x[1]['mean_latency']\n        )\n        \n        for i, (strategy, metrics) in enumerate(strategy_perf, 1):\n            report.append(f\"{i}. {strategy}: {metrics['mean_latency']:.2f}ms \"\n                         f\"(hit rate: {metrics['mean_hit_rate']:.3f})\")\n        report.append(\"\")\n        \n        # Batch size scaling insights\n        report.append(\"## Batch Size Scaling Analysis\")\n        for batch_size, metrics in analysis['batch_size_scaling'].items():\n            report.append(f\"Batch Size {batch_size}: {metrics['mean_latency']:.2f}ms \"\n                         f\"(hit rate: {metrics['mean_hit_rate']:.3f})\")\n        report.append(\"\")\n        \n        # Statistical significance highlights\n        report.append(\"## Key Statistical Findings\")\n        significant_comparisons = [\n            (comparison, stats) for comparison, stats in analysis['statistical_significance'].items()\n            if stats['significant']\n        ]\n        \n        report.append(f\"Found {len(significant_comparisons)} statistically significant strategy differences:\")\n        for comparison, stats in significant_comparisons[:5]:  # Top 5\n            report.append(f\"- {comparison}: p={stats['p_value']:.4f}, effect size={stats['effect_size']:.3f}\")\n        report.append(\"\")\n        \n        # Save report\n        report_content = \"\\n\".join(report)\n        report_file = self.results_dir / 'EVALUATION_SUMMARY.md'\n        with open(report_file, 'w') as f:\n            f.write(report_content)\n        \n        return report_content\n\n# Placeholder strategy adapters for existing strategies\nclass OnDemandStrategy:\n    def __init__(self, cache_framework):\n        self.cache = cache_framework\n    \n    def reset_strategy(self):\n        self.cache.reset_metrics()\n        self.cache.clear_cache()\n\nclass OracleStrategy:\n    def __init__(self, cache_framework, num_experts):\n        self.cache = cache_framework\n        self.num_experts = num_experts\n    \n    def reset_strategy(self):\n        self.cache.reset_metrics()\n        self.cache.clear_cache()\n\nclass TopKStrategy:\n    def __init__(self, cache_framework, num_experts):\n        self.cache = cache_framework\n        self.num_experts = num_experts\n    \n    def reset_strategy(self):\n        self.cache.reset_metrics()\n        self.cache.clear_cache()\n\nclass MultiLookAheadStrategy:\n    def __init__(self, cache_framework, num_experts):\n        self.cache = cache_framework\n        self.num_experts = num_experts\n    \n    def reset_strategy(self):\n        self.cache.reset_metrics()\n        self.cache.clear_cache()\n\nclass IntelligentStrategy:\n    def __init__(self, cache_framework, num_experts):\n        self.cache = cache_framework\n        self.num_experts = num_experts\n    \n    def reset_strategy(self):\n        self.cache.reset_metrics()\n        self.cache.clear_cache()\n\nif __name__ == \"__main__\":\n    # Example evaluation configuration\n    config = {\n        'batch_sizes': [1, 2, 4, 8, 16, 32],\n        'sequence_lengths': [512, 1024],\n        'cache_sizes_mb': [50, 100],\n        'num_replications': 3,\n        'results_dir': 'evalComparative/results'\n    }\n    \n    evaluator = MultiBatchEvaluator(config)\n    results_df = evaluator.run_comprehensive_evaluation()\n    analysis = evaluator.compute_statistical_analysis(results_df)\n    summary = evaluator.generate_summary_report(results_df, analysis)\n    \n    print(\"\\n\" + \"=\"*60)\n    print(\"EVALUATION COMPLETE\")\n    print(\"=\"*60)\n    print(summary)