#!/usr/bin/env python3
"""
Analyze Switch Transformer Prefetching Results

Statistical analysis of the 5×5 experiment matrix with visualization and reporting.
"""

import pickle
import json
import sys
from pathlib import Path

# Add scripts directory to path for ExperimentResult import
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from experiment_types import ExperimentResult
except ImportError:
    from run_single_experiment import ExperimentResult
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ResultsAnalyzer:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_data = self._load_all_results()
        self.strategies = ["A", "B", "C", "D", "E"]
        self.batch_sizes = [1, 2, 4, 8, 16]
        
        print(f"📊 Loaded results from {len(self.results_data)} experiments")
    
    def _load_all_results(self) -> Dict:
        """Load all experiment results"""
        results = {}
        
        for strategy in ["A", "B", "C", "D", "E"]:
            results[strategy] = {}
            for batch_size in [1, 2, 4, 8, 16]:
                file_path = self.results_dir / f"strategy_{strategy}_batch_{batch_size}.pkl"
                
                if file_path.exists():
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        results[strategy][batch_size] = data
                        print(f"✅ Loaded {strategy}_{batch_size}: {len(data)} runs")
                    except Exception as e:
                        print(f"❌ Failed to load {file_path}: {e}")
                        results[strategy][batch_size] = None
                else:
                    print(f"⚠️  Missing: {file_path}")
                    results[strategy][batch_size] = None
        
        return results
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame with means and standard deviations"""
        summary_data = []
        
        for strategy in self.strategies:
            for batch_size in self.batch_sizes:
                data = self.results_data[strategy].get(batch_size)
                
                if data is None:
                    # Missing data
                    summary_data.append({
                        'strategy': strategy,
                        'batch_size': batch_size,
                        'latency_mean': np.nan,
                        'latency_std': np.nan,
                        'hit_rate_mean': np.nan,
                        'hit_rate_std': np.nan,
                        'memory_usage_mean': np.nan,
                        'memory_usage_std': np.nan,
                        'prefetch_accuracy_mean': np.nan,
                        'prefetch_accuracy_std': np.nan,
                        'experts_loaded_mean': np.nan,
                        'experts_loaded_std': np.nan,
                        'num_runs': 0
                    })
                else:
                    # Extract metrics
                    latencies = [r.inference_latency_ms for r in data]
                    hit_rates = [r.cache_hit_rate for r in data]
                    memory_usage = [r.memory_usage_mb for r in data]
                    prefetch_accuracy = [r.prefetch_accuracy for r in data]
                    experts_loaded = [r.total_experts_loaded for r in data]
                    
                    summary_data.append({
                        'strategy': strategy,
                        'batch_size': batch_size,
                        'latency_mean': np.mean(latencies),
                        'latency_std': np.std(latencies),
                        'hit_rate_mean': np.mean(hit_rates),
                        'hit_rate_std': np.std(hit_rates),
                        'memory_usage_mean': np.mean(memory_usage),
                        'memory_usage_std': np.std(memory_usage),
                        'prefetch_accuracy_mean': np.mean(prefetch_accuracy),
                        'prefetch_accuracy_std': np.std(prefetch_accuracy),
                        'experts_loaded_mean': np.mean(experts_loaded),
                        'experts_loaded_std': np.std(experts_loaded),
                        'num_runs': len(data)
                    })
        
        return pd.DataFrame(summary_data)
    
    def generate_performance_plots(self, output_dir: Path):
        """Generate comprehensive performance plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.create_summary_dataframe()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Latency vs Batch Size
        ax1 = axes[0, 0]
        for strategy in self.strategies:
            strategy_data = df[df['strategy'] == strategy]
            if not strategy_data.empty:
                ax1.errorbar(strategy_data['batch_size'], 
                           strategy_data['latency_mean'],
                           yerr=strategy_data['latency_std'],
                           marker='o', label=f'Strategy {strategy}', 
                           linewidth=2, markersize=6)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Inference Latency (ms)')
        ax1.set_title('Inference Latency vs Batch Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Plot 2: Hit Rate vs Batch Size
        ax2 = axes[0, 1]
        for strategy in self.strategies:
            strategy_data = df[df['strategy'] == strategy]
            if not strategy_data.empty:
                ax2.errorbar(strategy_data['batch_size'],
                           strategy_data['hit_rate_mean'],
                           yerr=strategy_data['hit_rate_std'],
                           marker='s', label=f'Strategy {strategy}',
                           linewidth=2, markersize=6)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Cache Hit Rate')
        ax2.set_title('Cache Hit Rate vs Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # Plot 3: Memory Usage vs Batch Size
        ax3 = axes[0, 2]
        for strategy in self.strategies:
            strategy_data = df[df['strategy'] == strategy]
            if not strategy_data.empty:
                ax3.errorbar(strategy_data['batch_size'],
                           strategy_data['memory_usage_mean'],
                           yerr=strategy_data['memory_usage_std'],
                           marker='^', label=f'Strategy {strategy}',
                           linewidth=2, markersize=6)
        
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage vs Batch Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        
        # Plot 4: Strategy Comparison (Batch Size 1)
        ax4 = axes[1, 0]
        batch1_data = df[df['batch_size'] == 1]
        strategies = batch1_data['strategy'].tolist()
        latencies = batch1_data['latency_mean'].tolist()
        errors = batch1_data['latency_std'].tolist()
        
        bars = ax4.bar(strategies, latencies, yerr=errors, capsize=5, alpha=0.7)
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('Inference Latency (ms)')
        ax4.set_title('Strategy Comparison (Batch Size 1)')
        ax4.grid(True, alpha=0.3)
        
        # Color bars by performance
        for i, bar in enumerate(bars):
            if i == 0:  # Baseline
                bar.set_color('red')
            elif latencies[i] < latencies[0]:  # Better than baseline
                bar.set_color('green')
            else:  # Worse than baseline
                bar.set_color('orange')
        
        # Plot 5: Prefetch Accuracy vs Strategy
        ax5 = axes[1, 1]
        prefetch_strategies = ['B', 'C', 'D', 'E']  # A doesn't have prefetching
        prefetch_data = df[(df['strategy'].isin(prefetch_strategies)) & (df['batch_size'] == 1)]
        
        if not prefetch_data.empty:
            ax5.bar(prefetch_data['strategy'], 
                   prefetch_data['prefetch_accuracy_mean'],
                   yerr=prefetch_data['prefetch_accuracy_std'],
                   capsize=5, alpha=0.7)
        
        ax5.set_xlabel('Strategy')
        ax5.set_ylabel('Prefetch Accuracy')
        ax5.set_title('Prefetch Accuracy by Strategy')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)
        
        # Plot 6: Efficiency Analysis (Latency vs Memory)
        ax6 = axes[1, 2]
        for strategy in self.strategies:
            strategy_data = df[df['strategy'] == strategy]
            if not strategy_data.empty and not strategy_data['memory_usage_mean'].isna().all():
                ax6.scatter(strategy_data['memory_usage_mean'],
                          strategy_data['latency_mean'],
                          s=strategy_data['batch_size'] * 20,  # Size by batch size
                          label=f'Strategy {strategy}',
                          alpha=0.7)
        
        ax6.set_xlabel('Memory Usage (MB)')
        ax6.set_ylabel('Inference Latency (ms)')
        ax6.set_title('Efficiency: Latency vs Memory Usage')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'switch_prefetching_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance plots saved to: {output_dir}/switch_prefetching_analysis.png")
    
    def statistical_significance_test(self) -> Dict:
        """Perform statistical significance tests between strategies"""
        results = {}
        
        for batch_size in self.batch_sizes:
            batch_results = {}
            
            # Get data for this batch size
            strategies_data = {}
            for strategy in self.strategies:
                data = self.results_data[strategy].get(batch_size)
                if data:
                    strategies_data[strategy] = [r.inference_latency_ms for r in data]
            
            # Perform pairwise t-tests
            for i, strategy1 in enumerate(self.strategies):
                for strategy2 in self.strategies[i+1:]:
                    if strategy1 in strategies_data and strategy2 in strategies_data:
                        # Two-sample t-test
                        stat, p_value = stats.ttest_ind(
                            strategies_data[strategy1],
                            strategies_data[strategy2]
                        )
                        
                        batch_results[f"{strategy1}_vs_{strategy2}"] = {
                            "statistic": float(stat),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                            "effect_size": abs(np.mean(strategies_data[strategy1]) - 
                                             np.mean(strategies_data[strategy2]))
                        }
            
            results[f"batch_{batch_size}"] = batch_results
        
        return results
    
    def generate_insights(self) -> Dict:
        """Generate key insights from the analysis"""
        df = self.create_summary_dataframe()
        
        insights = {
            "best_overall_strategy": None,
            "best_low_latency_strategy": None,
            "best_memory_efficient_strategy": None,
            "scaling_behavior": {},
            "prefetching_effectiveness": {},
            "recommendations": []
        }
        
        # Find best overall strategy (lowest mean latency across all batch sizes)
        strategy_means = df.groupby('strategy')['latency_mean'].mean()
        best_strategy = strategy_means.idxmin()
        insights["best_overall_strategy"] = {
            "strategy": best_strategy,
            "mean_latency": float(strategy_means[best_strategy])
        }
        
        # Best for single request (batch size 1)
        batch1 = df[df['batch_size'] == 1]
        if not batch1.empty:
            best_single = batch1.loc[batch1['latency_mean'].idxmin()]
            insights["best_low_latency_strategy"] = {
                "strategy": best_single['strategy'],
                "latency": float(best_single['latency_mean']),
                "std": float(best_single['latency_std'])
            }
        
        # Memory efficiency analysis
        df['memory_efficiency'] = df['hit_rate_mean'] / (df['memory_usage_mean'] / 1000)  # Hit rate per GB
        best_efficient = df.loc[df['memory_efficiency'].idxmax()]
        insights["best_memory_efficient_strategy"] = {
            "strategy": best_efficient['strategy'],
            "batch_size": int(best_efficient['batch_size']),
            "efficiency": float(best_efficient['memory_efficiency'])
        }
        
        # Scaling behavior
        for strategy in self.strategies:
            strategy_data = df[df['strategy'] == strategy].sort_values('batch_size')
            if len(strategy_data) > 1:
                # Calculate scaling factor
                latency_1 = strategy_data[strategy_data['batch_size'] == 1]['latency_mean'].iloc[0]
                latency_16 = strategy_data[strategy_data['batch_size'] == 16]['latency_mean'].iloc[0]
                
                if not np.isnan(latency_1) and not np.isnan(latency_16):
                    scaling_factor = latency_16 / latency_1
                    insights["scaling_behavior"][strategy] = {
                        "batch_1_latency": float(latency_1),
                        "batch_16_latency": float(latency_16),
                        "scaling_factor": float(scaling_factor)
                    }
        
        # Prefetching effectiveness
        baseline_data = df[df['strategy'] == 'A']  # On-demand baseline
        for strategy in ['B', 'C', 'D', 'E']:
            strategy_data = df[df['strategy'] == strategy]
            
            # Compare with baseline for each batch size
            improvements = []
            for batch_size in self.batch_sizes:
                baseline_latency = baseline_data[baseline_data['batch_size'] == batch_size]['latency_mean']
                strategy_latency = strategy_data[strategy_data['batch_size'] == batch_size]['latency_mean']
                
                if not baseline_latency.empty and not strategy_latency.empty:
                    improvement = (baseline_latency.iloc[0] - strategy_latency.iloc[0]) / baseline_latency.iloc[0]
                    improvements.append(improvement)
            
            if improvements:
                insights["prefetching_effectiveness"][strategy] = {
                    "average_improvement": float(np.mean(improvements)),
                    "best_improvement": float(max(improvements)),
                    "worst_improvement": float(min(improvements))
                }
        
        # Generate recommendations
        recommendations = []
        
        if insights["best_low_latency_strategy"]:
            best = insights["best_low_latency_strategy"]
            recommendations.append(
                f"For lowest latency: Use Strategy {best['strategy']} "
                f"({best['latency']:.2f}ms ± {best['std']:.2f}ms)"
            )
        
        if insights["best_overall_strategy"]:
            best = insights["best_overall_strategy"]
            recommendations.append(
                f"Best overall performance: Strategy {best['strategy']} "
                f"(mean latency {best['mean_latency']:.2f}ms)"
            )
        
        # Prefetching recommendation
        best_prefetch = max(insights["prefetching_effectiveness"].items(), 
                           key=lambda x: x[1]["average_improvement"], 
                           default=(None, None))
        if best_prefetch[0]:
            recommendations.append(
                f"Best prefetching strategy: {best_prefetch[0]} "
                f"({best_prefetch[1]['average_improvement']:.1%} average improvement)"
            )
        
        insights["recommendations"] = recommendations
        
        return insights
    
    def save_analysis(self, output_file: Path):
        """Save complete analysis results"""
        # Create summary DataFrame
        df = self.create_summary_dataframe()
        
        # Statistical tests
        stat_tests = self.statistical_significance_test()
        
        # Generate insights
        insights = self.generate_insights()
        
        # Compile analysis
        analysis = {
            "summary_statistics": df.to_dict('records'),
            "statistical_tests": stat_tests,
            "insights": insights,
            "metadata": {
                "total_experiments": len([d for strategy_data in self.results_data.values() 
                                        for d in strategy_data.values() if d is not None]),
                "strategies_analyzed": self.strategies,
                "batch_sizes_analyzed": self.batch_sizes
            }
        }
        
        # Save to JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"📁 Analysis saved to: {output_file}")
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze Switch Prefetching Results")
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory with experiment results")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for analysis (default: input_dir/analysis.json)")
    parser.add_argument("--plots", action="store_true",
                       help="Generate performance plots")
    
    args = parser.parse_args()
    
    # Set up paths
    input_dir = Path(args.input)
    output_file = Path(args.output) if args.output else input_dir / "analysis.json"
    plots_dir = input_dir / "plots"
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(input_dir)
    
    # Run analysis
    analysis = analyzer.save_analysis(output_file)
    
    # Generate plots if requested
    if args.plots:
        analyzer.generate_performance_plots(plots_dir)
    
    # Print key insights
    print(f"\n🎯 KEY INSIGHTS:")
    for recommendation in analysis["insights"]["recommendations"]:
        print(f"   • {recommendation}")
    
    print(f"\n✅ Analysis completed!")
    print(f"   Results: {output_file}")
    if args.plots:
        print(f"   Plots: {plots_dir}")

if __name__ == "__main__":
    main()