#!/usr/bin/env python3

"""
Comparative plotting utilities for MoE expert prefetching evaluation results.

Provides publication-quality visualizations for comparing strategies across
different metrics, batch sizes, and hardware configurations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

def create_comprehensive_visualizations(results_df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Create comprehensive visualization suite for comparative evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    plot_files = {}
    
    # 1. Strategy performance comparison
    plot_files['strategy_comparison'] = create_strategy_comparison_plot(results_df, output_path)
    
    # 2. Batch size scaling analysis  
    plot_files['batch_scaling'] = create_batch_scaling_plot(results_df, output_path)
    
    # 3. Cache sensitivity heatmap
    plot_files['cache_sensitivity'] = create_cache_sensitivity_heatmap(results_df, output_path)
    
    # 4. Performance distribution analysis
    plot_files['performance_distributions'] = create_performance_distribution_plot(results_df, output_path)
    
    return plot_files

def create_strategy_comparison_plot(results_df: pd.DataFrame, output_path: Path) -> str:
    """Create comprehensive strategy comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Expert Prefetching Strategy Comparison', fontsize=16, fontweight='bold')
    
    # Latency comparison
    strategy_stats = results_df.groupby('strategy').agg({
        'total_latency': ['mean', 'std'],
        'overall_hit_rate': ['mean', 'std']
    }).round(3)
    
    ax1 = axes[0, 0]
    latency_means = strategy_stats['total_latency']['mean']
    latency_stds = strategy_stats['total_latency']['std']
    
    bars = ax1.bar(range(len(latency_means)), latency_means.values, 
                   yerr=latency_stds.values, capsize=5, alpha=0.8)
    ax1.set_title('Average Latency by Strategy')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_xticks(range(len(latency_means)))
    ax1.set_xticklabels(latency_means.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, latency_means.values, latency_stds.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Hit rate comparison
    ax2 = axes[0, 1]
    hitrate_means = strategy_stats['overall_hit_rate']['mean']
    hitrate_stds = strategy_stats['overall_hit_rate']['std']
    
    bars = ax2.bar(range(len(hitrate_means)), hitrate_means.values,
                   yerr=hitrate_stds.values, capsize=5, alpha=0.8, color='orange')
    ax2.set_title('Average Hit Rate by Strategy')
    ax2.set_ylabel('Hit Rate')
    ax2.set_xticks(range(len(hitrate_means)))
    ax2.set_xticklabels(hitrate_means.index, rotation=45, ha='right')
    ax2.set_ylim(0, 1.0)
    
    # Performance efficiency (latency vs hit rate)
    ax3 = axes[1, 0]
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        ax3.scatter(strategy_data['overall_hit_rate'], strategy_data['total_latency'], 
                   label=strategy, alpha=0.6, s=50)
    
    ax3.set_xlabel('Hit Rate')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Performance Efficiency (Hit Rate vs Latency)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Strategy ranking
    ax4 = axes[1, 1]
    # Calculate composite performance score (lower is better)
    composite_scores = []
    strategy_names = []
    
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        # Normalize latency and hit rate, then compute weighted score
        norm_latency = strategy_data['total_latency'].mean() / results_df['total_latency'].max()
        norm_hitrate = 1.0 - strategy_data['overall_hit_rate'].mean()  # Invert so lower is better
        
        composite_score = 0.6 * norm_latency + 0.4 * norm_hitrate
        composite_scores.append(composite_score)
        strategy_names.append(strategy)
    
    # Sort by performance score
    sorted_data = sorted(zip(strategy_names, composite_scores), key=lambda x: x[1])
    sorted_names, sorted_scores = zip(*sorted_data)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_names)))
    bars = ax4.barh(range(len(sorted_names)), sorted_scores, color=colors, alpha=0.8)
    ax4.set_yticks(range(len(sorted_names)))
    ax4.set_yticklabels(sorted_names)
    ax4.set_xlabel('Composite Performance Score (lower is better)')
    ax4.set_title('Strategy Performance Ranking')
    
    plt.tight_layout()
    
    # Save plots
    png_path = output_path / 'strategy_comparison.png'
    pdf_path = output_path / 'strategy_comparison.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    return str(png_path)

def create_batch_scaling_plot(results_df: pd.DataFrame, output_path: Path) -> str:
    """Create batch size scaling analysis plot"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Batch Size Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Latency scaling
    ax1 = axes[0]
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        batch_performance = strategy_data.groupby('batch_size')['total_latency'].mean()
        
        ax1.plot(batch_performance.index, batch_performance.values, 
                marker='o', linewidth=2, markersize=6, label=strategy)
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Average Latency (ms)')
    ax1.set_title('Latency Scaling with Batch Size')
    ax1.set_xscale('log', base=2)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Hit rate scaling
    ax2 = axes[1]
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        batch_hitrates = strategy_data.groupby('batch_size')['overall_hit_rate'].mean()
        
        ax2.plot(batch_hitrates.index, batch_hitrates.values,
                marker='s', linewidth=2, markersize=6, label=strategy)
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Average Hit Rate')
    ax2.set_title('Hit Rate Scaling with Batch Size')
    ax2.set_xscale('log', base=2)
    ax2.set_ylim(0, 1.0)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    png_path = output_path / 'batch_scaling.png'
    pdf_path = output_path / 'batch_scaling.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    return str(png_path)

def create_cache_sensitivity_heatmap(results_df: pd.DataFrame, output_path: Path) -> str:
    """Create cache size sensitivity heatmap"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Cache Size Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Latency heatmap
    latency_pivot = results_df.pivot_table(
        values='total_latency',
        index='strategy', 
        columns='cache_size_mb',
        aggfunc='mean'
    )
    
    sns.heatmap(latency_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=axes[0], cbar_kws={'label': 'Latency (ms)'})
    axes[0].set_title('Latency vs Cache Size')
    axes[0].set_ylabel('Strategy')
    axes[0].set_xlabel('Cache Size (MB)')
    
    # Hit rate heatmap
    hitrate_pivot = results_df.pivot_table(
        values='overall_hit_rate',
        index='strategy',
        columns='cache_size_mb', 
        aggfunc='mean'
    )
    
    sns.heatmap(hitrate_pivot, annot=True, fmt='.3f', cmap='YlGnBu',
                ax=axes[1], cbar_kws={'label': 'Hit Rate'})
    axes[1].set_title('Hit Rate vs Cache Size')
    axes[1].set_ylabel('Strategy') 
    axes[1].set_xlabel('Cache Size (MB)')
    
    plt.tight_layout()
    
    # Save plots
    png_path = output_path / 'cache_sensitivity.png'
    pdf_path = output_path / 'cache_sensitivity.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    return str(png_path)

def create_performance_distribution_plot(results_df: pd.DataFrame, output_path: Path) -> str:
    """Create performance distribution analysis plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Latency distributions
    ax1 = axes[0, 0]
    strategies_subset = results_df['strategy'].unique()[:6]  # Limit for clarity
    latency_data = [results_df[results_df['strategy'] == strategy]['total_latency'].values 
                   for strategy in strategies_subset]
    
    ax1.boxplot(latency_data, labels=strategies_subset)
    ax1.set_title('Latency Distribution by Strategy')
    ax1.set_ylabel('Latency (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Hit rate distributions
    ax2 = axes[0, 1]
    hitrate_data = [results_df[results_df['strategy'] == strategy]['overall_hit_rate'].values
                   for strategy in strategies_subset]
    
    ax2.boxplot(hitrate_data, labels=strategies_subset)
    ax2.set_title('Hit Rate Distribution by Strategy')
    ax2.set_ylabel('Hit Rate')
    ax2.tick_params(axis='x', rotation=45)
    
    # Violin plot for detailed distribution shape
    ax3 = axes[1, 0]
    strategy_latencies = []
    strategy_labels = []
    
    for strategy in strategies_subset:
        strategy_data = results_df[results_df['strategy'] == strategy]['total_latency']
        strategy_latencies.extend(strategy_data.values)
        strategy_labels.extend([strategy] * len(strategy_data))
    
    violin_df = pd.DataFrame({
        'Strategy': strategy_labels,
        'Latency': strategy_latencies
    })
    
    sns.violinplot(data=violin_df, x='Strategy', y='Latency', ax=ax3)
    ax3.set_title('Latency Distribution Shape Analysis')
    ax3.tick_params(axis='x', rotation=45)
    
    # Performance correlation analysis
    ax4 = axes[1, 1]
    ax4.scatter(results_df['overall_hit_rate'], results_df['total_latency'], 
               alpha=0.6, s=30)
    
    # Add correlation line
    correlation = np.corrcoef(results_df['overall_hit_rate'], results_df['total_latency'])[0, 1]
    z = np.polyfit(results_df['overall_hit_rate'], results_df['total_latency'], 1)
    p = np.poly1d(z)
    ax4.plot(results_df['overall_hit_rate'], p(results_df['overall_hit_rate']), 
            "r--", alpha=0.8, linewidth=2)
    
    ax4.set_xlabel('Hit Rate')
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title(f'Hit Rate vs Latency Correlation (r={correlation:.3f})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    png_path = output_path / 'performance_distributions.png'
    pdf_path = output_path / 'performance_distributions.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    return str(png_path)

if __name__ == "__main__":
    # Example usage with sample data
    import pandas as pd
    import numpy as np
    
    # Generate sample data for testing
    np.random.seed(42)
    strategies = ['on_demand', 'top_k', 'intelligent', 'pregated_moe', 'expertflow_plec']
    batch_sizes = [1, 4, 16, 64]
    cache_sizes = [50, 100, 200]
    
    sample_data = []
    for strategy in strategies:
        for batch_size in batch_sizes:
            for cache_size in cache_sizes:
                for rep in range(3):
                    # Simulate realistic performance differences
                    base_latency = np.random.uniform(100, 500)
                    if strategy == 'on_demand':
                        latency = base_latency * 2.0
                        hit_rate = 0.3
                    elif strategy == 'oracle':
                        latency = base_latency * 0.5
                        hit_rate = 1.0
                    elif strategy == 'pregated_moe':
                        latency = base_latency * 0.7
                        hit_rate = 0.85
                    elif strategy == 'expertflow_plec':
                        latency = base_latency * 0.8
                        hit_rate = 0.82
                    else:
                        latency = base_latency
                        hit_rate = 0.7
                    
                    sample_data.append({
                        'strategy': strategy,
                        'batch_size': batch_size,
                        'cache_size_mb': cache_size,
                        'replication': rep,
                        'total_latency': latency + np.random.normal(0, latency * 0.1),
                        'overall_hit_rate': max(0, min(1, hit_rate + np.random.normal(0, 0.05)))
                    })
    
    sample_df = pd.DataFrame(sample_data)
    
    # Test visualization creation
    plot_files = create_comprehensive_visualizations(sample_df, 'test_plots')
    print(f"Created test plots: {list(plot_files.keys())}")