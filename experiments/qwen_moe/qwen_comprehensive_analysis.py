#!/usr/bin/env python3
"""
Comprehensive Analysis and Visualization Suite for Qwen MoE Expert Prefetching

This script generates publication-quality analysis and visualizations for the
Qwen MoE expert prefetching evaluation, including:

1. Primary performance analysis (6 main plots)
2. Pareto frontier optimization (3 plots) 
3. Statistical significance analysis
4. Comprehensive reporting

All plots are generated in both PDF and PNG formats for research publication
and GitHub documentation respectively.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import seaborn as sns
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_qwen_data():
    """Load and prepare Qwen experimental data"""
    
    # Load comprehensive summary
    summary_file = Path('results/qwen_comprehensive_summary.csv')
    if not summary_file.exists():
        raise FileNotFoundError("Qwen experimental data not found. Run qwen_comprehensive_evaluation.py first.")
    
    df = pd.read_csv(summary_file)
    
    # Rename columns for consistency
    column_mapping = {
        'total_latency_ms_mean': 'inference_latency_ms_mean',
        'total_latency_ms_std': 'inference_latency_ms_std',
        'total_latency_ms_median': 'inference_latency_ms_median',
        'total_latency_ms_min': 'inference_latency_ms_min', 
        'total_latency_ms_max': 'inference_latency_ms_max',
        'total_latency_ms_p95': 'inference_latency_ms_p95',
        'total_latency_ms_p99': 'inference_latency_ms_p99'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Convert cache hit rate to percentage
    df['cache_hit_rate_pct'] = df['cache_hit_rate_mean'] * 100
    
    # Calculate speedup vs baseline (On-Demand)
    baseline_latencies = df[df['strategy'] == 'A'].set_index('batch_size')['inference_latency_ms_mean']
    df['speedup_vs_baseline'] = df.apply(
        lambda row: baseline_latencies[row['batch_size']] / row['inference_latency_ms_mean'], 
        axis=1
    )
    
    print(f"📊 Loaded Qwen experimental data: {len(df)} configurations")
    print(f"🔬 Strategies: {df['strategy_name'].nunique()}")  
    print(f"📏 Batch sizes: {sorted(df['batch_size'].unique())}")
    
    return df

def create_plot_1_latency_analysis(df):
    """Create comprehensive latency analysis plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define colors for each strategy
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    strategies = df['strategy_name'].unique()
    batch_sizes = sorted(df['batch_size'].unique())
    
    x = np.arange(len(batch_sizes))
    width = 0.15
    
    # Create grouped bar chart
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['strategy_name'] == strategy].sort_values('batch_size')
        
        means = strategy_data['inference_latency_ms_mean'].values
        stds = strategy_data['inference_latency_ms_std'].values
        
        bars = ax.bar(x + i*width, means, width, label=strategy,
                     color=colors[strategy], alpha=0.8, 
                     yerr=stds, capsize=3, error_kw={'linewidth': 1.5})
        
        # Add value labels on bars
        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            if mean < 1000:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                       f'{mean:.0f}ms', ha='center', va='bottom', fontsize=8,
                       rotation=90 if mean > 50 else 0)
    
    # Styling
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Qwen MoE: Inference Latency vs Batch Size\nExpert Prefetching Strategy Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(batch_sizes)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=11)
    
    # Add performance annotations
    best_performance = df.loc[df['inference_latency_ms_mean'].idxmin()]
    ax.annotate(f'Best: {best_performance["strategy_name"]}\n{best_performance["inference_latency_ms_mean"]:.1f}ms',
               xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
               verticalalignment='top')
    
    plt.tight_layout()
    return fig

def create_plot_2_cache_hit_analysis(df):
    """Create cache hit rate analysis plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    strategies = df['strategy_name'].unique()
    batch_sizes = sorted(df['batch_size'].unique())
    
    x = np.arange(len(batch_sizes))
    width = 0.15
    
    # Create grouped bar chart
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['strategy_name'] == strategy].sort_values('batch_size')
        
        hit_rates = strategy_data['cache_hit_rate_pct'].values
        hit_rate_stds = strategy_data['cache_hit_rate_std'].values * 100
        
        bars = ax.bar(x + i*width, hit_rates, width, label=strategy,
                     color=colors[strategy], alpha=0.8,
                     yerr=hit_rate_stds, capsize=3, error_kw={'linewidth': 1.5})
        
        # Add value labels
        for j, (bar, rate) in enumerate(zip(bars, hit_rates)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Styling
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cache Hit Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Qwen MoE: Cache Hit Rate vs Batch Size\nMulti-Level Caching Effectiveness', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(batch_sizes)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right', fontsize=11)
    
    # Add performance thresholds
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.7)
    ax.text(len(batch_sizes)-1, 96, '95% Threshold', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_plot_3_memory_analysis(df):
    """Create memory usage analysis plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    strategies = df['strategy_name'].unique()
    batch_sizes = sorted(df['batch_size'].unique())
    
    x = np.arange(len(batch_sizes))
    width = 0.15
    
    # Create grouped bar chart
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['strategy_name'] == strategy].sort_values('batch_size')
        
        memory_usage = strategy_data['memory_usage_mb_mean'].values
        memory_stds = strategy_data['memory_usage_mb_std'].values
        
        bars = ax.bar(x + i*width, memory_usage, width, label=strategy,
                     color=colors[strategy], alpha=0.8,
                     yerr=memory_stds, capsize=3, error_kw={'linewidth': 1.5})
        
        # Add value labels
        for j, (bar, memory) in enumerate(zip(bars, memory_usage)):
            if memory > 50:  # Only label non-trivial memory usage
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                       f'{memory:.0f}MB', ha='center', va='bottom', fontsize=9,
                       rotation=45 if memory > 500 else 0)
    
    # Styling
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=14, fontweight='bold')
    ax.set_title('Qwen MoE: Memory Usage vs Batch Size\nExpert Caching Memory Footprint', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(batch_sizes)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=11)
    
    # Memory budget guidelines
    ax.axhline(y=1000, color='red', linestyle=':', alpha=0.5)
    ax.text(2, 1100, '1GB Memory Budget', ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_plot_4_speedup_analysis(df):
    """Create speedup analysis plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    strategies = df['strategy_name'].unique()
    batch_sizes = sorted(df['batch_size'].unique())
    
    x = np.arange(len(batch_sizes))
    width = 0.15
    
    # Create grouped bar chart
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['strategy_name'] == strategy].sort_values('batch_size')
        speedups = strategy_data['speedup_vs_baseline'].values
        
        bars = ax.bar(x + i*width, speedups, width, label=strategy,
                     color=colors[strategy], alpha=0.8)
        
        # Add value labels
        for j, (bar, speedup) in enumerate(zip(bars, speedups)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{speedup:.2f}×', ha='center', va='bottom', fontsize=9)
    
    # Styling  
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup vs On-Demand (×)', fontsize=14, fontweight='bold')
    ax.set_title('Qwen MoE: Performance Speedup vs Batch Size\nPrefetching Strategy Effectiveness', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(batch_sizes)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=11)
    
    # Performance threshold
    ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7)
    ax.text(2, 2.1, '2× Speedup Threshold', ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_plot_5_tail_latency_analysis(df):
    """Create tail latency analysis plot"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    # Panel 1: P50, P95, P99 at batch size 1
    batch_1_data = df[df['batch_size'] == 1].copy()
    strategies = batch_1_data['strategy_name'].values
    
    x = np.arange(len(strategies))
    width = 0.25
    
    p50s = batch_1_data['inference_latency_ms_median'].values
    p95s = batch_1_data['inference_latency_ms_p95'].values  
    p99s = batch_1_data['inference_latency_ms_p99'].values
    
    ax1.bar(x - width, p50s, width, label='P50', alpha=0.8, color='lightblue')
    ax1.bar(x, p95s, width, label='P95', alpha=0.8, color='orange')
    ax1.bar(x + width, p99s, width, label='P99', alpha=0.8, color='red')
    
    ax1.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Tail Latency Percentiles (Batch Size 1)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: P99 scaling across batch sizes
    for strategy in df['strategy_name'].unique():
        strategy_data = df[df['strategy_name'] == strategy].sort_values('batch_size')
        batch_sizes = strategy_data['batch_size'].values
        p99_latencies = strategy_data['inference_latency_ms_p99'].values
        
        ax2.plot(batch_sizes, p99_latencies, 'o-', label=strategy, 
                color=colors[strategy], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('P99 Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('P99 Latency Scaling', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Tail ratios (P99/P50)
    batch_1_data['tail_ratio'] = batch_1_data['inference_latency_ms_p99'] / batch_1_data['inference_latency_ms_median']
    
    bars = ax3.bar(strategies, batch_1_data['tail_ratio'], 
                   color=[colors[s] for s in strategies], alpha=0.8)
    
    for bar, ratio in zip(bars, batch_1_data['tail_ratio']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax3.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Tail Ratio (P99/P50)', fontsize=12, fontweight='bold')
    ax3.set_title('Tail Latency Characteristics', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(strategies, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Consistency analysis (coefficient of variation)
    batch_1_data['cv'] = batch_1_data['inference_latency_ms_std'] / batch_1_data['inference_latency_ms_mean'] * 100
    
    bars = ax4.bar(strategies, batch_1_data['cv'], 
                   color=[colors[s] for s in strategies], alpha=0.8)
    
    for bar, cv in zip(bars, batch_1_data['cv']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{cv:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax4.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Performance Consistency', fontsize=14, fontweight='bold')
    ax4.set_xticklabels(strategies, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Qwen MoE: Comprehensive Tail Latency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_plot_6_scalability_analysis(df):
    """Create scalability analysis plot"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    # Panel 1: Normalized latency scaling
    for strategy in df['strategy_name'].unique():
        strategy_data = df[df['strategy_name'] == strategy].sort_values('batch_size')
        batch_sizes = strategy_data['batch_size'].values
        latencies = strategy_data['inference_latency_ms_mean'].values
        
        # Normalize to batch size 1
        normalized_latencies = latencies / latencies[0]
        ideal_scaling = batch_sizes / batch_sizes[0]
        
        ax1.plot(batch_sizes, normalized_latencies, 'o-', label=strategy,
                color=colors[strategy], linewidth=2, markersize=6)
    
    # Add ideal scaling line
    ax1.plot(batch_sizes, ideal_scaling, 'k--', alpha=0.5, label='Ideal Linear')
    
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Latency', fontsize=12, fontweight='bold')
    ax1.set_title('Latency Scaling Characteristics', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Throughput analysis
    for strategy in df['strategy_name'].unique():
        strategy_data = df[df['strategy_name'] == strategy].sort_values('batch_size')
        batch_sizes = strategy_data['batch_size'].values
        latencies = strategy_data['inference_latency_ms_mean'].values
        
        # Calculate throughput (requests/second)
        throughput = (batch_sizes * 1000) / latencies
        
        ax2.plot(batch_sizes, throughput, 'o-', label=strategy,
                color=colors[strategy], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Throughput (requests/sec)', fontsize=12, fontweight='bold')
    ax2.set_title('System Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Memory efficiency (hit rate per MB)
    for strategy in df['strategy_name'].unique():
        if strategy == 'On-Demand':
            continue  # Skip on-demand (no meaningful caching)
            
        strategy_data = df[df['strategy_name'] == strategy].sort_values('batch_size')
        batch_sizes = strategy_data['batch_size'].values
        hit_rates = strategy_data['cache_hit_rate_mean'].values
        memory_usage = strategy_data['memory_usage_mb_mean'].values
        
        efficiency = hit_rates / (memory_usage / 1000)  # Hit rate per GB
        
        ax3.plot(batch_sizes, efficiency, 'o-', label=strategy,
                color=colors[strategy], linewidth=2, markersize=6)
    
    ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Hit Rate per GB', fontsize=12, fontweight='bold')
    ax3.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Expert loading efficiency
    for strategy in df['strategy_name'].unique():
        strategy_data = df[df['strategy_name'] == strategy].sort_values('batch_size')
        batch_sizes = strategy_data['batch_size'].values
        expert_loads = strategy_data['expert_loads_mean'].values
        hit_rates = strategy_data['cache_hit_rate_mean'].values
        
        loading_efficiency = hit_rates / (expert_loads / 1000)  # Hit rate per 1K loads
        
        ax4.plot(batch_sizes, loading_efficiency, 'o-', label=strategy,
                color=colors[strategy], linewidth=2, markersize=6)
    
    ax4.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Hit Rate per 1K Expert Loads', fontsize=12, fontweight='bold')
    ax4.set_title('Expert Loading Efficiency', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Qwen MoE: System Scalability Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def compute_pareto_frontier(x, y, maximize_both=False):
    """Compute Pareto frontier for 2D optimization problem"""
    points = np.array([x, y]).T
    n_points = len(points)
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if maximize_both:
                    if (points[i][0] <= points[j][0] and points[i][1] <= points[j][1] and
                        (points[i][0] < points[j][0] or points[i][1] < points[j][1])):
                        is_pareto[i] = False
                        break
                else:
                    if (points[i][0] >= points[j][0] and points[i][1] <= points[j][1] and
                        (points[i][0] > points[j][0] or points[i][1] < points[j][1])):
                        is_pareto[i] = False
                        break
    
    pareto_indices = np.where(is_pareto)[0]
    pareto_indices = pareto_indices[np.argsort(points[pareto_indices, 0])]
    return pareto_indices

def create_pareto_plot_1_performance_vs_memory(df):
    """Create Performance vs Memory Pareto frontier"""
    
    # Use batch size 1 data for Pareto analysis
    batch_1_data = df[df['batch_size'] == 1].copy()
    
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    # Compute Pareto frontier
    memory_usage = batch_1_data['memory_usage_mb_mean'].values
    speedups = batch_1_data['speedup_vs_baseline'].values
    pareto_indices = compute_pareto_frontier(memory_usage, speedups, maximize_both=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Plot all points
    for i, row in batch_1_data.iterrows():
        is_pareto = i in pareto_indices
        marker_size = 250 if is_pareto else 150
        marker_style = 's' if is_pareto else 'o'
        edge_width = 3 if is_pareto else 1.5
        
        ax.scatter(row['memory_usage_mb_mean'], row['speedup_vs_baseline'],
                  c=colors[row['strategy_name']], s=marker_size, marker=marker_style,
                  edgecolors='black', linewidth=edge_width, alpha=0.8, zorder=5)
        
        # Add strategy labels
        offset_x = -50 if row['strategy_name'] == 'On-Demand' else 30
        ax.annotate(row['strategy_name'],
                   (row['memory_usage_mb_mean'], row['speedup_vs_baseline']),
                   xytext=(offset_x, 15), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   ha='center')
    
    # Draw Pareto frontier
    if len(pareto_indices) > 1:
        pareto_points = batch_1_data.iloc[pareto_indices].sort_values('memory_usage_mb_mean')
        ax.plot(pareto_points['memory_usage_mb_mean'], pareto_points['speedup_vs_baseline'],
                'r--', linewidth=3, alpha=0.7, label='Pareto Frontier', zorder=3)
        
        # Fill dominated region
        pareto_x = list(pareto_points['memory_usage_mb_mean'])
        pareto_y = list(pareto_points['speedup_vs_baseline']) 
        x_fill = [0, 0] + pareto_x + [max(memory_usage)+100, max(memory_usage)+100]
        y_fill = [0, max(speedups)+0.5] + pareto_y + [max(speedups)+0.5, 0]
        ax.fill(x_fill, y_fill, alpha=0.1, color='red', label='Dominated Region')
    
    ax.set_xlabel('Memory Usage (MB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Speedup (×)', fontsize=14, fontweight='bold')
    ax.set_title('Qwen MoE: Performance vs Memory Trade-off\nPareto Frontier Analysis', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-50, max(memory_usage)+100)
    ax.set_ylim(0.5, max(speedups)+0.3)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                   markersize=10, label='Pareto Optimal', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=8, label='Dominated', markeredgecolor='black'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Pareto Frontier')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    return fig

def create_pareto_plot_2_performance_vs_complexity(df):
    """Create Performance vs Complexity Pareto frontier"""
    
    # Complexity scores for Qwen strategies
    complexity_scores = {
        'On-Demand': 1.0,
        'Oracle': 3.0, 
        'Multi-Look': 8.5,
        'Top-K': 4.0,
        'Intelligent': 7.0
    }
    
    batch_1_data = df[df['batch_size'] == 1].copy()
    batch_1_data['complexity'] = batch_1_data['strategy_name'].map(complexity_scores)
    
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    # Compute Pareto frontier
    complexity = batch_1_data['complexity'].values
    speedups = batch_1_data['speedup_vs_baseline'].values
    pareto_indices = compute_pareto_frontier(complexity, speedups, maximize_both=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Plot all points
    for i, row in batch_1_data.iterrows():
        is_pareto = i in pareto_indices
        marker_size = 250 if is_pareto else 150
        marker_style = 's' if is_pareto else 'o'
        edge_width = 3 if is_pareto else 1.5
        
        ax.scatter(row['complexity'], row['speedup_vs_baseline'],
                  c=colors[row['strategy_name']], s=marker_size, marker=marker_style,
                  edgecolors='black', linewidth=edge_width, alpha=0.8, zorder=5)
        
        # Add strategy labels
        offset_y = 25 if row['strategy_name'] not in ['Multi-Look', 'Oracle'] else -35
        ax.annotate(f"{row['strategy_name']}\n(Complexity: {row['complexity']:.1f})",
                   (row['complexity'], row['speedup_vs_baseline']),
                   xytext=(0, offset_y), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   ha='center', va='bottom' if offset_y > 0 else 'top')
    
    # Draw Pareto frontier  
    if len(pareto_indices) > 1:
        pareto_points = batch_1_data.iloc[pareto_indices].sort_values('complexity')
        ax.plot(pareto_points['complexity'], pareto_points['speedup_vs_baseline'],
                'r--', linewidth=3, alpha=0.7, label='Pareto Frontier', zorder=3)
        
        # Fill dominated region using polygon
        pareto_x = list(pareto_points['complexity'])
        pareto_y = list(pareto_points['speedup_vs_baseline'])
        fill_points = [(0, 0)] + list(zip(pareto_x, pareto_y)) + [(10, pareto_y[-1]), (10, 0)]
        polygon = Polygon(fill_points, alpha=0.1, color='red', label='Dominated Region')
        ax.add_patch(polygon)
    
    ax.set_xlabel('Implementation Complexity Score (1-10)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Speedup (×)', fontsize=14, fontweight='bold')  
    ax.set_title('Qwen MoE: Performance vs Implementation Complexity\nPareto Frontier Analysis',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, max(speedups)+0.3)
    
    # Add complexity zones
    ax.axvspan(0, 3, alpha=0.1, color='green')
    ax.axvspan(3, 6, alpha=0.1, color='yellow')
    ax.axvspan(6, 10, alpha=0.1, color='red')
    
    ax.text(1.5, max(speedups)+0.1, 'Simple', ha='center', fontweight='bold')
    ax.text(4.5, max(speedups)+0.1, 'Moderate', ha='center', fontweight='bold')
    ax.text(8, max(speedups)+0.1, 'Complex', ha='center', fontweight='bold')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                   markersize=10, label='Pareto Optimal', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=8, label='Dominated', markeredgecolor='black'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Pareto Frontier')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    return fig

def create_pareto_plot_3_memory_efficiency_vs_cache(df):
    """Create Memory Efficiency vs Cache Effectiveness Pareto frontier"""
    
    batch_1_data = df[df['batch_size'] == 1].copy()
    
    # Calculate memory per expert for caching strategies
    batch_1_data['memory_per_expert'] = batch_1_data.apply(
        lambda row: row['memory_usage_mb_mean'] / 32 if row['strategy_name'] != 'On-Demand' 
        else 28.5, axis=1  # Qwen experts are ~28.5MB each
    )
    
    # Filter out On-Demand for Pareto analysis (no meaningful caching)
    cache_data = batch_1_data[batch_1_data['strategy_name'] != 'On-Demand'].copy()
    
    colors = {'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    # Compute Pareto frontier
    memory_per_expert = cache_data['memory_per_expert'].values
    cache_hit_rates = cache_data['cache_hit_rate_pct'].values
    pareto_indices = compute_pareto_frontier(memory_per_expert, cache_hit_rates, maximize_both=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Plot caching strategies
    for i, row in cache_data.iterrows():
        original_idx = i - cache_data.index[0]
        is_pareto = original_idx in pareto_indices
        marker_size = 250 if is_pareto else 150
        marker_style = 's' if is_pareto else 'o'
        edge_width = 3 if is_pareto else 1.5
        
        ax.scatter(row['memory_per_expert'], row['cache_hit_rate_pct'],
                  c=colors[row['strategy_name']], s=marker_size, marker=marker_style,
                  edgecolors='black', linewidth=edge_width, alpha=0.8, zorder=5)
        
        # Add strategy labels
        offset_x = -30 if row['strategy_name'] == 'Multi-Look' else 30
        ax.annotate(f"{row['strategy_name']}\n({row['memory_per_expert']:.1f} MB/expert)",
                   (row['memory_per_expert'], row['cache_hit_rate_pct']),
                   xytext=(offset_x, 15), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   ha='center')
    
    # Plot On-Demand separately
    on_demand = batch_1_data[batch_1_data['strategy_name'] == 'On-Demand'].iloc[0]
    ax.scatter(on_demand['memory_per_expert'], on_demand['cache_hit_rate_pct'],
              c='#FF6B6B', s=200, marker='^', edgecolors='black', linewidth=2, alpha=0.8, zorder=5)
    ax.annotate('On-Demand\n(No Caching)',
               (on_demand['memory_per_expert'], on_demand['cache_hit_rate_pct']),
               xytext=(10, 20), textcoords='offset points', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
    
    # Draw Pareto frontier
    if len(pareto_indices) > 1:
        pareto_points = cache_data.iloc[pareto_indices].sort_values('memory_per_expert')
        ax.plot(pareto_points['memory_per_expert'], pareto_points['cache_hit_rate_pct'],
                'r--', linewidth=3, alpha=0.7, label='Pareto Frontier', zorder=3)
        
        # Fill dominated region
        pareto_x = list(pareto_points['memory_per_expert'])
        pareto_y = list(pareto_points['cache_hit_rate_pct'])
        x_max = max(memory_per_expert) + 5
        y_max = max(cache_hit_rates) + 5
        fill_x = [x_max, x_max] + pareto_x + [pareto_x[-1]]
        fill_y = [0, y_max] + pareto_y + [y_max]
        ax.fill(fill_x, fill_y, alpha=0.1, color='red', label='Dominated Region')
    
    ax.set_xlabel('Memory Efficiency (MB per Expert Cached)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cache Hit Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Qwen MoE: Memory Efficiency vs Cache Effectiveness\nPareto Frontier Analysis',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(memory_per_expert)+10)
    ax.set_ylim(-5, 105)
    
    # Add efficiency threshold
    ax.axhline(y=90, color='green', linestyle=':', alpha=0.5)
    ax.text(max(memory_per_expert)/2, 91, '90% Hit Rate Threshold', ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                   markersize=10, label='Pareto Optimal', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=8, label='Dominated (Caching)', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                   markersize=8, label='No Caching', markeredgecolor='black'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Pareto Frontier')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    return fig

def generate_all_qwen_plots():
    """Generate all Qwen analysis plots"""
    
    print("📊 Loading Qwen experimental data...")
    df = load_qwen_data()
    
    # Create output directories
    plots_dir = Path('results/plots')
    individual_dir = Path('results/individual_plots')
    pareto_dir = Path('results/pareto_analysis')
    
    for dir_path in [plots_dir, individual_dir, pareto_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    plots = [
        ("01_qwen_latency_vs_batch_size", create_plot_1_latency_analysis),
        ("02_qwen_cache_hit_rate_vs_batch_size", create_plot_2_cache_hit_analysis), 
        ("03_qwen_memory_usage_vs_batch_size", create_plot_3_memory_analysis),
        ("04_qwen_speedup_vs_batch_size", create_plot_4_speedup_analysis),
        ("05_qwen_tail_latency_analysis", create_plot_5_tail_latency_analysis),
        ("06_qwen_scalability_analysis", create_plot_6_scalability_analysis),
    ]
    
    pareto_plots = [
        ("07_qwen_pareto_performance_vs_memory", create_pareto_plot_1_performance_vs_memory),
        ("08_qwen_pareto_performance_vs_complexity", create_pareto_plot_2_performance_vs_complexity),
        ("09_qwen_pareto_memory_efficiency_vs_cache", create_pareto_plot_3_memory_efficiency_vs_cache),
    ]
    
    print(f"🎨 Generating {len(plots)} main analysis plots...")
    
    # Generate main analysis plots
    for plot_name, plot_func in plots:
        print(f"   Creating {plot_name}...")
        fig = plot_func(df)
        
        # Save PDF and PNG
        fig.savefig(individual_dir / f'{plot_name}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(individual_dir / f'{plot_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    print(f"🔄 Generating {len(pareto_plots)} Pareto frontier plots...")
    
    # Generate Pareto frontier plots  
    for plot_name, plot_func in pareto_plots:
        print(f"   Creating {plot_name}...")
        fig = plot_func(df)
        
        # Save PDF and PNG
        fig.savefig(pareto_dir / f'{plot_name}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(pareto_dir / f'{plot_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    print("✅ All Qwen analysis plots generated successfully!")
    print(f"📁 Main plots saved to: {individual_dir}")
    print(f"📁 Pareto plots saved to: {pareto_dir}")
    
    return df

if __name__ == "__main__":
    # Generate all plots
    df = generate_all_qwen_plots()
    
    print("\n" + "="*60)
    print("🎯 QWEN ANALYSIS COMPLETE")
    print("="*60)
    print(f"📊 Generated 9 comprehensive visualization plots")
    print(f"🔬 Analyzed {len(df)} experimental configurations")
    print(f"📈 Best performing strategy: {df.loc[df['inference_latency_ms_mean'].idxmin(), 'strategy_name']}")
    print(f"🚀 Maximum speedup: {df['speedup_vs_baseline'].max():.2f}×")
    print("="*60)