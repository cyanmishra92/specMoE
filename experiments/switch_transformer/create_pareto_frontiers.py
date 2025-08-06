#!/usr/bin/env python3
"""
Pareto Frontier Analysis for Switch Transformer Expert Prefetching Strategies

This script generates three comprehensive Pareto frontier analyses:
1. Performance vs Memory Trade-off
2. Performance vs Implementation Complexity  
3. Memory Efficiency vs Cache Effectiveness

Each analysis helps identify optimal strategies under different constraints.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def compute_pareto_frontier(x, y, maximize_both=False):
    """
    Compute Pareto frontier for 2D optimization problem.
    
    Args:
        x: x-axis values 
        y: y-axis values
        maximize_both: If True, both objectives are maximized. If False, x minimized, y maximized.
    
    Returns:
        indices of points on Pareto frontier, sorted by x-value
    """
    points = np.array([x, y]).T
    n_points = len(points)
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if maximize_both:
                    # Both objectives maximized - point i dominated if both coords <= point j
                    if (points[i][0] <= points[j][0] and points[i][1] <= points[j][1] and
                        (points[i][0] < points[j][0] or points[i][1] < points[j][1])):
                        is_pareto[i] = False
                        break
                else:
                    # x minimized, y maximized - point i dominated if x >= and y <=
                    if (points[i][0] >= points[j][0] and points[i][1] <= points[j][1] and
                        (points[i][0] > points[j][0] or points[i][1] < points[j][1])):
                        is_pareto[i] = False
                        break
    
    pareto_indices = np.where(is_pareto)[0]
    # Sort by x-coordinate for plotting
    pareto_indices = pareto_indices[np.argsort(points[pareto_indices, 0])]
    return pareto_indices

def create_pareto_plot_1_performance_vs_memory():
    """Create Pareto frontier: Performance vs Memory Trade-off"""
    
    # Data from comprehensive_summary.csv (batch_size=1 for baseline comparison)
    data = {
        'strategy': ['A', 'B', 'C', 'D', 'E'],
        'strategy_name': ['On-Demand', 'Oracle', 'Multi-Look', 'Top-K', 'Intelligent'],
        'memory_mb': [28.0, 3584.0, 3584.0, 3584.0, 3584.0],
        'latency_ms': [2281.472, 143.939, 215.105, 205.686, 174.615],
        'speedup': [1.0, 15.85, 10.61, 11.09, 13.07]  # vs on-demand baseline
    }
    
    df = pd.DataFrame(data)
    
    # Compute Pareto frontier (minimize memory, maximize speedup)
    pareto_indices = compute_pareto_frontier(df['memory_mb'], df['speedup'], maximize_both=False)
    
    # Create figure with specific size for publication
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Define colors for each strategy
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    # Plot all points
    for _, row in df.iterrows():
        is_pareto = row.name in pareto_indices
        marker_size = 200 if is_pareto else 120
        marker_style = 's' if is_pareto else 'o'
        edge_width = 3 if is_pareto else 1.5
        
        ax.scatter(row['memory_mb'], row['speedup'], 
                  c=colors[row['strategy_name']], 
                  s=marker_size, marker=marker_style,
                  edgecolors='black', linewidth=edge_width,
                  alpha=0.8, zorder=5)
        
        # Add strategy labels
        offset_x = -200 if row['strategy_name'] == 'On-Demand' else 100
        offset_y = 0.3 if row['strategy_name'] != 'Multi-Look' else -0.5
        ax.annotate(row['strategy_name'], 
                   (row['memory_mb'], row['speedup']),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   ha='center' if row['strategy_name'] != 'On-Demand' else 'right')
    
    # Draw Pareto frontier
    pareto_points = df.iloc[pareto_indices]
    pareto_points_sorted = pareto_points.sort_values('memory_mb')
    
    # Connect Pareto points with lines
    ax.plot(pareto_points_sorted['memory_mb'], pareto_points_sorted['speedup'], 
            'r--', linewidth=3, alpha=0.7, label='Pareto Frontier', zorder=3)
    
    # Fill dominated region
    pareto_x = list(pareto_points_sorted['memory_mb'])
    pareto_y = list(pareto_points_sorted['speedup'])
    x_fill = [0, 0] + pareto_x + [4000, 4000]
    y_fill = [0, max(df['speedup'])] + pareto_y + [max(df['speedup']), 0]
    ax.fill(x_fill, y_fill, alpha=0.1, color='red', label='Dominated Region')
    
    # Styling
    ax.set_xlabel('Memory Usage (MB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Speedup (×)', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier: Performance vs Memory Trade-off\nExpert Prefetching Strategy Optimization', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(-200, 4000)
    ax.set_ylim(0, 17)
    
    # Add performance annotations
    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5)
    ax.text(2000, 10.3, '10× Performance Threshold', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.7))
    
    # Memory budget guidelines
    ax.axvline(x=1000, color='orange', linestyle=':', alpha=0.5)
    ax.text(1100, 8, '1GB Memory\nBudget', ha='left', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    
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

def create_pareto_plot_2_performance_vs_complexity():
    """Create Pareto frontier: Performance vs Implementation Complexity"""
    
    # Implementation complexity scores (1-10 scale, based on algorithm sophistication)
    complexity_scores = {
        'On-Demand': 1.0,    # Trivial - no prefetching
        'Oracle': 3.0,       # Medium - requires perfect prediction oracle
        'Multi-Look': 8.5,   # Very High - complex multi-step prediction
        'Top-K': 4.0,        # Medium-High - static top-k selection with routing analysis
        'Intelligent': 7.0   # High - adaptive learning with feedback loops
    }
    
    data = {
        'strategy': ['A', 'B', 'C', 'D', 'E'],
        'strategy_name': ['On-Demand', 'Oracle', 'Multi-Look', 'Top-K', 'Intelligent'],
        'complexity': [complexity_scores[name] for name in ['On-Demand', 'Oracle', 'Multi-Look', 'Top-K', 'Intelligent']],
        'speedup': [1.0, 15.85, 10.61, 11.09, 13.07],
        'latency_ms': [2281.472, 143.939, 215.105, 205.686, 174.615]
    }
    
    df = pd.DataFrame(data)
    
    # Compute Pareto frontier (minimize complexity, maximize speedup)
    pareto_indices = compute_pareto_frontier(df['complexity'], df['speedup'], maximize_both=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    # Plot all points with complexity-based sizing
    for _, row in df.iterrows():
        is_pareto = row.name in pareto_indices
        marker_size = 250 if is_pareto else 150
        marker_style = 's' if is_pareto else 'o'
        edge_width = 3 if is_pareto else 1.5
        
        ax.scatter(row['complexity'], row['speedup'], 
                  c=colors[row['strategy_name']], 
                  s=marker_size, marker=marker_style,
                  edgecolors='black', linewidth=edge_width,
                  alpha=0.8, zorder=5)
        
        # Add strategy labels with complexity scores
        offset_y = 0.5 if row['strategy_name'] not in ['Multi-Look', 'Oracle'] else -0.8
        ax.annotate(f"{row['strategy_name']}\n(Complexity: {row['complexity']:.1f})", 
                   (row['complexity'], row['speedup']),
                   xytext=(0, 25 if offset_y > 0 else -35), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   ha='center', va='bottom' if offset_y > 0 else 'top')
    
    # Draw Pareto frontier
    pareto_points = df.iloc[pareto_indices]
    pareto_points_sorted = pareto_points.sort_values('complexity')
    
    ax.plot(pareto_points_sorted['complexity'], pareto_points_sorted['speedup'], 
            'r--', linewidth=3, alpha=0.7, label='Pareto Frontier', zorder=3)
    
    # Fill dominated region
    x_max = max(df['complexity']) + 1
    y_max = max(df['speedup']) + 1
    
    # Create polygon for dominated region
    pareto_x = list(pareto_points_sorted['complexity'])
    pareto_y = list(pareto_points_sorted['speedup'])
    
    # Add boundary points for filling
    fill_points = [(0, 0)] + list(zip(pareto_x, pareto_y)) + [(x_max, pareto_y[-1]), (x_max, 0)]
    polygon = Polygon(fill_points, alpha=0.1, color='red', label='Dominated Region')
    ax.add_patch(polygon)
    
    # Styling
    ax.set_xlabel('Implementation Complexity Score (1-10)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Speedup (×)', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier: Performance vs Implementation Complexity\nEngineering Effort vs Performance Trade-off', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 17)
    
    # Add complexity zone annotations
    ax.axvspan(0, 3, alpha=0.1, color='green', label='Low Complexity')
    ax.axvspan(3, 6, alpha=0.1, color='yellow', label='Medium Complexity')  
    ax.axvspan(6, 10, alpha=0.1, color='red', label='High Complexity')
    
    ax.text(1.5, 16, 'Simple\nImplementation', ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
    ax.text(4.5, 16, 'Moderate\nImplementation', ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    ax.text(8, 16, 'Complex\nImplementation', ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
    
    # Performance threshold
    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5)
    ax.text(5, 10.5, '10× Performance Threshold', ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.7))
    
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

def create_pareto_plot_3_memory_efficiency_vs_cache():
    """Create Pareto frontier: Memory Efficiency vs Cache Effectiveness"""
    
    data = {
        'strategy': ['A', 'B', 'C', 'D', 'E'],
        'strategy_name': ['On-Demand', 'Oracle', 'Multi-Look', 'Top-K', 'Intelligent'],
        'memory_mb': [28.0, 3584.0, 3584.0, 3584.0, 3584.0],
        'experts_cached': [1.0, 138.0, 187.9, 158.7, 152.0],
        'cache_hit_rate': [0.0, 99.8, 99.0, 99.4, 99.4],  # Convert to percentage
        'memory_per_expert': [28.0, 26.0, 19.1, 22.6, 23.6]  # MB per expert
    }
    
    df = pd.DataFrame(data)
    
    # Compute Pareto frontier (minimize memory per expert, maximize cache hit rate)
    # Filter out On-Demand (no meaningful caching)
    df_cache = df[df['strategy_name'] != 'On-Demand'].copy()
    pareto_indices = compute_pareto_frontier(df_cache['memory_per_expert'], df_cache['cache_hit_rate'], maximize_both=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    colors = {'On-Demand': '#FF6B6B', 'Oracle': '#4ECDC4', 'Multi-Look': '#45B7D1', 
              'Top-K': '#96CEB4', 'Intelligent': '#FFEAA7'}
    
    # Plot caching strategies
    for idx, row in df_cache.iterrows():
        original_idx = idx - 1  # Adjust for filtered dataframe
        is_pareto = original_idx in pareto_indices
        marker_size = 250 if is_pareto else 150
        marker_style = 's' if is_pareto else 'o'
        edge_width = 3 if is_pareto else 1.5
        
        ax.scatter(row['memory_per_expert'], row['cache_hit_rate'], 
                  c=colors[row['strategy_name']], 
                  s=marker_size, marker=marker_style,
                  edgecolors='black', linewidth=edge_width,
                  alpha=0.8, zorder=5)
        
        # Add strategy labels with efficiency metrics
        offset_x = -30 if row['strategy_name'] == 'Multi-Look' else 30
        offset_y = 15 if row['strategy_name'] not in ['Oracle', 'Top-K'] else -25
        ax.annotate(f"{row['strategy_name']}\n({row['memory_per_expert']:.1f} MB/expert)", 
                   (row['memory_per_expert'], row['cache_hit_rate']),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   ha='center', va='center')
    
    # Plot On-Demand separately (special case)
    on_demand = df[df['strategy_name'] == 'On-Demand'].iloc[0]
    ax.scatter(on_demand['memory_per_expert'], on_demand['cache_hit_rate'], 
              c=colors['On-Demand'], s=200, marker='^',
              edgecolors='black', linewidth=2, alpha=0.8, zorder=5)
    ax.annotate('On-Demand\n(No Caching)', 
               (on_demand['memory_per_expert'], on_demand['cache_hit_rate']),
               xytext=(10, 20), textcoords='offset points',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8),
               ha='left', va='bottom')
    
    # Draw Pareto frontier for caching strategies
    pareto_points = df_cache.iloc[pareto_indices]
    pareto_points_sorted = pareto_points.sort_values('memory_per_expert')
    
    ax.plot(pareto_points_sorted['memory_per_expert'], pareto_points_sorted['cache_hit_rate'], 
            'r--', linewidth=3, alpha=0.7, label='Pareto Frontier', zorder=3)
    
    # Fill dominated region for caching strategies
    x_min, x_max = 18, 28
    y_min, y_max = 98, 100.5
    
    pareto_x = list(pareto_points_sorted['memory_per_expert'])
    pareto_y = list(pareto_points_sorted['cache_hit_rate'])
    
    # Create stepped boundary for dominated region
    fill_x = [x_max, x_max] + pareto_x + [pareto_x[-1]]
    fill_y = [y_min, y_max] + pareto_y + [y_max]
    ax.fill(fill_x, fill_y, alpha=0.1, color='red', label='Dominated Region')
    
    # Styling
    ax.set_xlabel('Memory Efficiency (MB per Expert Cached)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cache Hit Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier: Memory Efficiency vs Cache Effectiveness\nOptimal Cache Design Trade-offs', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 30)
    ax.set_ylim(-5, 101)
    
    # Add efficiency zones
    ax.axhline(y=99, color='green', linestyle=':', alpha=0.5)
    ax.text(15, 99.2, '99% Hit Rate Threshold', ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
    
    ax.axvline(x=25, color='orange', linestyle=':', alpha=0.5)
    ax.text(25.5, 50, 'Memory\nEfficiency\nThreshold', ha='left', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    
    # Efficiency annotations
    ax.text(21, 97, 'High Memory\nEfficiency Zone', ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
    
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

def main():
    """Generate all three Pareto frontier analyses"""
    
    # Create output directory
    output_dir = Path('/data/research/specMoE/specMoE/evalSwitchB8/results/pareto_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating Pareto Frontier Analysis...")
    
    # Generate Plot 1: Performance vs Memory
    print("1. Creating Performance vs Memory Pareto frontier...")
    fig1 = create_pareto_plot_1_performance_vs_memory()
    fig1.savefig(output_dir / '07_pareto_performance_vs_memory.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig(output_dir / '07_pareto_performance_vs_memory.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Generate Plot 2: Performance vs Complexity
    print("2. Creating Performance vs Complexity Pareto frontier...")
    fig2 = create_pareto_plot_2_performance_vs_complexity()
    fig2.savefig(output_dir / '08_pareto_performance_vs_complexity.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(output_dir / '08_pareto_performance_vs_complexity.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    # Generate Plot 3: Memory Efficiency vs Cache Effectiveness
    print("3. Creating Memory Efficiency vs Cache Effectiveness Pareto frontier...")
    fig3 = create_pareto_plot_3_memory_efficiency_vs_cache()
    fig3.savefig(output_dir / '09_pareto_memory_efficiency_vs_cache.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig3.savefig(output_dir / '09_pareto_memory_efficiency_vs_cache.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    
    print(f"\n✅ All Pareto frontier plots saved to: {output_dir}")
    print("Files created:")
    print("- 07_pareto_performance_vs_memory.pdf/png")
    print("- 08_pareto_performance_vs_complexity.pdf/png") 
    print("- 09_pareto_memory_efficiency_vs_cache.pdf/png")

if __name__ == "__main__":
    main()