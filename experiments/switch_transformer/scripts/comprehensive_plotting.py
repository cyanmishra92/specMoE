#!/usr/bin/env python3
"""
Comprehensive Plotting and Tail Latency Analysis for Switch Prefetching Results

Generates publication-ready plots as PDFs:
- Performance comparison plots
- Tail latency analysis (P50, P95, P99)
- Scalability analysis
- Memory efficiency plots
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.backends.backend_pdf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for publication quality
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42  # True Type fonts for better PDF compatibility
})

class ComprehensivePlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.strategies = ["A", "B", "C", "D", "E"]
        self.batch_sizes = [1, 2, 4, 8, 16]
        self.strategy_names = {
            "A": "On-Demand",
            "B": "Oracle", 
            "C": "Multi-Look",
            "D": "Top-K",
            "E": "Intelligent"
        }
        self.strategy_colors = {
            "A": "#E74C3C",  # Red (baseline)
            "B": "#2ECC71",  # Green (oracle)
            "C": "#3498DB",  # Blue (multi-look)
            "D": "#F39C12",  # Orange (top-k)
            "E": "#9B59B6"   # Purple (intelligent)
        }
        
        # Load all data
        self.df_detailed, self.df_summary = self.load_all_data()
        
        print(f"📊 Loaded data: {len(self.df_detailed)} individual results")
        print(f"📊 Summary: {len(self.df_summary)} strategy-batch combinations")
    
    def load_all_data(self):
        """Load all results and create detailed and summary DataFrames"""
        all_data = []
        
        for strategy in self.strategies:
            for batch_size in self.batch_sizes:
                json_file = self.results_dir / f"strategy_{strategy}_batch_{batch_size}.json"
                
                if json_file.exists():
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                        
                        for run in data:
                            all_data.append({
                                'strategy': strategy,
                                'strategy_name': self.strategy_names[strategy],
                                'batch_size': run['batch_size'],
                                'run_id': run['run_id'],
                                'inference_latency_ms': run['inference_latency_ms'],
                                'memory_usage_mb': run['memory_usage_mb'],
                                'cache_hit_rate': run['cache_hit_rate'],
                                'prefetch_accuracy': run['prefetch_accuracy'],
                                'total_experts_loaded': run['total_experts_loaded'],
                                'cache_misses': run['cache_misses']
                            })
                    
                    except Exception as e:
                        print(f"❌ Failed to load {json_file}: {e}")
        
        df_detailed = pd.DataFrame(all_data)
        
        # Create summary statistics
        df_summary = df_detailed.groupby(['strategy', 'strategy_name', 'batch_size']).agg({
            'inference_latency_ms': ['mean', 'std', 'median', 'min', 'max', 
                                   lambda x: np.percentile(x, 95), lambda x: np.percentile(x, 99)],
            'memory_usage_mb': ['mean', 'std'],
            'cache_hit_rate': ['mean', 'std'],
            'prefetch_accuracy': ['mean', 'std'],
            'total_experts_loaded': ['mean', 'std'],
            'cache_misses': ['mean', 'std']
        }).round(3)
        
        # Flatten column names and rename percentiles
        df_summary.columns = ['_'.join(col) if col[1] != '<lambda>' else col[0] + '_p' + ('95' if 'lambda_0' in str(col) else '99') 
                             for col in df_summary.columns.values]
        df_summary.columns = [col.replace('<lambda_0>', 'p95').replace('<lambda_1>', 'p99') for col in df_summary.columns]
        
        df_summary = df_summary.reset_index()
        
        return df_detailed, df_summary
    
    def plot_latency_comparison(self, pdf):
        """Plot latency comparison across strategies and batch sizes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Latency vs Batch Size (log scale)
        for strategy in self.strategies:
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy]
            if not strategy_data.empty:
                ax1.errorbar(strategy_data['batch_size'], 
                           strategy_data['inference_latency_ms_mean'],
                           yerr=strategy_data['inference_latency_ms_std'],
                           marker='o', label=f'{self.strategy_names[strategy]}',
                           linewidth=2.5, markersize=8, capsize=4,
                           color=self.strategy_colors[strategy])
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Inference Latency (ms)')
        ax1.set_title('Inference Latency vs Batch Size')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        
        # Plot 2: Strategy Comparison at Batch Size 1
        batch1_data = self.df_summary[self.df_summary['batch_size'] == 1].sort_values('inference_latency_ms_mean')
        
        bars = ax2.bar(range(len(batch1_data)), batch1_data['inference_latency_ms_mean'],
                      yerr=batch1_data['inference_latency_ms_std'], capsize=4,
                      color=[self.strategy_colors[s] for s in batch1_data['strategy']],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Inference Latency (ms)')
        ax2.set_title('Strategy Comparison (Batch Size = 1)')
        ax2.set_xticks(range(len(batch1_data)))
        ax2.set_xticklabels(batch1_data['strategy_name'], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val, err) in enumerate(zip(bars, batch1_data['inference_latency_ms_mean'], 
                                               batch1_data['inference_latency_ms_std'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 50,
                    f'{val:.0f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Switch Transformer Prefetching Performance Analysis', fontsize=18, y=1.02)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def plot_tail_latency_analysis(self, pdf):
        """Plot comprehensive tail latency analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Percentile comparison at batch size 1
        ax1 = axes[0, 0]
        batch1 = self.df_summary[self.df_summary['batch_size'] == 1]
        
        x = np.arange(len(batch1))
        width = 0.15
        
        metrics = ['inference_latency_ms_mean', 'inference_latency_ms_median', 
                  'inference_latency_ms_p95', 'inference_latency_ms_p99']
        labels = ['Mean', 'P50 (Median)', 'P95', 'P99']
        colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
        
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            values = batch1[metric].values
            ax1.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)
        
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Tail Latency Analysis (Batch Size = 1)')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(batch1['strategy_name'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
        
        # Plot 2: P99 latency across batch sizes
        ax2 = axes[0, 1]
        for strategy in self.strategies:
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy]
            if not strategy_data.empty:
                ax2.plot(strategy_data['batch_size'], strategy_data['inference_latency_ms_p99'],
                        marker='o', label=f'{self.strategy_names[strategy]}',
                        linewidth=2.5, markersize=8, color=self.strategy_colors[strategy])
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('P99 Latency (ms)')
        ax2.set_title('P99 Tail Latency vs Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        
        # Plot 3: Latency distribution for each strategy (batch size 1)
        ax3 = axes[1, 0]
        
        # Create violin plots for latency distributions
        batch1_detailed = self.df_detailed[self.df_detailed['batch_size'] == 1]
        strategy_order = batch1_detailed.groupby('strategy_name')['inference_latency_ms'].mean().sort_values().index
        
        violin_parts = ax3.violinplot([batch1_detailed[batch1_detailed['strategy_name'] == s]['inference_latency_ms'].values 
                                      for s in strategy_order],
                                     positions=range(len(strategy_order)),
                                     showmeans=True, showmedians=True, showextrema=True)
        
        # Color the violins
        for i, (pc, strategy) in enumerate(zip(violin_parts['bodies'], strategy_order)):
            strategy_key = [k for k, v in self.strategy_names.items() if v == strategy][0]
            pc.set_facecolor(self.strategy_colors[strategy_key])
            pc.set_alpha(0.7)
        
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Latency Distribution by Strategy (Batch Size = 1)')
        ax3.set_xticks(range(len(strategy_order)))
        ax3.set_xticklabels(strategy_order, rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Tail ratio analysis (P99/P50)
        ax4 = axes[1, 1]
        
        # Calculate tail ratios
        tail_ratios = []
        for _, row in self.df_summary.iterrows():
            if row['inference_latency_ms_median'] > 0:
                ratio = row['inference_latency_ms_p99'] / row['inference_latency_ms_median']
                tail_ratios.append({
                    'strategy': row['strategy'],
                    'strategy_name': row['strategy_name'],
                    'batch_size': row['batch_size'],
                    'tail_ratio': ratio
                })
        
        tail_df = pd.DataFrame(tail_ratios)
        
        for strategy in self.strategies:
            strategy_data = tail_df[tail_df['strategy'] == strategy]
            if not strategy_data.empty:
                ax4.plot(strategy_data['batch_size'], strategy_data['tail_ratio'],
                        marker='s', label=f'{self.strategy_names[strategy]}',
                        linewidth=2.5, markersize=8, color=self.strategy_colors[strategy])
        
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Tail Ratio (P99/P50)')
        ax4.set_title('Latency Tail Behavior vs Batch Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        
        plt.suptitle('Comprehensive Tail Latency Analysis', fontsize=18, y=0.98)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def plot_cache_performance(self, pdf):
        """Plot cache hit rates and memory efficiency"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Cache hit rates vs batch size
        ax1 = axes[0, 0]
        for strategy in self.strategies:
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy]
            if not strategy_data.empty:
                ax1.errorbar(strategy_data['batch_size'], 
                           strategy_data['cache_hit_rate_mean'] * 100,
                           yerr=strategy_data['cache_hit_rate_std'] * 100,
                           marker='o', label=f'{self.strategy_names[strategy]}',
                           linewidth=2.5, markersize=8, capsize=4,
                           color=self.strategy_colors[strategy])
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Cache Hit Rate (%)')
        ax1.set_title('Cache Hit Rate vs Batch Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Plot 2: Memory usage vs batch size
        ax2 = axes[0, 1]
        for strategy in self.strategies:
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy]
            if not strategy_data.empty:
                ax2.errorbar(strategy_data['batch_size'], 
                           strategy_data['memory_usage_mb_mean'],
                           yerr=strategy_data['memory_usage_mb_std'],
                           marker='s', label=f'{self.strategy_names[strategy]}',
                           linewidth=2.5, markersize=8, capsize=4,
                           color=self.strategy_colors[strategy])
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # Plot 3: Prefetch accuracy for applicable strategies
        ax3 = axes[1, 0]
        prefetch_strategies = ['B', 'C', 'D', 'E']
        for strategy in prefetch_strategies:
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy]
            if not strategy_data.empty and not strategy_data['prefetch_accuracy_mean'].isna().all():
                ax3.errorbar(strategy_data['batch_size'], 
                           strategy_data['prefetch_accuracy_mean'] * 100,
                           yerr=strategy_data['prefetch_accuracy_std'] * 100,
                           marker='^', label=f'{self.strategy_names[strategy]}',
                           linewidth=2.5, markersize=8, capsize=4,
                           color=self.strategy_colors[strategy])
        
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Prefetch Accuracy (%)')
        ax3.set_title('Prefetch Accuracy vs Batch Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        ax3.set_ylim(0, 100)
        
        # Plot 4: Memory efficiency (hit rate per MB)
        ax4 = axes[1, 1]
        
        # Calculate memory efficiency
        efficiency_data = []
        for _, row in self.df_summary.iterrows():
            if row['memory_usage_mb_mean'] > 0:
                efficiency = row['cache_hit_rate_mean'] / (row['memory_usage_mb_mean'] / 1000)  # Hit rate per GB
                efficiency_data.append({
                    'strategy': row['strategy'],
                    'strategy_name': row['strategy_name'],
                    'batch_size': row['batch_size'],
                    'efficiency': efficiency
                })
        
        eff_df = pd.DataFrame(efficiency_data)
        
        for strategy in self.strategies:
            strategy_data = eff_df[eff_df['strategy'] == strategy]
            if not strategy_data.empty:
                ax4.plot(strategy_data['batch_size'], strategy_data['efficiency'],
                        marker='d', label=f'{self.strategy_names[strategy]}',
                        linewidth=2.5, markersize=8, color=self.strategy_colors[strategy])
        
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Memory Efficiency (Hit Rate / GB)')
        ax4.set_title('Memory Efficiency vs Batch Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        
        plt.suptitle('Cache Performance and Memory Analysis', fontsize=18, y=0.98)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def plot_scalability_analysis(self, pdf):
        """Plot scalability and efficiency analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Normalized latency (relative to batch size 1)
        ax1 = axes[0, 0]
        
        for strategy in self.strategies:
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if len(strategy_data) > 1:
                baseline = strategy_data[strategy_data['batch_size'] == 1]['inference_latency_ms_mean'].iloc[0]
                normalized = strategy_data['inference_latency_ms_mean'] / baseline
                
                ax1.plot(strategy_data['batch_size'], normalized,
                        marker='o', label=f'{self.strategy_names[strategy]}',
                        linewidth=2.5, markersize=8, color=self.strategy_colors[strategy])
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Normalized Latency (vs Batch=1)')
        ax1.set_title('Latency Scalability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Add ideal scaling line
        batch_sizes = [1, 2, 4, 8, 16]
        ax1.plot(batch_sizes, batch_sizes, 'k--', alpha=0.5, label='Linear Scaling')
        ax1.legend()
        
        # Plot 2: Throughput analysis (batch_size / latency)
        ax2 = axes[0, 1]
        
        for strategy in self.strategies:
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy]
            if not strategy_data.empty:
                throughput = strategy_data['batch_size'] / (strategy_data['inference_latency_ms_mean'] / 1000)  # requests/second
                
                ax2.plot(strategy_data['batch_size'], throughput,
                        marker='s', label=f'{self.strategy_names[strategy]}',
                        linewidth=2.5, markersize=8, color=self.strategy_colors[strategy])
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (requests/second)')
        ax2.set_title('Throughput vs Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        
        # Plot 3: Speedup vs baseline (Strategy A)
        ax3 = axes[1, 0]
        
        baseline_data = self.df_summary[self.df_summary['strategy'] == 'A']
        
        for strategy in ['B', 'C', 'D', 'E']:  # Skip baseline
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy]
            
            speedups = []
            batch_sizes_with_data = []
            
            for batch_size in self.batch_sizes:
                baseline_latency = baseline_data[baseline_data['batch_size'] == batch_size]['inference_latency_ms_mean']
                strategy_latency = strategy_data[strategy_data['batch_size'] == batch_size]['inference_latency_ms_mean']
                
                if not baseline_latency.empty and not strategy_latency.empty:
                    speedup = baseline_latency.iloc[0] / strategy_latency.iloc[0]
                    speedups.append(speedup)
                    batch_sizes_with_data.append(batch_size)
            
            if speedups:
                ax3.plot(batch_sizes_with_data, speedups,
                        marker='^', label=f'{self.strategy_names[strategy]}',
                        linewidth=2.5, markersize=8, color=self.strategy_colors[strategy])
        
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (On-Demand)')
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Speedup vs Baseline')
        ax3.set_title('Performance Improvement vs On-Demand')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        
        # Plot 4: Experts loaded efficiency
        ax4 = axes[1, 1]
        
        for strategy in self.strategies:
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy]
            if not strategy_data.empty:
                # Efficiency: cache hits per expert loaded
                efficiency = strategy_data['cache_hit_rate_mean'] * 100 / (strategy_data['total_experts_loaded_mean'] + 1e-6)
                
                ax4.plot(strategy_data['batch_size'], efficiency,
                        marker='d', label=f'{self.strategy_names[strategy]}',
                        linewidth=2.5, markersize=8, color=self.strategy_colors[strategy])
        
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Cache Hit % per Expert Loaded')
        ax4.set_title('Expert Loading Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        
        plt.suptitle('Scalability and Efficiency Analysis', fontsize=18, y=0.98)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def plot_statistical_analysis(self, pdf):
        """Plot statistical significance and confidence intervals"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Confidence intervals (batch size 1)
        ax1 = axes[0, 0]
        batch1 = self.df_summary[self.df_summary['batch_size'] == 1].sort_values('inference_latency_ms_mean')
        
        # Calculate 95% confidence intervals
        y_pos = np.arange(len(batch1))
        means = batch1['inference_latency_ms_mean'].values
        stds = batch1['inference_latency_ms_std'].values
        ci = 1.96 * stds / np.sqrt(10)  # 95% CI for n=10 runs
        
        colors = [self.strategy_colors[s] for s in batch1['strategy']]
        
        ax1.barh(y_pos, means, xerr=ci, color=colors, alpha=0.8, capsize=4)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(batch1['strategy_name'])
        ax1.set_xlabel('Inference Latency (ms)')
        ax1.set_title('95% Confidence Intervals (Batch Size = 1)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Coefficient of Variation
        ax2 = axes[0, 1]
        
        cv_data = []
        for strategy in self.strategies:
            for batch_size in self.batch_sizes:
                strategy_batch = self.df_summary[
                    (self.df_summary['strategy'] == strategy) & 
                    (self.df_summary['batch_size'] == batch_size)
                ]
                if not strategy_batch.empty:
                    mean_val = strategy_batch['inference_latency_ms_mean'].iloc[0]
                    std_val = strategy_batch['inference_latency_ms_std'].iloc[0]
                    cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
                    
                    cv_data.append({
                        'strategy': strategy,
                        'strategy_name': self.strategy_names[strategy],
                        'batch_size': batch_size,
                        'cv': cv
                    })
        
        cv_df = pd.DataFrame(cv_data)
        
        for strategy in self.strategies:
            strategy_data = cv_df[cv_df['strategy'] == strategy]
            if not strategy_data.empty:
                ax2.plot(strategy_data['batch_size'], strategy_data['cv'],
                        marker='o', label=f'{self.strategy_names[strategy]}',
                        linewidth=2.5, markersize=8, color=self.strategy_colors[strategy])
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Coefficient of Variation (%)')
        ax2.set_title('Latency Variability (CV)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # Plot 3: Effect sizes (Cohen's d) vs baseline
        ax3 = axes[1, 0]
        
        baseline_detailed = self.df_detailed[
            (self.df_detailed['strategy'] == 'A') & 
            (self.df_detailed['batch_size'] == 1)
        ]['inference_latency_ms']
        
        effect_sizes = []
        
        for strategy in ['B', 'C', 'D', 'E']:
            strategy_detailed = self.df_detailed[
                (self.df_detailed['strategy'] == strategy) & 
                (self.df_detailed['batch_size'] == 1)
            ]['inference_latency_ms']
            
            if not strategy_detailed.empty:
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(baseline_detailed) - 1) * baseline_detailed.var() + 
                                    (len(strategy_detailed) - 1) * strategy_detailed.var()) / 
                                   (len(baseline_detailed) + len(strategy_detailed) - 2))
                
                cohens_d = (baseline_detailed.mean() - strategy_detailed.mean()) / pooled_std
                
                effect_sizes.append({
                    'strategy': strategy,
                    'strategy_name': self.strategy_names[strategy],
                    'effect_size': cohens_d
                })
        
        if effect_sizes:
            effect_df = pd.DataFrame(effect_sizes)
            colors = [self.strategy_colors[s] for s in effect_df['strategy']]
            
            bars = ax3.bar(range(len(effect_df)), effect_df['effect_size'], 
                          color=colors, alpha=0.8, edgecolor='black')
            
            ax3.set_xlabel('Strategy')
            ax3.set_ylabel("Cohen's d (Effect Size)")
            ax3.set_title('Effect Size vs Baseline (Batch Size = 1)')
            ax3.set_xticks(range(len(effect_df)))
            ax3.set_xticklabels(effect_df['strategy_name'], rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add effect size interpretation lines
            ax3.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small Effect')
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect')
            ax3.legend()
        
        # Plot 4: Performance consistency (range analysis)
        ax4 = axes[1, 1]
        
        range_data = []
        for strategy in self.strategies:
            for batch_size in self.batch_sizes:
                strategy_batch = self.df_summary[
                    (self.df_summary['strategy'] == strategy) & 
                    (self.df_summary['batch_size'] == batch_size)
                ]
                if not strategy_batch.empty:
                    min_val = strategy_batch['inference_latency_ms_min'].iloc[0]
                    max_val = strategy_batch['inference_latency_ms_max'].iloc[0]
                    range_val = max_val - min_val
                    
                    range_data.append({
                        'strategy': strategy,
                        'strategy_name': self.strategy_names[strategy],
                        'batch_size': batch_size,
                        'range': range_val
                    })
        
        range_df = pd.DataFrame(range_data)
        
        for strategy in self.strategies:
            strategy_data = range_df[range_df['strategy'] == strategy]
            if not strategy_data.empty:
                ax4.plot(strategy_data['batch_size'], strategy_data['range'],
                        marker='s', label=f'{self.strategy_names[strategy]}',
                        linewidth=2.5, markersize=8, color=self.strategy_colors[strategy])
        
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Latency Range (Max - Min, ms)')
        ax4.set_title('Performance Consistency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        
        plt.suptitle('Statistical Analysis and Significance', fontsize=18, y=0.98)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def generate_all_plots(self, output_file: Path):
        """Generate all plots and save to PDF"""
        print(f"📊 Generating comprehensive plots...")
        
        with matplotlib.backends.backend_pdf.PdfPages(output_file) as pdf:
            # Page 1: Main latency comparison
            print("   📈 Latency comparison plots...")
            self.plot_latency_comparison(pdf)
            
            # Page 2: Tail latency analysis
            print("   📈 Tail latency analysis...")
            self.plot_tail_latency_analysis(pdf)
            
            # Page 3: Cache performance
            print("   📈 Cache performance plots...")
            self.plot_cache_performance(pdf)
            
            # Page 4: Scalability analysis
            print("   📈 Scalability analysis...")
            self.plot_scalability_analysis(pdf)
            
            # Page 5: Statistical analysis
            print("   📈 Statistical analysis...")
            self.plot_statistical_analysis(pdf)
            
            # Set PDF metadata
            d = pdf.infodict()
            d['Title'] = 'Switch Transformer Prefetching Analysis'
            d['Author'] = 'MoE Research Team'
            d['Subject'] = 'Performance Analysis and Tail Latency Study'
            d['Keywords'] = 'Switch Transformer, Prefetching, Tail Latency, Performance'
            d['Creator'] = 'Python matplotlib'
        
        print(f"✅ All plots saved to: {output_file}")

def main():
    results_dir = Path("../results")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("🚀 Comprehensive Plotting and Tail Latency Analysis")
    print("=" * 55)
    
    # Initialize plotter
    plotter = ComprehensivePlotter(results_dir)
    
    if plotter.df_detailed.empty:
        print("❌ No data found! Make sure experiments have been run.")
        return
    
    # Generate comprehensive PDF report
    pdf_file = plots_dir / "switch_prefetching_comprehensive_analysis.pdf"
    plotter.generate_all_plots(pdf_file)
    
    # Also save summary data
    summary_file = results_dir / "comprehensive_summary.csv"
    plotter.df_summary.to_csv(summary_file, index=False)
    print(f"💾 Summary statistics: {summary_file}")
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"📄 Comprehensive PDF report: {pdf_file}")
    print(f"📊 Data summary: {summary_file}")
    
    # Print key findings
    batch1 = plotter.df_summary[plotter.df_summary['batch_size'] == 1].sort_values('inference_latency_ms_mean')
    print(f"\n⚡ TOP PERFORMERS (Batch Size = 1):")
    for _, row in batch1.head(3).iterrows():
        print(f"   {row['strategy_name']}: {row['inference_latency_ms_mean']:.1f}ms "
              f"(P95: {row['inference_latency_ms_p95']:.1f}ms, P99: {row['inference_latency_ms_p99']:.1f}ms)")

if __name__ == "__main__":
    main()