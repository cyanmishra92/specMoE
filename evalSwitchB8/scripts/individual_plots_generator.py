#!/usr/bin/env python3
"""
Generate Individual PDF Plots for Research Paper

Creates separate PDF files for each plot type with:
- Bar charts for batch size comparisons
- Fixed effect size analysis
- Publication-ready formatting
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
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42
})

class IndividualPlotGenerator:
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
        if not df_detailed.empty:
            df_summary = df_detailed.groupby(['strategy', 'strategy_name', 'batch_size']).agg({
                'inference_latency_ms': ['mean', 'std', 'median', 'min', 'max', 
                                       lambda x: np.percentile(x, 95), lambda x: np.percentile(x, 99)],
                'memory_usage_mb': ['mean', 'std'],
                'cache_hit_rate': ['mean', 'std'],
                'prefetch_accuracy': ['mean', 'std'],
                'total_experts_loaded': ['mean', 'std'],
                'cache_misses': ['mean', 'std']
            }).round(3)
            
            # Flatten column names and fix percentiles
            new_columns = []
            for col in df_summary.columns:
                if 'lambda' in str(col[1]):
                    if 'lambda_0' in str(col):
                        new_columns.append(f"{col[0]}_p95")
                    else:
                        new_columns.append(f"{col[0]}_p99")
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            
            df_summary.columns = new_columns
            df_summary = df_summary.reset_index()
        else:
            df_summary = pd.DataFrame()
        
        return df_detailed, df_summary
    
    def plot_latency_vs_batch_size_bars(self, output_file: Path):
        """Plot latency vs batch size as grouped bar chart"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.batch_sizes))
        width = 0.15
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if not strategy_data.empty:
                means = strategy_data['inference_latency_ms_mean'].values
                stds = strategy_data['inference_latency_ms_std'].values
                
                bars = ax.bar(x + i * width, means, width, 
                             yerr=stds, capsize=4,
                             label=self.strategy_names[strategy],
                             color=self.strategy_colors[strategy],
                             alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Inference Latency (ms)', fontweight='bold')
        ax.set_title('Inference Latency vs Batch Size', fontweight='bold', pad=20)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(self.batch_sizes)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        with matplotlib.backends.backend_pdf.PdfPages(output_file) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Latency vs Batch Size (bars): {output_file}")
    
    def plot_cache_hit_rate_bars(self, output_file: Path):
        """Plot cache hit rate vs batch size as grouped bar chart"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.batch_sizes))
        width = 0.15
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if not strategy_data.empty:
                means = strategy_data['cache_hit_rate_mean'].values * 100
                stds = strategy_data['cache_hit_rate_std'].values * 100
                
                bars = ax.bar(x + i * width, means, width,
                             yerr=stds, capsize=4,
                             label=self.strategy_names[strategy],
                             color=self.strategy_colors[strategy],
                             alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Cache Hit Rate (%)', fontweight='bold')
        ax.set_title('Cache Hit Rate vs Batch Size', fontweight='bold', pad=20)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(self.batch_sizes)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        with matplotlib.backends.backend_pdf.PdfPages(output_file) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Cache Hit Rate (bars): {output_file}")
    
    def plot_memory_usage_bars(self, output_file: Path):
        """Plot memory usage vs batch size as grouped bar chart"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.batch_sizes))
        width = 0.15
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if not strategy_data.empty:
                means = strategy_data['memory_usage_mb_mean'].values
                stds = strategy_data['memory_usage_mb_std'].values
                
                bars = ax.bar(x + i * width, means, width,
                             yerr=stds, capsize=4,
                             label=self.strategy_names[strategy],
                             color=self.strategy_colors[strategy],
                             alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Memory Usage (MB)', fontweight='bold')
        ax.set_title('Memory Usage vs Batch Size', fontweight='bold', pad=20)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(self.batch_sizes)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        with matplotlib.backends.backend_pdf.PdfPages(output_file) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Memory Usage (bars): {output_file}")
    
    def plot_tail_latency_analysis(self, output_file: Path):
        """Plot comprehensive tail latency analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: P50, P95, P99 comparison at batch size 1
        ax1 = axes[0, 0]
        batch1 = self.df_summary[self.df_summary['batch_size'] == 1].sort_values('inference_latency_ms_mean')
        
        x = np.arange(len(batch1))
        width = 0.2
        
        metrics = ['inference_latency_ms_median', 'inference_latency_ms_p95', 'inference_latency_ms_p99']
        labels = ['P50 (Median)', 'P95', 'P99']
        colors = ['#2ECC71', '#F39C12', '#E74C3C']
        
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            values = batch1[metric].values
            bars = ax1.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        
        ax1.set_xlabel('Strategy', fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontweight='bold')
        ax1.set_title('Tail Latency Percentiles (Batch Size = 1)', fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(batch1['strategy_name'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
        
        # Plot 2: P99 latency across batch sizes (bars)
        ax2 = axes[0, 1]
        
        x = np.arange(len(self.batch_sizes))
        width = 0.15
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if not strategy_data.empty:
                p99_values = strategy_data['inference_latency_ms_p99'].values
                
                bars = ax2.bar(x + i * width, p99_values, width,
                              label=self.strategy_names[strategy],
                              color=self.strategy_colors[strategy],
                              alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Batch Size', fontweight='bold')
        ax2.set_ylabel('P99 Latency (ms)', fontweight='bold')
        ax2.set_title('P99 Tail Latency vs Batch Size', fontweight='bold')
        ax2.set_xticks(x + width * 2)
        ax2.set_xticklabels(self.batch_sizes)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_yscale('log')
        
        # Plot 3: Latency distribution violin plots (batch size 1)
        ax3 = axes[1, 0]
        
        batch1_detailed = self.df_detailed[self.df_detailed['batch_size'] == 1]
        strategy_order = batch1_detailed.groupby('strategy_name')['inference_latency_ms'].mean().sort_values().index
        
        violin_data = [batch1_detailed[batch1_detailed['strategy_name'] == s]['inference_latency_ms'].values 
                      for s in strategy_order]
        
        violin_parts = ax3.violinplot(violin_data, positions=range(len(strategy_order)),
                                     showmeans=True, showmedians=True, showextrema=True)
        
        # Color the violins
        for i, (pc, strategy) in enumerate(zip(violin_parts['bodies'], strategy_order)):
            strategy_key = [k for k, v in self.strategy_names.items() if v == strategy][0]
            pc.set_facecolor(self.strategy_colors[strategy_key])
            pc.set_alpha(0.7)
        
        ax3.set_xlabel('Strategy', fontweight='bold')
        ax3.set_ylabel('Latency (ms)', fontweight='bold')
        ax3.set_title('Latency Distribution (Batch Size = 1)', fontweight='bold')
        ax3.set_xticks(range(len(strategy_order)))
        ax3.set_xticklabels(strategy_order, rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Tail ratio analysis (P99/P50) as bars
        ax4 = axes[1, 1]
        
        x = np.arange(len(self.batch_sizes))
        width = 0.15
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if not strategy_data.empty:
                tail_ratios = strategy_data['inference_latency_ms_p99'] / strategy_data['inference_latency_ms_median']
                
                bars = ax4.bar(x + i * width, tail_ratios, width,
                              label=self.strategy_names[strategy],
                              color=self.strategy_colors[strategy],
                              alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax4.set_xlabel('Batch Size', fontweight='bold')
        ax4.set_ylabel('Tail Ratio (P99/P50)', fontweight='bold')
        ax4.set_title('Latency Tail Behavior vs Batch Size', fontweight='bold')
        ax4.set_xticks(x + width * 2)
        ax4.set_xticklabels(self.batch_sizes)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        with matplotlib.backends.backend_pdf.PdfPages(output_file) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Tail Latency Analysis: {output_file}")
    
    def plot_effect_size_analysis(self, output_file: Path):
        """Plot comprehensive effect size analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Effect sizes (Cohen's d) vs baseline for batch size 1
        ax1 = axes[0, 0]
        
        baseline_data_1 = self.df_detailed[
            (self.df_detailed['strategy'] == 'A') & 
            (self.df_detailed['batch_size'] == 1)
        ]['inference_latency_ms']
        
        effect_sizes_1 = []
        
        for strategy in ['B', 'C', 'D', 'E']:
            strategy_data = self.df_detailed[
                (self.df_detailed['strategy'] == strategy) & 
                (self.df_detailed['batch_size'] == 1)
            ]['inference_latency_ms']
            
            if not strategy_data.empty and len(baseline_data_1) > 0:
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(baseline_data_1) - 1) * baseline_data_1.var() + 
                                    (len(strategy_data) - 1) * strategy_data.var()) / 
                                   (len(baseline_data_1) + len(strategy_data) - 2))
                
                if pooled_std > 0:
                    cohens_d = (baseline_data_1.mean() - strategy_data.mean()) / pooled_std
                else:
                    cohens_d = 0
                
                effect_sizes_1.append({
                    'strategy': strategy,
                    'strategy_name': self.strategy_names[strategy],
                    'effect_size': cohens_d
                })
        
        if effect_sizes_1:
            effect_df_1 = pd.DataFrame(effect_sizes_1)
            colors = [self.strategy_colors[s] for s in effect_df_1['strategy']]
            
            bars = ax1.bar(range(len(effect_df_1)), effect_df_1['effect_size'], 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, val in zip(bars, effect_df_1['effect_size']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_xlabel('Strategy', fontweight='bold')
            ax1.set_ylabel("Cohen's d (Effect Size)", fontweight='bold')
            ax1.set_title('Effect Size vs Baseline (Batch Size = 1)', fontweight='bold')
            ax1.set_xticks(range(len(effect_df_1)))
            ax1.set_xticklabels(effect_df_1['strategy_name'])
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add effect size interpretation lines
            ax1.axhline(y=0.2, color='gray', linestyle='--', alpha=0.7, label='Small Effect (0.2)')
            ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Effect (0.5)')
            ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large Effect (0.8)')
            ax1.legend(loc='upper right')
        
        # Plot 2: Effect sizes across all batch sizes
        ax2 = axes[0, 1]
        
        x = np.arange(len(self.batch_sizes))
        width = 0.2
        
        for i, strategy in enumerate(['B', 'C', 'D', 'E']):
            effect_sizes_all = []
            
            for batch_size in self.batch_sizes:
                baseline_data = self.df_detailed[
                    (self.df_detailed['strategy'] == 'A') & 
                    (self.df_detailed['batch_size'] == batch_size)
                ]['inference_latency_ms']
                
                strategy_data = self.df_detailed[
                    (self.df_detailed['strategy'] == strategy) & 
                    (self.df_detailed['batch_size'] == batch_size)
                ]['inference_latency_ms']
                
                if not strategy_data.empty and len(baseline_data) > 0:
                    pooled_std = np.sqrt(((len(baseline_data) - 1) * baseline_data.var() + 
                                        (len(strategy_data) - 1) * strategy_data.var()) / 
                                       (len(baseline_data) + len(strategy_data) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (baseline_data.mean() - strategy_data.mean()) / pooled_std
                    else:
                        cohens_d = 0
                    
                    effect_sizes_all.append(cohens_d)
                else:
                    effect_sizes_all.append(0)
            
            bars = ax2.bar(x + i * width, effect_sizes_all, width,
                          label=self.strategy_names[strategy],
                          color=self.strategy_colors[strategy],
                          alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Batch Size', fontweight='bold')
        ax2.set_ylabel("Cohen's d (Effect Size)", fontweight='bold')
        ax2.set_title('Effect Size vs Baseline Across Batch Sizes', fontweight='bold')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(self.batch_sizes)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Speedup vs baseline
        ax3 = axes[1, 0]
        
        x = np.arange(len(self.batch_sizes))
        width = 0.2
        
        for i, strategy in enumerate(['B', 'C', 'D', 'E']):
            speedups = []
            
            for batch_size in self.batch_sizes:
                baseline_latency = self.df_summary[
                    (self.df_summary['strategy'] == 'A') & 
                    (self.df_summary['batch_size'] == batch_size)
                ]['inference_latency_ms_mean']
                
                strategy_latency = self.df_summary[
                    (self.df_summary['strategy'] == strategy) & 
                    (self.df_summary['batch_size'] == batch_size)
                ]['inference_latency_ms_mean']
                
                if not baseline_latency.empty and not strategy_latency.empty:
                    speedup = baseline_latency.iloc[0] / strategy_latency.iloc[0]
                    speedups.append(speedup)
                else:
                    speedups.append(1.0)
            
            bars = ax3.bar(x + i * width, speedups, width,
                          label=self.strategy_names[strategy],
                          color=self.strategy_colors[strategy],
                          alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
        ax3.set_xlabel('Batch Size', fontweight='bold')
        ax3.set_ylabel('Speedup vs Baseline', fontweight='bold')
        ax3.set_title('Performance Speedup vs On-Demand', fontweight='bold')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(self.batch_sizes)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Statistical significance (p-values)
        ax4 = axes[1, 1]
        
        p_values = []
        strategy_labels = []
        
        baseline_data_1 = self.df_detailed[
            (self.df_detailed['strategy'] == 'A') & 
            (self.df_detailed['batch_size'] == 1)
        ]['inference_latency_ms']
        
        for strategy in ['B', 'C', 'D', 'E']:
            strategy_data = self.df_detailed[
                (self.df_detailed['strategy'] == strategy) & 
                (self.df_detailed['batch_size'] == 1)
            ]['inference_latency_ms']
            
            if not strategy_data.empty and len(baseline_data_1) > 0:
                # Two-sample t-test
                t_stat, p_val = stats.ttest_ind(baseline_data_1, strategy_data)
                p_values.append(p_val)
                strategy_labels.append(self.strategy_names[strategy])
        
        if p_values:
            colors = [self.strategy_colors[s] for s in ['B', 'C', 'D', 'E'][:len(p_values)]]
            bars = ax4.bar(range(len(p_values)), [-np.log10(p) for p in p_values],
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add significance threshold line
            ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, 
                       label='Significance Threshold (p=0.05)')
            ax4.axhline(y=-np.log10(0.01), color='darkred', linestyle='--', alpha=0.7,
                       label='High Significance (p=0.01)')
            
            ax4.set_xlabel('Strategy', fontweight='bold')
            ax4.set_ylabel('-log10(p-value)', fontweight='bold')
            ax4.set_title('Statistical Significance vs Baseline', fontweight='bold')
            ax4.set_xticks(range(len(strategy_labels)))
            ax4.set_xticklabels(strategy_labels)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        with matplotlib.backends.backend_pdf.PdfPages(output_file) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Effect Size Analysis: {output_file}")
    
    def plot_scalability_analysis(self, output_file: Path):
        """Plot scalability and throughput analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Normalized latency (relative to batch size 1) - bars
        ax1 = axes[0, 0]
        
        x = np.arange(len(self.batch_sizes))
        width = 0.15
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if len(strategy_data) > 1:
                baseline = strategy_data[strategy_data['batch_size'] == 1]['inference_latency_ms_mean'].iloc[0]
                normalized = strategy_data['inference_latency_ms_mean'] / baseline
                
                bars = ax1.bar(x + i * width, normalized, width,
                              label=self.strategy_names[strategy],
                              color=self.strategy_colors[strategy],
                              alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add ideal scaling line
        ax1.plot(x, self.batch_sizes, 'k--', alpha=0.7, linewidth=2, label='Linear Scaling')
        
        ax1.set_xlabel('Batch Size', fontweight='bold')
        ax1.set_ylabel('Normalized Latency (vs Batch=1)', fontweight='bold')
        ax1.set_title('Latency Scalability', fontweight='bold')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(self.batch_sizes)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
        
        # Plot 2: Throughput analysis (requests/second) - bars
        ax2 = axes[0, 1]
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if not strategy_data.empty:
                throughput = strategy_data['batch_size'] / (strategy_data['inference_latency_ms_mean'] / 1000)
                
                bars = ax2.bar(x + i * width, throughput, width,
                              label=self.strategy_names[strategy],
                              color=self.strategy_colors[strategy],
                              alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Batch Size', fontweight='bold')
        ax2.set_ylabel('Throughput (requests/second)', fontweight='bold')
        ax2.set_title('Throughput vs Batch Size', fontweight='bold')
        ax2.set_xticks(x + width * 2)
        ax2.set_xticklabels(self.batch_sizes)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_yscale('log')
        
        # Plot 3: Memory efficiency (hit rate per MB) - bars
        ax3 = axes[1, 0]
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if not strategy_data.empty:
                efficiency = (strategy_data['cache_hit_rate_mean'] * 100) / (strategy_data['memory_usage_mb_mean'] + 1e-6)
                
                bars = ax3.bar(x + i * width, efficiency, width,
                              label=self.strategy_names[strategy],
                              color=self.strategy_colors[strategy],
                              alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax3.set_xlabel('Batch Size', fontweight='bold')
        ax3.set_ylabel('Cache Hit % per MB Memory', fontweight='bold')
        ax3.set_title('Memory Efficiency vs Batch Size', fontweight='bold')
        ax3.set_xticks(x + width * 2)
        ax3.set_xticklabels(self.batch_sizes)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Expert loading efficiency - bars
        ax4 = axes[1, 1]
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = self.df_summary[self.df_summary['strategy'] == strategy].sort_values('batch_size')
            if not strategy_data.empty:
                efficiency = (strategy_data['cache_hit_rate_mean'] * 100) / (strategy_data['total_experts_loaded_mean'] + 1e-6)
                
                bars = ax4.bar(x + i * width, efficiency, width,
                              label=self.strategy_names[strategy],
                              color=self.strategy_colors[strategy],
                              alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax4.set_xlabel('Batch Size', fontweight='bold')
        ax4.set_ylabel('Cache Hit % per Expert Loaded', fontweight='bold')
        ax4.set_title('Expert Loading Efficiency', fontweight='bold')
        ax4.set_xticks(x + width * 2)
        ax4.set_xticklabels(self.batch_sizes)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        with matplotlib.backends.backend_pdf.PdfPages(output_file) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Scalability Analysis: {output_file}")
    
    def generate_all_individual_plots(self, plots_dir: Path):
        """Generate all individual PDF plots"""
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Generating individual PDF plots...")
        
        if self.df_detailed.empty:
            print("❌ No data found!")
            return
        
        # Generate individual plots
        self.plot_latency_vs_batch_size_bars(plots_dir / "01_latency_vs_batch_size.pdf")
        self.plot_cache_hit_rate_bars(plots_dir / "02_cache_hit_rate_vs_batch_size.pdf")
        self.plot_memory_usage_bars(plots_dir / "03_memory_usage_vs_batch_size.pdf")
        self.plot_tail_latency_analysis(plots_dir / "04_tail_latency_analysis.pdf")
        self.plot_effect_size_analysis(plots_dir / "05_effect_size_analysis.pdf")
        self.plot_scalability_analysis(plots_dir / "06_scalability_analysis.pdf")
        
        print(f"✅ All individual plots generated in: {plots_dir}")

def main():
    results_dir = Path("../results")
    plots_dir = results_dir / "individual_plots"
    
    print("🚀 Individual Plot Generator")
    print("=" * 35)
    
    # Initialize generator
    generator = IndividualPlotGenerator(results_dir)
    
    # Generate all plots
    generator.generate_all_individual_plots(plots_dir)
    
    print(f"\n🎯 INDIVIDUAL PLOTS COMPLETE!")
    print(f"📁 Location: {plots_dir}")
    print(f"📄 Files generated:")
    for pdf_file in sorted(plots_dir.glob("*.pdf")):
        print(f"   - {pdf_file.name}")

if __name__ == "__main__":
    main()