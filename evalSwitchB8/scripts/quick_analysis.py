#!/usr/bin/env python3
"""
Quick Analysis using JSON files to create CSV report
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_all_results(results_dir):
    """Load all JSON results"""
    strategies = ["A", "B", "C", "D", "E"]
    batch_sizes = [1, 2, 4, 8, 16]
    
    all_data = []
    
    for strategy in strategies:
        for batch_size in batch_sizes:
            json_file = results_dir / f"strategy_{strategy}_batch_{batch_size}.json"
            
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    # Extract metrics for each run
                    for run in data:
                        all_data.append({
                            'strategy': strategy,
                            'strategy_name': {
                                'A': 'OnDemand', 'B': 'Oracle', 'C': 'MultiLook', 
                                'D': 'TopK', 'E': 'Intelligent'
                            }[strategy],
                            'batch_size': run['batch_size'],
                            'run_id': run['run_id'],
                            'inference_latency_ms': run['inference_latency_ms'],
                            'memory_usage_mb': run['memory_usage_mb'],
                            'cache_hit_rate': run['cache_hit_rate'],
                            'prefetch_accuracy': run['prefetch_accuracy'],
                            'total_experts_loaded': run['total_experts_loaded'],
                            'cache_misses': run['cache_misses']
                        })
                    
                    print(f"✅ Loaded {strategy}_{batch_size}: {len(data)} runs")
                    
                except Exception as e:
                    print(f"❌ Failed to load {json_file}: {e}")
    
    return pd.DataFrame(all_data)

def create_summary_statistics(df):
    """Create summary statistics"""
    summary = df.groupby(['strategy', 'strategy_name', 'batch_size']).agg({
        'inference_latency_ms': ['mean', 'std', 'min', 'max'],
        'memory_usage_mb': ['mean', 'std'],
        'cache_hit_rate': ['mean', 'std'],
        'prefetch_accuracy': ['mean', 'std'],
        'total_experts_loaded': ['mean', 'std'],
        'cache_misses': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary

def main():
    results_dir = Path("../results")
    
    print("🚀 Quick Analysis of Switch Prefetching Results")
    print("=" * 50)
    
    # Load all results
    df = load_all_results(results_dir)
    print(f"\n📊 Total data points loaded: {len(df)}")
    
    if df.empty:
        print("❌ No data loaded!")
        return
    
    # Create summary statistics
    summary = create_summary_statistics(df)
    
    # Save detailed results
    detailed_csv = results_dir / "switch_detailed_results.csv"
    df.to_csv(detailed_csv, index=False)
    print(f"💾 Detailed results: {detailed_csv}")
    
    # Save summary statistics
    summary_csv = results_dir / "switch_summary_results.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"💾 Summary statistics: {summary_csv}")
    
    # Print key insights
    print(f"\n🎯 KEY FINDINGS:")
    print("-" * 30)
    
    # Best strategy overall
    strategy_performance = summary.groupby('strategy_name')['inference_latency_ms_mean'].mean()
    best_strategy = strategy_performance.idxmin()
    best_latency = strategy_performance.min()
    
    print(f"🏆 Best Overall Strategy: {best_strategy}")
    print(f"   Average Latency: {best_latency:.2f}ms")
    
    # Performance by batch size 1
    batch1 = summary[summary['batch_size'] == 1].sort_values('inference_latency_ms_mean')
    print(f"\n📈 Performance at Batch Size 1:")
    for _, row in batch1.iterrows():
        print(f"   {row['strategy_name']}: {row['inference_latency_ms_mean']:.2f} ± {row['inference_latency_ms_std']:.2f}ms")
    
    # Cache hit rates
    print(f"\n💾 Cache Hit Rates (Batch Size 1):")
    batch1_hits = summary[summary['batch_size'] == 1].sort_values('cache_hit_rate_mean', ascending=False)
    for _, row in batch1_hits.iterrows():
        print(f"   {row['strategy_name']}: {row['cache_hit_rate_mean']:.2%}")
    
    # Improvements over baseline
    baseline = summary[(summary['strategy'] == 'A') & (summary['batch_size'] == 1)]['inference_latency_ms_mean'].iloc[0]
    
    print(f"\n⚡ Speedup vs Baseline (OnDemand @ Batch=1):")
    batch1_sorted = batch1.sort_values('inference_latency_ms_mean')
    for _, row in batch1_sorted.iterrows():
        if row['strategy'] != 'A':
            speedup = (baseline - row['inference_latency_ms_mean']) / baseline * 100
            print(f"   {row['strategy_name']}: {speedup:.1f}% improvement")
    
    print(f"\n✅ Analysis completed! Check CSV files for detailed results.")

if __name__ == "__main__":
    main()