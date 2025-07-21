#!/usr/bin/env python3
"""
Generate Research Paper Summary Table
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Load the comprehensive summary
    results_dir = Path("../results")
    summary_file = results_dir / "comprehensive_summary.csv"
    
    if not summary_file.exists():
        print(f"❌ Summary file not found: {summary_file}")
        return
    
    df = pd.read_csv(summary_file)
    
    # Create research paper table
    research_table = []
    
    for batch_size in [1, 2, 4, 8, 16]:
        batch_data = df[df['batch_size'] == batch_size].copy()
        
        # Sort by mean latency
        batch_data = batch_data.sort_values('inference_latency_ms_mean')
        
        for _, row in batch_data.iterrows():
            strategy_name = row['strategy_name']
            
            # Calculate metrics
            mean_lat = row['inference_latency_ms_mean']
            std_lat = row['inference_latency_ms_std']
            p95_lat = row['inference_latency_ms_p95']
            p99_lat = row['inference_latency_ms_p99']
            hit_rate = row['cache_hit_rate_mean'] * 100
            memory_mb = row['memory_usage_mb_mean']
            
            # Calculate speedup vs baseline (OnDemand)
            baseline_row = batch_data[batch_data['strategy'] == 'A']
            if not baseline_row.empty:
                baseline_latency = baseline_row['inference_latency_ms_mean'].iloc[0]
                speedup = baseline_latency / mean_lat
            else:
                speedup = 1.0
            
            research_table.append({
                'Strategy': strategy_name,
                'Batch Size': batch_size,
                'Mean Latency (ms)': f"{mean_lat:.1f} ± {std_lat:.1f}",
                'P95 Latency (ms)': f"{p95_lat:.1f}",
                'P99 Latency (ms)': f"{p99_lat:.1f}",
                'Cache Hit Rate (%)': f"{hit_rate:.1f}",
                'Memory (MB)': f"{memory_mb:.0f}",
                'Speedup': f"{speedup:.2f}×"
            })
    
    research_df = pd.DataFrame(research_table)
    
    # Save research table
    research_file = results_dir / "research_paper_table.csv"
    research_df.to_csv(research_file, index=False)
    
    # Print LaTeX table
    print("📄 Research Paper Summary Table (LaTeX format):")
    print("=" * 60)
    
    # Group by batch size for better presentation
    for batch_size in [1, 2, 4, 8, 16]:
        batch_subset = research_df[research_df['Batch Size'] == batch_size]
        
        print(f"\n\\textbf{{Batch Size {batch_size}:}}")
        print("\\begin{tabular}{lrrrrr}")
        print("\\hline")
        print("Strategy & Mean Latency & P95 & P99 & Hit Rate & Speedup \\\\")
        print("& (ms) & (ms) & (ms) & (\\%) & \\\\")
        print("\\hline")
        
        for _, row in batch_subset.iterrows():
            print(f"{row['Strategy']} & {row['Mean Latency (ms)']} & {row['P95 Latency (ms)']} & {row['P99 Latency (ms)']} & {row['Cache Hit Rate (%)']} & {row['Speedup']} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
    
    print(f"\n💾 Full table saved to: {research_file}")
    
    # Key insights for paper
    print(f"\n🎯 KEY INSIGHTS FOR RESEARCH PAPER:")
    print("=" * 50)
    
    # Best performer
    best_batch1 = research_df[research_df['Batch Size'] == 1].iloc[0]
    print(f"🏆 Best Performance: {best_batch1['Strategy']} achieves {best_batch1['Speedup']} speedup")
    
    # Tail latency insight
    batch1_data = research_df[research_df['Batch Size'] == 1]
    print(f"📊 Tail Latency: Oracle shows {batch1_data.iloc[0]['P99 Latency (ms)']} P99 latency")
    
    # Cache effectiveness
    oracle_hit = float(batch1_data[batch1_data['Strategy'] == 'Oracle']['Cache Hit Rate (%)'].iloc[0].rstrip('%'))
    baseline_hit = float(batch1_data[batch1_data['Strategy'] == 'OnDemand']['Cache Hit Rate (%)'].iloc[0].rstrip('%'))
    print(f"💾 Cache Impact: {oracle_hit:.1f}% vs {baseline_hit:.1f}% hit rate difference")

if __name__ == "__main__":
    main()