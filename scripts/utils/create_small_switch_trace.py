#!/usr/bin/env python3
"""
Create Small Switch Transformer Routing Trace for Proof of Concept

Generates realistic but small routing trace based on Switch Transformer patterns:
- Power-law expert popularity distribution  
- Temporal locality patterns
- Realistic expert usage statistics
"""

import numpy as np
import json
import pickle
from pathlib import Path
from collections import Counter
import argparse

def create_small_switch_trace(num_traces: int = 1000, 
                            num_experts: int = 128,
                            output_file: Path = Path("routing_data/small_switch_trace.json")):
    """Create small but realistic Switch routing trace"""
    
    print(f"🔧 Creating small Switch trace:")
    print(f"   Traces: {num_traces}")
    print(f"   Experts: {num_experts}")
    
    # Create realistic expert popularity using power law (like real Switch)
    # Some experts are MUCH more popular than others
    alpha = 0.8  # Power law exponent (lower = more skewed)
    expert_weights = np.random.power(alpha, num_experts)
    expert_weights = expert_weights / np.sum(expert_weights)  # Normalize
    
    # Sort to make most popular experts have lower IDs (like real systems)
    sorted_indices = np.argsort(expert_weights)[::-1]  # Descending order
    expert_probs = np.zeros(num_experts)
    expert_probs[sorted_indices] = np.sort(expert_weights)[::-1]
    
    # Generate base trace with power law distribution
    expert_ids = np.arange(num_experts)
    base_traces = np.random.choice(expert_ids, size=num_traces, p=expert_probs)
    
    # Add temporal locality (burst patterns - same expert accessed multiple times)
    final_traces = []
    i = 0
    while i < len(base_traces):
        expert_id = base_traces[i]
        
        # With some probability, create a burst of the same expert
        if np.random.random() < 0.3:  # 30% chance of burst
            burst_length = np.random.randint(2, 6)  # 2-5 consecutive accesses
            for _ in range(min(burst_length, len(base_traces) - i)):
                final_traces.append(expert_id)
                i += 1
        else:
            final_traces.append(expert_id)
            i += 1
        
        if len(final_traces) >= num_traces:
            break
    
    final_traces = final_traces[:num_traces]
    
    # Create statistics
    trace_counter = Counter(final_traces)
    expert_distribution = {str(k): int(v) for k, v in trace_counter.items()}
    
    # Calculate key statistics
    total_accesses = len(final_traces)
    unique_experts = len(trace_counter)
    most_popular = trace_counter.most_common(1)[0]
    
    # Calculate entropy (measure of randomness)
    probabilities = [count / total_accesses for count in trace_counter.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    # Create trace data structure (ensure JSON serializable)
    trace_data = {
        "metadata": {
            "model_name": "switch-base-128-synthetic",
            "total_traces": int(total_accesses),
            "num_experts": int(num_experts),
            "unique_experts_accessed": int(unique_experts),
            "most_popular_expert": {
                "id": int(most_popular[0]),
                "count": int(most_popular[1]),
                "percentage": float(most_popular[1] / total_accesses * 100)
            },
            "access_entropy": float(entropy),
            "generation_params": {
                "power_law_alpha": float(alpha),
                "temporal_locality_prob": 0.3,
                "burst_length_range": [2, 5]
            }
        },
        "expert_distribution": expert_distribution,
        "trace_sequence": [int(x) for x in final_traces]  # Convert to regular ints
    }
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(trace_data, f, indent=2)
    
    print(f"✅ Small trace created: {output_file}")
    print(f"📊 Statistics:")
    print(f"   Total traces: {total_accesses}")
    print(f"   Unique experts: {unique_experts} / {num_experts}")
    print(f"   Most popular expert: {most_popular[0]} ({most_popular[1]} accesses, {most_popular[1]/total_accesses:.1%})")
    print(f"   Access entropy: {entropy:.2f} bits")
    print(f"   Top 10 experts: {trace_counter.most_common(10)}")
    
    return trace_data

def analyze_trace_patterns(trace_data):
    """Analyze patterns in the generated trace"""
    sequence = trace_data["trace_sequence"]
    
    # Temporal locality analysis
    locality_windows = [5, 10, 20]
    locality_stats = {}
    
    for window_size in locality_windows:
        repeats = 0
        total_windows = len(sequence) - window_size + 1
        
        for i in range(total_windows):
            window = sequence[i:i + window_size]
            unique_in_window = len(set(window))
            if unique_in_window < len(window):  # Repeated experts
                repeats += 1
        
        locality_ratio = repeats / total_windows if total_windows > 0 else 0
        locality_stats[f"window_{window_size}"] = locality_ratio
    
    print(f"\n📈 Temporal Locality Analysis:")
    for window, ratio in locality_stats.items():
        print(f"   {window}: {ratio:.2%} of windows have repeated experts")
    
    # Expert popularity distribution analysis
    counts = Counter(sequence)
    sorted_counts = sorted(counts.values(), reverse=True)
    
    # Calculate what percentage of accesses top experts account for
    top_percentages = [0.1, 0.2, 0.5]  # Top 10%, 20%, 50% of experts
    total_experts = len(counts)
    total_accesses = len(sequence)
    
    print(f"\n🏆 Expert Popularity Concentration:")
    for pct in top_percentages:
        top_n = max(1, int(total_experts * pct))
        top_accesses = sum(sorted_counts[:top_n])
        concentration = top_accesses / total_accesses
        print(f"   Top {pct:.0%} experts account for {concentration:.1%} of accesses")
    
    return locality_stats

def main():
    parser = argparse.ArgumentParser(description="Create Small Switch Trace")
    parser.add_argument("--traces", type=int, default=1000, 
                       help="Number of traces to generate")
    parser.add_argument("--experts", type=int, default=128,
                       help="Number of experts in model") 
    parser.add_argument("--output", type=str,
                       default="routing_data/small_switch_trace.json",
                       help="Output file path")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze generated trace patterns")
    
    args = parser.parse_args()
    
    # Create trace
    output_path = Path(args.output)
    trace_data = create_small_switch_trace(
        num_traces=args.traces,
        num_experts=args.experts, 
        output_file=output_path
    )
    
    # Analyze patterns if requested
    if args.analyze:
        analyze_trace_patterns(trace_data)
    
    print(f"\n🚀 Ready for cache hit analysis!")
    print(f"   Run: python scripts/analysis/switch_cache_hit_analysis.py --traces {output_path}")

if __name__ == "__main__":
    main()