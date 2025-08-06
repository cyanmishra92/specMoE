#!/usr/bin/env python3
"""
Unified Evaluation Script for SpecMoE Expert Prefetching Strategies

This script provides a single entry point for all evaluation tasks including:
- Switch Transformer evaluation
- Qwen MoE evaluation  
- Comparative analysis with baselines
- Expert deduplication analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_switch_transformer_evaluation(
    strategies: List[str],
    batch_sizes: List[int],
    cache_size: int = 50,
    num_runs: int = 10,
    output_dir: str = "results/switch_transformer"
) -> Dict:
    """Run Switch Transformer evaluation."""
    print(f"🔄 Running Switch Transformer evaluation...")
    print(f"  Strategies: {strategies}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Cache size: {cache_size}MB")
    
    # Import and run Switch Transformer evaluation
    try:
        from experiments.switch_transformer.switch_evaluation import run_evaluation
        results = run_evaluation(
            strategies=strategies,
            batch_sizes=batch_sizes,
            cache_size_mb=cache_size,
            num_runs=num_runs,
            output_dir=output_dir
        )
        print("✅ Switch Transformer evaluation completed")
        return results
    except ImportError as e:
        print(f"❌ Error importing Switch Transformer evaluation: {e}")
        return {}

def run_qwen_moe_evaluation(
    strategies: List[str],
    batch_sizes: List[int], 
    cache_size: int = 160,
    num_runs: int = 10,
    output_dir: str = "results/qwen_moe"
) -> Dict:
    """Run Qwen MoE evaluation."""
    print(f"🔄 Running Qwen MoE evaluation...")
    print(f"  Strategies: {strategies}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Cache size: {cache_size}MB")
    
    try:
        from experiments.qwen_moe.qwen_comprehensive_evaluation import run_evaluation
        results = run_evaluation(
            strategies=strategies,
            batch_sizes=batch_sizes,
            cache_size_mb=cache_size,
            num_runs=num_runs,
            output_dir=output_dir
        )
        print("✅ Qwen MoE evaluation completed")
        return results
    except ImportError as e:
        print(f"❌ Error importing Qwen MoE evaluation: {e}")
        return {}

def run_comparative_evaluation(
    include_baselines: bool = True,
    enable_deduplication: bool = True,
    output_dir: str = "results/comparative"
) -> Dict:
    """Run comparative evaluation with paper baselines."""
    print(f"🔄 Running comparative evaluation...")
    print(f"  Include baselines: {include_baselines}")
    print(f"  Enable deduplication: {enable_deduplication}")
    
    try:
        from experiments.comparative.corrected_comparative_evaluation import main
        results = main(
            include_baselines=include_baselines,
            enable_deduplication=enable_deduplication,
            output_dir=output_dir
        )
        print("✅ Comparative evaluation completed") 
        return results
    except ImportError as e:
        print(f"❌ Error importing comparative evaluation: {e}")
        return {}

def run_deduplication_analysis(
    batch_sizes: List[int] = None,
    output_dir: str = "results/deduplication"
) -> Dict:
    """Run expert deduplication analysis."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
    print(f"🔄 Running expert deduplication analysis...")
    print(f"  Batch sizes: {batch_sizes}")
    
    try:
        from expert_deduplication_analysis import main
        results = main(
            batch_sizes=batch_sizes,
            output_dir=output_dir
        )
        print("✅ Expert deduplication analysis completed")
        return results
    except ImportError as e:
        print(f"❌ Error importing deduplication analysis: {e}")
        return {}

def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="SpecMoE Unified Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Switch Transformer evaluation with intelligent strategy
  python -m src.evaluation.run_evaluation --architecture switch_transformer --strategy intelligent

  # Run full comparative analysis
  python -m src.evaluation.run_evaluation --architecture comparative --include_baselines

  # Run expert deduplication analysis
  python -m src.evaluation.run_evaluation --architecture deduplication --batch_sizes 1,8,16,32,64

  # Run all evaluations
  python -m src.evaluation.run_evaluation --architecture all
        """
    )
    
    # Architecture selection
    parser.add_argument(
        "--architecture", 
        choices=["switch_transformer", "qwen_moe", "comparative", "deduplication", "all"],
        default="switch_transformer",
        help="Architecture or analysis to evaluate"
    )
    
    # Strategy selection
    parser.add_argument(
        "--strategy",
        nargs="*",
        default=["intelligent", "topk", "multilook", "oracle"],
        help="Prefetching strategies to evaluate"
    )
    
    # Batch sizes
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,8,16,32",
        help="Comma-separated batch sizes to test"
    )
    
    # Cache configuration
    parser.add_argument(
        "--cache_size",
        type=int,
        help="Cache size in MB (default: 50 for Switch, 160 for Qwen)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of evaluation runs for statistical significance"
    )
    
    # Comparative evaluation options
    parser.add_argument(
        "--include_baselines",
        action="store_true",
        help="Include paper baseline methods in comparison"
    )
    
    parser.add_argument(
        "--enable_deduplication",
        action="store_true", 
        help="Enable expert deduplication optimization"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base directory for output results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    # Set cache size defaults if not specified
    cache_size = args.cache_size
    if cache_size is None:
        cache_size = 50 if args.architecture == "switch_transformer" else 160
    
    print("🚀 SpecMoE Evaluation Framework")
    print("=" * 50)
    print(f"Architecture: {args.architecture}")
    print(f"Strategies: {args.strategy}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Cache size: {cache_size}MB")
    print(f"Output: {args.output_dir}")
    print("=" * 50)
    
    results = {}
    
    if args.architecture == "switch_transformer" or args.architecture == "all":
        results["switch_transformer"] = run_switch_transformer_evaluation(
            strategies=args.strategy,
            batch_sizes=batch_sizes,
            cache_size=cache_size,
            num_runs=args.num_runs,
            output_dir=f"{args.output_dir}/switch_transformer"
        )
    
    if args.architecture == "qwen_moe" or args.architecture == "all":
        qwen_cache_size = 160 if cache_size == 50 else cache_size  # Adjust for Qwen
        results["qwen_moe"] = run_qwen_moe_evaluation(
            strategies=args.strategy,
            batch_sizes=batch_sizes,
            cache_size=qwen_cache_size,
            num_runs=args.num_runs,
            output_dir=f"{args.output_dir}/qwen_moe"
        )
    
    if args.architecture == "comparative" or args.architecture == "all":
        results["comparative"] = run_comparative_evaluation(
            include_baselines=args.include_baselines,
            enable_deduplication=args.enable_deduplication,
            output_dir=f"{args.output_dir}/comparative"
        )
    
    if args.architecture == "deduplication" or args.architecture == "all":
        results["deduplication"] = run_deduplication_analysis(
            batch_sizes=batch_sizes,
            output_dir=f"{args.output_dir}/deduplication"
        )
    
    print("\n🎉 Evaluation Summary")
    print("=" * 50)
    for arch, result in results.items():
        if result:
            print(f"✅ {arch.title()}: {len(result)} results generated")
        else:
            print(f"❌ {arch.title()}: No results generated")
    
    print(f"\nResults saved to: {args.output_dir}/")
    print("See generated markdown reports for detailed analysis.")

if __name__ == "__main__":
    main()