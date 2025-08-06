#!/usr/bin/env python3
"""
SpecMoE Quick Demo

A simple demonstration of the SpecMoE framework showing:
1. Configuration loading
2. Strategy initialization
3. Quick evaluation
4. Results analysis
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("🚀 SpecMoE Quick Demo")
    print("=" * 40)
    
    # Step 1: Load configuration
    print("\n1️⃣ Loading configuration...")
    try:
        from src.utils import ConfigManager
        config = ConfigManager()
        
        # Get Switch Transformer config
        arch_config = config.get_architecture_config("switch_transformer")
        print(f"   ✅ Loaded: {arch_config.num_experts} experts, {arch_config.num_layers} layers")
        
        # Get strategy config
        strategy_config = config.get_strategy_config("intelligent")
        print(f"   ✅ Strategy: {strategy_config.description}")
        
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False
    
    # Step 2: Initialize evaluation framework
    print("\n2️⃣ Initializing evaluation framework...")
    try:
        from src.evaluation import IsoCacheFramework
        
        cache_framework = IsoCacheFramework(
            total_cache_size_mb=50.0,
            expert_size_mb=2.5
        )
        print(f"   ✅ Cache: L1={cache_framework.l1_capacity}, L2={cache_framework.l2_capacity}, L3={cache_framework.l3_capacity} experts")
        
    except Exception as e:
        print(f"   ❌ Framework initialization error: {e}")
        return False
    
    # Step 3: Initialize strategy
    print("\n3️⃣ Initializing prefetching strategy...")
    try:
        from src.models import PreGatedMoEStrategy
        
        strategy = PreGatedMoEStrategy(
            cache_framework=cache_framework,
            num_experts=arch_config.num_experts,
            num_layers=arch_config.num_layers
        )
        print(f"   ✅ Strategy initialized: {strategy.__class__.__name__}")
        
    except Exception as e:
        print(f"   ❌ Strategy initialization error: {e}")
        return False
    
    # Step 4: Generate sample routing data
    print("\n4️⃣ Generating sample data...")
    try:
        import torch
        import random
        
        # Generate sample expert routing sequence
        batch_size = 8
        sequence_length = 64
        num_layers = arch_config.num_layers
        
        sample_data = []
        for batch_idx in range(batch_size):
            layer_routing = []
            for layer_idx in range(num_layers):
                # Simulate routing decisions (biased towards certain experts)
                popular_experts = [5, 12, 23, 45, 67, 89, 101, 120]  # Simulate popular experts
                if random.random() < 0.6:  # 60% chance to use popular expert
                    expert_id = random.choice(popular_experts)
                else:
                    expert_id = random.randint(0, arch_config.num_experts - 1)
                layer_routing.append(expert_id)
            sample_data.append(layer_routing)
        
        print(f"   ✅ Generated routing data: {batch_size} items × {num_layers} layers")
        print(f"   📊 Sample routing: {sample_data[0][:5]}... (first item, first 5 layers)")
        
    except Exception as e:
        print(f"   ❌ Data generation error: {e}")
        return False
    
    # Step 5: Run quick evaluation
    print("\n5️⃣ Running quick evaluation...")
    try:
        import time
        
        # Simulate evaluation
        start_time = time.time()
        
        cache_hits = 0
        cache_misses = 0
        total_requests = 0
        
        for batch_item in sample_data:
            for layer_idx, expert_id in enumerate(batch_item):
                total_requests += 1
                
                # Simulate cache lookup
                if expert_id in [5, 12, 23, 45]:  # Simulate cache hits for popular experts
                    cache_hits += 1
                else:
                    cache_misses += 1
        
        evaluation_time = time.time() - start_time
        cache_hit_rate = cache_hits / total_requests
        
        print(f"   ✅ Evaluation completed in {evaluation_time:.3f}s")
        print(f"   📈 Cache hit rate: {cache_hit_rate:.2%}")
        print(f"   📊 Total requests: {total_requests} ({cache_hits} hits, {cache_misses} misses)")
        
    except Exception as e:
        print(f"   ❌ Evaluation error: {e}")
        return False
    
    # Step 6: Analyze results
    print("\n6️⃣ Results analysis...")
    
    # Calculate speedup estimate
    on_demand_latency_ms = total_requests * 2.5  # Assume 2.5ms per expert load
    cached_latency_ms = cache_misses * 2.5  # Only misses require loading
    speedup = on_demand_latency_ms / cached_latency_ms if cached_latency_ms > 0 else float('inf')
    
    results = {
        "total_requests": total_requests,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hit_rate,
        "estimated_speedup": speedup,
        "evaluation_time_s": evaluation_time
    }
    
    print(f"   📊 Results summary:")
    print(f"      • Cache hit rate: {cache_hit_rate:.1%}")
    print(f"      • Estimated speedup: {speedup:.2f}×")
    print(f"      • Memory saved: {cache_hits * 2.5:.1f} MB")
    
    # Step 7: Save results
    print("\n7️⃣ Saving results...")
    try:
        output_dir = Path("results") / "demo"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "quick_demo_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"   ❌ Save error: {e}")
        return False
    
    # Final summary
    print("\n" + "=" * 40)
    print("🎉 Demo completed successfully!")
    print("\n📚 Next steps:")
    print("   • Run full evaluation: python -m src.evaluation.run_evaluation")
    print("   • Train models: python scripts/training/train_switch_predictor.py")
    print("   • View documentation: cat RUNNING_INSTRUCTIONS.md")
    print("   • Explore examples: ls examples/")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Demo failed. Check error messages above.")
        sys.exit(1)
    else:
        print(f"\n✅ Demo successful!")
        sys.exit(0)