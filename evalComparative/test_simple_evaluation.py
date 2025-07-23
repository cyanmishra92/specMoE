#!/usr/bin/env python3

"""
Simple test evaluation to validate the framework components work correctly.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add paths for imports
sys.path.append('../evalSwitchB8')
sys.path.append('../evalQwenB8')

def test_iso_cache_framework():
    """Test the iso-cache framework"""
    print("Testing iso-cache framework...")
    
    from evaluation.iso_cache_framework import IsoCacheFramework
    
    # Create cache framework
    cache = IsoCacheFramework(total_cache_size_mb=100.0, expert_size_mb=2.5)
    
    # Test basic operations
    latency1, level1 = cache.access_expert(0)
    latency2, level2 = cache.access_expert(1)
    latency3, level3 = cache.access_expert(0)  # Should be faster (cached)
    
    print(f"  Expert 0 first access: {latency1:.2f}ms ({level1})")
    print(f"  Expert 1 first access: {latency2:.2f}ms ({level2})")
    print(f"  Expert 0 second access: {latency3:.2f}ms ({level3})")
    
    # Test prefetching
    success = cache.prefetch_expert(5, 'L2')
    print(f"  Prefetch success: {success}")
    
    # Get performance metrics
    metrics = cache.get_performance_metrics()
    print(f"  Total accesses: {metrics['total_accesses']}")
    print(f"  Overall hit rate: {metrics['overall_hit_rate']:.3f}")
    
    print("✓ Iso-cache framework test passed\n")
    return True

def test_hardware_cost_model():
    """Test the hardware cost model"""
    print("Testing hardware cost model...")
    
    from evaluation.hardware_cost_model import HardwareAwareCostModel, DeviceType
    
    # Create cost model
    cost_model = HardwareAwareCostModel(DeviceType.RTX_4090)
    
    # Test transfer cost calculation
    transfer_cost = cost_model.calculate_cpu_gpu_transfer_cost(num_experts=16, concurrent_operations=1)
    print(f"  Transfer cost for 16 experts: {transfer_cost:.2f}ms")
    
    # Test batch processing efficiency
    batch_efficiency = cost_model.model_batch_processing_efficiency(batch_size=8, experts_per_batch=4)
    print(f"  Batch efficiency (batch=8): {batch_efficiency['overall_efficiency']:.3f}")
    
    # Get hardware characteristics
    characteristics = cost_model.get_performance_characteristics()
    print(f"  Device: {characteristics['device_type']}")
    print(f"  GPU memory: {characteristics['gpu_memory_gb']}GB")
    
    print("✓ Hardware cost model test passed\n")
    return True

def test_strategy_implementations():
    """Test the new strategy implementations"""
    print("Testing strategy implementations...")
    
    from evaluation.iso_cache_framework import IsoCacheFramework
    from strategies.pg_moe_strategy import PreGatedMoEStrategy
    from strategies.expertflow_plec_strategy import ExpertFlowPLECStrategy
    
    # Create cache framework
    cache = IsoCacheFramework(total_cache_size_mb=100.0)
    
    # Test Pre-gated MoE strategy
    print("  Testing Pre-gated MoE strategy...")
    pg_strategy = PreGatedMoEStrategy(cache, num_experts=64, num_layers=12, top_k=2)
    
    # Process a few layers
    latency1, details1 = pg_strategy.process_layer(0, [1, 5, 10])
    latency2, details2 = pg_strategy.process_layer(1, [2, 6, 11])
    
    print(f"    Layer 0 latency: {latency1:.2f}ms")
    print(f"    Layer 1 latency: {latency2:.2f}ms")
    
    # Get strategy metrics
    metrics = pg_strategy.get_strategy_metrics()
    print(f"    Predictions made: {metrics['predictions_made']}")
    print(f"    Strategy name: {metrics['strategy_name']}")
    
    # Reset and test ExpertFlow PLEC strategy
    cache.reset_metrics()
    cache.clear_cache()
    
    print("  Testing ExpertFlow PLEC strategy...")
    ef_strategy = ExpertFlowPLECStrategy(cache, num_experts=64, num_layers=12, top_k=2)
    
    # Process a few layers
    latency1, details1 = ef_strategy.process_layer(0, [3, 7, 12])
    latency2, details2 = ef_strategy.process_layer(1, [4, 8, 13])
    
    print(f"    Layer 0 latency: {latency1:.2f}ms")
    print(f"    Layer 1 latency: {latency2:.2f}ms")
    
    # Get strategy metrics
    metrics = ef_strategy.get_strategy_metrics()
    print(f"    Total predictions: {metrics['total_predictions']}")
    print(f"    Strategy name: {metrics['strategy_name']}")
    
    print("✓ Strategy implementations test passed\n")
    return True

def test_simple_evaluation():
    """Run a simple end-to-end evaluation test"""
    print("Running simple evaluation test...")
    
    from evaluation.iso_cache_framework import IsoCacheFramework
    from strategies.pg_moe_strategy import PreGatedMoEStrategy
    from strategies.expertflow_plec_strategy import ExpertFlowPLECStrategy
    
    # Configuration
    model_config = {
        'num_experts': 32,
        'num_layers': 8,
        'top_k': 2,
        'expert_size_mb': 2.5
    }
    
    cache_size_mb = 50
    batch_size = 4
    sequence_length = 10
    
    results = []
    
    # Test strategies
    strategies = {
        'pregated_moe': PreGatedMoEStrategy,
        'expertflow_plec': ExpertFlowPLECStrategy
    }
    
    for strategy_name, strategy_class in strategies.items():
        print(f"  Testing {strategy_name}...")
        
        # Create fresh cache and strategy
        cache = IsoCacheFramework(total_cache_size_mb=cache_size_mb, expert_size_mb=model_config['expert_size_mb'])
        strategy = strategy_class(cache, model_config['num_experts'], model_config['num_layers'], model_config['top_k'])
        
        # Generate simple routing trace
        np.random.seed(42)  # For reproducibility
        total_latency = 0.0
        
        for step in range(sequence_length):
            for layer in range(model_config['num_layers']):
                # Generate random expert selection
                required_experts = np.random.choice(
                    model_config['num_experts'], 
                    size=model_config['top_k'], 
                    replace=False
                ).tolist()
                
                # Process layer
                latency, details = strategy.process_layer(layer, required_experts)
                total_latency += latency
        
        # Collect results
        cache_metrics = cache.get_performance_metrics()
        strategy_metrics = strategy.get_strategy_metrics()
        
        result = {
            'strategy': strategy_name,
            'total_latency': total_latency,
            'hit_rate': cache_metrics['overall_hit_rate'],
            'l1_hit_rate': cache_metrics['l1_hit_rate'],
            'l2_hit_rate': cache_metrics['l2_hit_rate'],
            'l3_hit_rate': cache_metrics['l3_hit_rate'],
            'strategy_specific': strategy_metrics.get('predictions_made', 0)
        }
        
        results.append(result)
        
        print(f"    Total latency: {total_latency:.2f}ms")
        print(f"    Hit rate: {cache_metrics['overall_hit_rate']:.3f}")
        print(f"    L1/L2/L3 hits: {cache_metrics['l1_hit_rate']:.3f}/{cache_metrics['l2_hit_rate']:.3f}/{cache_metrics['l3_hit_rate']:.3f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n  Summary Results:")
    print(results_df.to_string(index=False))
    
    # Save test results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / 'simple_test_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n  Results saved to: {results_file}")
    
    print("✓ Simple evaluation test passed\n")
    return results_df

def main():
    """Run all tests"""
    print("Running framework validation tests...\n")
    
    try:
        # Test individual components
        test_iso_cache_framework()
        test_hardware_cost_model() 
        test_strategy_implementations()
        
        # Run simple end-to-end evaluation
        results_df = test_simple_evaluation()
        
        print("="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print(f"Framework is ready for comprehensive evaluation.")
        print(f"Test results saved to: results/simple_test_results.csv")
        
        return True
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()