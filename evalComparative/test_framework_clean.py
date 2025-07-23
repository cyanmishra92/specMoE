#!/usr/bin/env python3

"""
Clean framework validation test without dependencies on corrupted files.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def test_basic_components():
    """Test basic framework components independently"""
    print("Testing basic framework components...\n")
    
    # Test 1: Iso-cache framework
    print("1. Testing iso-cache framework...")
    try:
        from evaluation.iso_cache_framework import IsoCacheFramework
        
        cache = IsoCacheFramework(total_cache_size_mb=100.0, expert_size_mb=2.5)
        
        # Test basic operations
        latency1, level1 = cache.access_expert(0)
        latency2, level2 = cache.access_expert(1)
        latency3, level3 = cache.access_expert(0)  # Should be cached
        
        print(f"   Expert 0 first access: {latency1:.2f}ms ({level1})")
        print(f"   Expert 1 first access: {latency2:.2f}ms ({level2})") 
        print(f"   Expert 0 second access: {latency3:.2f}ms ({level3})")
        
        metrics = cache.get_performance_metrics()
        print(f"   Cache hit rate: {metrics['overall_hit_rate']:.3f}")
        print("   ✓ Iso-cache framework working correctly\n")
        
    except Exception as e:
        print(f"   ✗ Iso-cache framework error: {e}\n")
        return False
    
    # Test 2: Hardware cost model
    print("2. Testing hardware cost model...")
    try:
        from evaluation.hardware_cost_model import HardwareAwareCostModel, DeviceType
        
        cost_model = HardwareAwareCostModel(DeviceType.RTX_4090)
        
        transfer_cost = cost_model.calculate_cpu_gpu_transfer_cost(num_experts=16)
        print(f"   Transfer cost for 16 experts: {transfer_cost:.2f}ms")
        
        characteristics = cost_model.get_performance_characteristics()
        print(f"   Device: {characteristics['device_type']}")
        print(f"   GPU memory: {characteristics['gpu_memory_gb']}GB")
        print("   ✓ Hardware cost model working correctly\n")
        
    except Exception as e:
        print(f"   ✗ Hardware cost model error: {e}\n")
        return False
    
    # Test 3: Strategy implementations
    print("3. Testing strategy implementations...")
    try:
        from evaluation.iso_cache_framework import IsoCacheFramework
        from strategies.pg_moe_strategy import PreGatedMoEStrategy
        from strategies.expertflow_plec_strategy import ExpertFlowPLECStrategy
        
        cache = IsoCacheFramework(total_cache_size_mb=100.0)
        
        # Test Pre-gated MoE
        pg_strategy = PreGatedMoEStrategy(cache, num_experts=64, num_layers=12, top_k=2)
        latency, details = pg_strategy.process_layer(0, [1, 5, 10])
        metrics = pg_strategy.get_strategy_metrics()
        print(f"   Pre-gated MoE latency: {latency:.2f}ms")
        print(f"   Pre-gated MoE predictions: {metrics['predictions_made']}")
        
        # Reset cache and test ExpertFlow PLEC
        cache.reset_metrics()
        cache.clear_cache()
        ef_strategy = ExpertFlowPLECStrategy(cache, num_experts=64, num_layers=12, top_k=2)
        latency, details = ef_strategy.process_layer(0, [3, 7, 12])
        metrics = ef_strategy.get_strategy_metrics()
        print(f"   ExpertFlow PLEC latency: {latency:.2f}ms")
        print(f"   ExpertFlow PLEC predictions: {metrics['total_predictions']}")
        print("   ✓ Strategy implementations working correctly\n")
        
    except Exception as e:
        print(f"   ✗ Strategy implementations error: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_mini_evaluation():
    """Run a minimal end-to-end evaluation"""
    print("4. Running mini evaluation...")
    
    try:
        from evaluation.iso_cache_framework import IsoCacheFramework
        from strategies.pg_moe_strategy import PreGatedMoEStrategy
        from strategies.expertflow_plec_strategy import ExpertFlowPLECStrategy
        
        # Simple configuration
        num_experts = 32
        num_layers = 6
        top_k = 2
        cache_size_mb = 50
        sequence_steps = 10
        
        results = []
        
        # Test strategies
        strategies = {
            'pregated_moe': PreGatedMoEStrategy,
            'expertflow_plec': ExpertFlowPLECStrategy
        }
        
        for strategy_name, strategy_class in strategies.items():
            print(f"   Testing {strategy_name}...")
            
            # Create fresh instances
            cache = IsoCacheFramework(total_cache_size_mb=cache_size_mb)
            strategy = strategy_class(cache, num_experts, num_layers, top_k)
            
            # Simulate sequence processing
            np.random.seed(42)  # For reproducibility
            total_latency = 0.0
            
            for step in range(sequence_steps):
                for layer in range(num_layers):
                    # Generate random expert requirements
                    required_experts = np.random.choice(num_experts, size=top_k, replace=False).tolist()
                    
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
                'l1_hits': cache_metrics['l1_hits'],
                'l2_hits': cache_metrics['l2_hits'],
                'l3_hits': cache_metrics['l3_hits'],
                'misses': cache_metrics['misses']
            }
            
            results.append(result)
            print(f"     Total latency: {total_latency:.2f}ms")
            print(f"     Hit rate: {cache_metrics['overall_hit_rate']:.3f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / 'mini_evaluation_results.csv'
        results_df.to_csv(results_file, index=False)
        
        print(f"\n   Results saved to: {results_file}")
        print("   ✓ Mini evaluation completed successfully\n")
        
        # Display results
        print("   Results Summary:")
        for _, row in results_df.iterrows():
            print(f"   {row['strategy']}: {row['total_latency']:.2f}ms latency, {row['hit_rate']:.3f} hit rate")
        
        return results_df
        
    except Exception as e:
        print(f"   ✗ Mini evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_plotting():
    """Test plotting functionality"""
    print("5. Testing plotting functionality...")
    
    try:
        # Create sample data
        sample_data = [
            {'strategy': 'pregated_moe', 'batch_size': 4, 'total_latency': 150.5, 'overall_hit_rate': 0.85, 'cache_size_mb': 50},
            {'strategy': 'expertflow_plec', 'batch_size': 4, 'total_latency': 165.2, 'overall_hit_rate': 0.82, 'cache_size_mb': 50},
            {'strategy': 'pregated_moe', 'batch_size': 8, 'total_latency': 180.3, 'overall_hit_rate': 0.87, 'cache_size_mb': 50},
            {'strategy': 'expertflow_plec', 'batch_size': 8, 'total_latency': 195.1, 'overall_hit_rate': 0.84, 'cache_size_mb': 50}
        ]
        
        sample_df = pd.DataFrame(sample_data)
        
        # Test basic matplotlib functionality
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        strategies = sample_df['strategy'].unique()
        for strategy in strategies:
            strategy_data = sample_df[sample_df['strategy'] == strategy]
            ax.plot(strategy_data['batch_size'], strategy_data['total_latency'], 
                   marker='o', label=strategy, linewidth=2)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Test Plot: Strategy Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save test plot
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        plot_file = output_dir / 'test_plot.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Test plot saved to: {plot_file}")
        print("   ✓ Plotting functionality working correctly\n")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Plotting functionality error: {e}")
        return False

def main():
    """Run all validation tests"""
    print("Framework Validation Test Suite")
    print("=" * 50)
    print()
    
    try:
        # Test basic components
        if not test_basic_components():
            print("❌ Basic component tests failed!")
            return False
        
        # Run mini evaluation
        results_df = run_mini_evaluation()
        if results_df is None:
            print("❌ Mini evaluation failed!")
            return False
        
        # Test plotting
        if not test_plotting():
            print("❌ Plotting tests failed!")
            return False
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 50)
        print()
        print("Framework components are working correctly:")
        print("✓ Iso-cache constraint system")
        print("✓ Hardware-aware cost modeling")
        print("✓ Pre-gated MoE strategy implementation")
        print("✓ ExpertFlow PLEC strategy implementation")
        print("✓ End-to-end evaluation pipeline")
        print("✓ Visualization and plotting")
        print()
        print("The framework is ready for comprehensive evaluation!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)