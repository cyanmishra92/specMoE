#!/usr/bin/env python3
"""
Expert Deduplication Analysis
Analyzes the impact of expert deduplication on memory transfer efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple, Dict
import json
from pathlib import Path
from collections import Counter

class ExpertDeduplicationAnalysis:
    """Analyzes expert deduplication patterns and efficiency"""
    
    def __init__(self, num_experts: int = 128):
        self.num_experts = num_experts
        self.analysis_results = {}
    
    def simulate_expert_requests(self, batch_size: int, experts_per_batch: int, 
                                num_simulations: int = 1000) -> Dict:
        """Simulate expert requests and analyze deduplication"""
        print(f"Simulating: batch_size={batch_size}, experts_per_batch={experts_per_batch}")
        
        total_requested = 0
        total_unique = 0
        duplicate_counts = []
        efficiency_ratios = []
        
        for _ in range(num_simulations):
            # Generate expert requests for each batch item
            all_experts = []
            for batch_item in range(batch_size):
                # Each batch item requests experts_per_batch experts
                experts = np.random.choice(
                    self.num_experts, 
                    size=experts_per_batch, 
                    replace=False
                ).tolist()
                all_experts.extend(experts)
            
            # Count total and unique experts
            total_experts = len(all_experts)
            unique_experts = len(set(all_experts))
            
            total_requested += total_experts
            total_unique += unique_experts
            
            # Calculate metrics
            duplicate_count = total_experts - unique_experts
            efficiency_ratio = unique_experts / total_experts
            
            duplicate_counts.append(duplicate_count)
            efficiency_ratios.append(efficiency_ratio)
        
        # Calculate statistics
        avg_requested = total_requested / num_simulations
        avg_unique = total_unique / num_simulations
        avg_duplicates = np.mean(duplicate_counts)
        avg_efficiency = np.mean(efficiency_ratios)
        
        results = {
            'batch_size': batch_size,
            'experts_per_batch': experts_per_batch,
            'avg_requested': avg_requested,
            'avg_unique': avg_unique,
            'avg_duplicates': avg_duplicates,
            'avg_efficiency': avg_efficiency,
            'efficiency_std': np.std(efficiency_ratios),
            'duplicate_counts': duplicate_counts,
            'efficiency_ratios': efficiency_ratios,
            'theoretical_max': batch_size * experts_per_batch,
            'memory_savings_pct': (1 - avg_efficiency) * 100
        }
        
        return results
    
    def analyze_all_configurations(self) -> Dict:
        """Analyze all batch size and expert count combinations"""
        batch_sizes = [1, 2, 4, 8, 16]
        expert_counts = [1, 3, 5, 10, 20]
        
        results = {}
        
        for batch_size in batch_sizes:
            for expert_count in expert_counts:
                config_key = f"batch_{batch_size}_experts_{expert_count}"
                results[config_key] = self.simulate_expert_requests(
                    batch_size, expert_count, num_simulations=1000
                )
        
        self.analysis_results = results
        return results
    
    def calculate_theoretical_efficiency(self, batch_size: int, experts_per_batch: int) -> float:
        """Calculate theoretical deduplication efficiency"""
        # Probability that k experts are unique when drawing from n experts
        # This is a hypergeometric distribution problem
        total_requests = batch_size * experts_per_batch
        
        if total_requests <= self.num_experts:
            # If we can't exceed the number of experts, efficiency depends on overlap
            # Expected number of unique experts (approximate)
            prob_unique = 1.0
            for i in range(total_requests):
                prob_unique *= (self.num_experts - i) / self.num_experts
            
            expected_unique = total_requests * prob_unique
            return expected_unique / total_requests
        else:
            # If we exceed the number of experts, efficiency is limited
            return min(1.0, self.num_experts / total_requests)
    
    def create_efficiency_heatmap(self):
        """Create efficiency heatmap"""
        if not self.analysis_results:
            self.analyze_all_configurations()
        
        # Prepare data for heatmap
        batch_sizes = []
        expert_counts = []
        efficiencies = []
        
        for config_key, results in self.analysis_results.items():
            batch_size = results['batch_size']
            expert_count = results['experts_per_batch']
            efficiency = results['avg_efficiency']
            
            batch_sizes.append(batch_size)
            expert_counts.append(expert_count)
            efficiencies.append(efficiency)
        
        # Create pivot table
        df = pd.DataFrame({
            'Batch Size': batch_sizes,
            'Experts per Batch': expert_counts,
            'Efficiency': efficiencies
        })
        
        pivot_table = df.pivot(index='Experts per Batch', 
                              columns='Batch Size', 
                              values='Efficiency')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Deduplication Efficiency'})
        plt.title('Expert Deduplication Efficiency\n(Lower is better for memory savings)')
        plt.xlabel('Batch Size')
        plt.ylabel('Experts per Batch Item')
        plt.tight_layout()
        plt.savefig('benchmarks/expert_deduplication_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_memory_savings_analysis(self):
        """Create memory savings analysis"""
        if not self.analysis_results:
            self.analyze_all_configurations()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Memory savings by batch size
        batch_sizes = [1, 2, 4, 8, 16]
        expert_count = 10  # Fixed expert count
        
        memory_savings = []
        for batch_size in batch_sizes:
            config_key = f"batch_{batch_size}_experts_{expert_count}"
            if config_key in self.analysis_results:
                savings = self.analysis_results[config_key]['memory_savings_pct']
                memory_savings.append(savings)
            else:
                memory_savings.append(0)
        
        axes[0, 0].bar(batch_sizes, memory_savings, color='skyblue')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Memory Savings (%)')
        axes[0, 0].set_title(f'Memory Savings by Batch Size\n(Fixed {expert_count} experts per batch)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Memory savings by expert count
        expert_counts = [1, 3, 5, 10, 20]
        batch_size = 4  # Fixed batch size
        
        memory_savings = []
        for expert_count in expert_counts:
            config_key = f"batch_{batch_size}_experts_{expert_count}"
            if config_key in self.analysis_results:
                savings = self.analysis_results[config_key]['memory_savings_pct']
                memory_savings.append(savings)
            else:
                memory_savings.append(0)
        
        axes[0, 1].bar(expert_counts, memory_savings, color='lightgreen')
        axes[0, 1].set_xlabel('Experts per Batch Item')
        axes[0, 1].set_ylabel('Memory Savings (%)')
        axes[0, 1].set_title(f'Memory Savings by Expert Count\n(Fixed batch size {batch_size})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Efficiency distribution
        all_efficiencies = []
        for results in self.analysis_results.values():
            all_efficiencies.extend(results['efficiency_ratios'])
        
        axes[1, 0].hist(all_efficiencies, bins=50, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Deduplication Efficiency')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Deduplication Efficiency')
        axes[1, 0].axvline(np.mean(all_efficiencies), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_efficiencies):.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Theoretical vs actual efficiency
        theoretical_effs = []
        actual_effs = []
        
        for config_key, results in self.analysis_results.items():
            batch_size = results['batch_size']
            expert_count = results['experts_per_batch']
            
            theoretical = self.calculate_theoretical_efficiency(batch_size, expert_count)
            actual = results['avg_efficiency']
            
            theoretical_effs.append(theoretical)
            actual_effs.append(actual)
        
        axes[1, 1].scatter(theoretical_effs, actual_effs, alpha=0.7, color='purple')
        axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect match')
        axes[1, 1].set_xlabel('Theoretical Efficiency')
        axes[1, 1].set_ylabel('Actual Efficiency')
        axes[1, 1].set_title('Theoretical vs Actual Efficiency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmarks/memory_savings_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_memory_transfer_benefits(self):
        """Calculate actual memory transfer time benefits"""
        if not self.analysis_results:
            self.analyze_all_configurations()
        
        # Assume 27MB per expert and 3.5ms transfer time per expert
        expert_size_mb = 27.0
        transfer_time_per_expert_ms = 3.5
        
        benefits = {}
        
        for config_key, results in self.analysis_results.items():
            batch_size = results['batch_size']
            expert_count = results['experts_per_batch']
            
            # Without deduplication
            total_requested = results['avg_requested']
            time_without_dedup = total_requested * transfer_time_per_expert_ms
            memory_without_dedup = total_requested * expert_size_mb
            
            # With deduplication
            total_unique = results['avg_unique']
            time_with_dedup = total_unique * transfer_time_per_expert_ms
            memory_with_dedup = total_unique * expert_size_mb
            
            # Calculate benefits
            time_saved = time_without_dedup - time_with_dedup
            memory_saved = memory_without_dedup - memory_with_dedup
            speedup = time_without_dedup / time_with_dedup if time_with_dedup > 0 else 1.0
            
            benefits[config_key] = {
                'batch_size': batch_size,
                'expert_count': expert_count,
                'time_without_dedup_ms': time_without_dedup,
                'time_with_dedup_ms': time_with_dedup,
                'time_saved_ms': time_saved,
                'memory_without_dedup_mb': memory_without_dedup,
                'memory_with_dedup_mb': memory_with_dedup,
                'memory_saved_mb': memory_saved,
                'speedup': speedup,
                'efficiency': results['avg_efficiency']
            }
        
        return benefits
    
    def print_summary(self):
        """Print comprehensive summary"""
        if not self.analysis_results:
            self.analyze_all_configurations()
        
        print("\n" + "=" * 60)
        print("EXPERT DEDUPLICATION ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Calculate overall statistics
        all_efficiencies = []
        all_savings = []
        
        for results in self.analysis_results.values():
            all_efficiencies.extend(results['efficiency_ratios'])
            all_savings.append(results['memory_savings_pct'])
        
        print(f"Overall Statistics:")
        print(f"  Average deduplication efficiency: {np.mean(all_efficiencies):.3f}")
        print(f"  Average memory savings: {np.mean(all_savings):.1f}%")
        print(f"  Maximum memory savings: {np.max(all_savings):.1f}%")
        
        # Calculate memory transfer benefits
        benefits = self.calculate_memory_transfer_benefits()
        
        print(f"\nMemory Transfer Benefits:")
        
        # Find best configurations
        best_speedup = max(benefits.values(), key=lambda x: x['speedup'])
        best_memory_savings = max(benefits.values(), key=lambda x: x['memory_saved_mb'])
        
        print(f"  Best speedup: {best_speedup['speedup']:.2f}× (batch {best_speedup['batch_size']}, {best_speedup['expert_count']} experts)")
        print(f"  Best memory savings: {best_memory_savings['memory_saved_mb']:.1f} MB (batch {best_memory_savings['batch_size']}, {best_memory_savings['expert_count']} experts)")
        
        # Show impact for common configurations
        print(f"\nCommon Configuration Analysis:")
        
        common_configs = [
            ('batch_4_experts_10', 'Batch 4, 10 experts'),
            ('batch_8_experts_10', 'Batch 8, 10 experts'),
            ('batch_16_experts_10', 'Batch 16, 10 experts')
        ]
        
        for config_key, description in common_configs:
            if config_key in benefits:
                b = benefits[config_key]
                print(f"  {description}:")
                print(f"    Without dedup: {b['time_without_dedup_ms']:.1f}ms, {b['memory_without_dedup_mb']:.1f}MB")
                print(f"    With dedup: {b['time_with_dedup_ms']:.1f}ms, {b['memory_with_dedup_mb']:.1f}MB")
                print(f"    Savings: {b['time_saved_ms']:.1f}ms ({b['speedup']:.2f}×), {b['memory_saved_mb']:.1f}MB")
        
        # Recommendations
        print(f"\nRecommendations:")
        print(f"  • Deduplication provides significant benefits for batch_size > 1")
        print(f"  • Memory savings increase with batch size and expert count")
        print(f"  • Implement deduplication for production inference")
        print(f"  • Consider expert caching for additional benefits")
    
    def save_results(self, filename: str = "expert_deduplication_analysis.json"):
        """Save analysis results"""
        output_path = Path("benchmarks") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for config_key, results in self.analysis_results.items():
            json_results[config_key] = {
                'batch_size': int(results['batch_size']),
                'experts_per_batch': int(results['experts_per_batch']),
                'avg_requested': float(results['avg_requested']),
                'avg_unique': float(results['avg_unique']),
                'avg_duplicates': float(results['avg_duplicates']),
                'avg_efficiency': float(results['avg_efficiency']),
                'efficiency_std': float(results['efficiency_std']),
                'theoretical_max': int(results['theoretical_max']),
                'memory_savings_pct': float(results['memory_savings_pct'])
            }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")

def main():
    """Run expert deduplication analysis"""
    print("=" * 80)
    print("EXPERT DEDUPLICATION ANALYSIS")
    print("=" * 80)
    
    # Initialize analysis
    analyzer = ExpertDeduplicationAnalysis(num_experts=128)
    
    # Run analysis
    results = analyzer.analyze_all_configurations()
    
    # Create visualizations
    analyzer.create_efficiency_heatmap()
    analyzer.create_memory_savings_analysis()
    
    # Print summary
    analyzer.print_summary()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()