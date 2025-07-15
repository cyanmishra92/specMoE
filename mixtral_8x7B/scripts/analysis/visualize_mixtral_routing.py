#!/usr/bin/env python3
"""
Visualize Mixtral 8x7B Expert Routing Patterns
Analysis and visualization of expert usage, routing patterns, and token journeys
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
import logging
from collections import defaultdict, Counter
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MixtralRoutingAnalyzer:
    """Analyzer for Mixtral 8x7B expert routing patterns"""
    
    def __init__(self, traces_path="routing_data/mixtral_8x7b_traces.pkl"):
        self.traces_path = Path(traces_path)
        self.traces = None
        self.num_experts = 8
        self.load_traces()
        
    def load_traces(self):
        """Load Mixtral traces"""
        if not self.traces_path.exists():
            logger.error(f"Traces not found at {self.traces_path}")
            logger.error("Please run collect_mixtral_traces.py first")
            return
            
        logger.info(f"Loading traces from {self.traces_path}")
        with open(self.traces_path, 'rb') as f:
            self.traces = pickle.load(f)
        
        logger.info(f"Loaded {len(self.traces)} traces")
        
    def analyze_expert_usage(self):
        """Analyze expert usage patterns across layers"""
        logger.info("Analyzing expert usage patterns...")
        
        # Collect expert usage by layer
        layer_expert_usage = defaultdict(Counter)
        
        for trace in self.traces:
            layer_id = trace.layer_id
            
            # Get top-2 experts (Mixtral uses top-2 routing)
            if hasattr(trace, 'target_top_k') and trace.target_top_k.numel() >= 2:
                top_experts = trace.target_top_k.flatten()[:2]
                for expert in top_experts:
                    if expert.item() < self.num_experts:
                        layer_expert_usage[layer_id][expert.item()] += 1
        
        return layer_expert_usage
    
    def plot_expert_usage_heatmap(self, layer_expert_usage, output_dir="routing_data"):
        """Plot expert usage heatmap"""
        logger.info("Creating expert usage heatmap...")
        
        # Create usage matrix
        layers = sorted(layer_expert_usage.keys())
        usage_matrix = np.zeros((len(layers), self.num_experts))
        
        for i, layer_id in enumerate(layers):
            for expert_id in range(self.num_experts):
                usage_matrix[i, expert_id] = layer_expert_usage[layer_id][expert_id]
        
        # Normalize by row (layer)
        usage_matrix = usage_matrix / (usage_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            usage_matrix,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            xticklabels=[f'Expert {i}' for i in range(self.num_experts)],
            yticklabels=[f'Layer {l}' for l in layers]
        )
        plt.title('Mixtral 8x7B Expert Usage by Layer (Top-2 Routing)')
        plt.xlabel('Expert ID')
        plt.ylabel('Layer ID')
        plt.tight_layout()
        
        output_path = Path(output_dir) / "mixtral_expert_usage_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Expert usage heatmap saved to {output_path}")
        
    def analyze_routing_patterns(self):
        """Analyze routing patterns and diversity"""
        logger.info("Analyzing routing patterns...")
        
        # Calculate routing diversity metrics
        layer_diversity = {}
        layer_expert_counts = defaultdict(Counter)
        
        for trace in self.traces:
            layer_id = trace.layer_id
            
            if hasattr(trace, 'target_top_k') and trace.target_top_k.numel() >= 2:
                top_experts = trace.target_top_k.flatten()[:2]
                for expert in top_experts:
                    if expert.item() < self.num_experts:
                        layer_expert_counts[layer_id][expert.item()] += 1
        
        # Calculate entropy for each layer
        for layer_id, expert_counts in layer_expert_counts.items():
            total_activations = sum(expert_counts.values())
            probabilities = np.array([expert_counts[i] / total_activations for i in range(self.num_experts)])
            probabilities = probabilities[probabilities > 0]  # Remove zeros
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            layer_diversity[layer_id] = {
                'entropy': entropy,
                'max_entropy': np.log2(self.num_experts),
                'normalized_entropy': entropy / np.log2(self.num_experts),
                'total_activations': total_activations,
                'active_experts': len([c for c in expert_counts.values() if c > 0])
            }
        
        return layer_diversity
    
    def plot_routing_diversity(self, layer_diversity, output_dir="routing_data"):
        """Plot routing diversity metrics"""
        logger.info("Creating routing diversity plots...")
        
        layers = sorted(layer_diversity.keys())
        entropies = [layer_diversity[l]['entropy'] for l in layers]
        normalized_entropies = [layer_diversity[l]['normalized_entropy'] for l in layers]
        active_experts = [layer_diversity[l]['active_experts'] for l in layers]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Entropy plot
        axes[0, 0].bar(layers, entropies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Expert Routing Entropy by Layer')
        axes[0, 0].set_xlabel('Layer ID')
        axes[0, 0].set_ylabel('Entropy (bits)')
        axes[0, 0].axhline(y=np.log2(self.num_experts), color='red', linestyle='--', label='Max Entropy')
        axes[0, 0].legend()
        
        # Normalized entropy plot
        axes[0, 1].bar(layers, normalized_entropies, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Normalized Expert Routing Entropy by Layer')
        axes[0, 1].set_xlabel('Layer ID')
        axes[0, 1].set_ylabel('Normalized Entropy')
        axes[0, 1].set_ylim(0, 1)
        
        # Active experts plot
        axes[1, 0].bar(layers, active_experts, color='orange', alpha=0.7)
        axes[1, 0].set_title('Active Experts by Layer')
        axes[1, 0].set_xlabel('Layer ID')
        axes[1, 0].set_ylabel('Number of Active Experts')
        axes[1, 0].axhline(y=self.num_experts, color='red', linestyle='--', label='Total Experts')
        axes[1, 0].legend()
        
        # Expert utilization distribution
        all_entropies = [layer_diversity[l]['normalized_entropy'] for l in layers]
        axes[1, 1].hist(all_entropies, bins=10, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Normalized Entropy Across Layers')
        axes[1, 1].set_xlabel('Normalized Entropy')
        axes[1, 1].set_ylabel('Number of Layers')
        
        plt.tight_layout()
        output_path = Path(output_dir) / "mixtral_routing_diversity.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Routing diversity plots saved to {output_path}")
        
    def create_token_journey_visualization(self, output_dir="routing_data"):
        """Create token journey visualization across MoE layers"""
        logger.info("Creating token journey visualization...")
        
        # Group traces by sample and create token journeys
        sample_traces = defaultdict(list)
        
        for trace in self.traces:
            sample_key = f"{trace.dataset_name}_{trace.sample_id.split('_seq_')[0]}"
            sample_traces[sample_key].append(trace)
        
        # Select a few samples for visualization
        selected_samples = list(sample_traces.keys())[:4]
        
        plt.figure(figsize=(16, 10))
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, sample_key in enumerate(selected_samples):
            traces = sample_traces[sample_key]
            
            # Sort traces by layer
            traces.sort(key=lambda x: x.layer_id)
            
            # Group by layer
            layer_traces = defaultdict(list)
            for trace in traces:
                layer_traces[trace.layer_id].append(trace)
            
            # Create journey for each sequence position
            sequence_positions = set()
            for trace in traces:
                if '_seq_' in trace.sample_id:
                    seq_pos = int(trace.sample_id.split('_seq_')[-1])
                    sequence_positions.add(seq_pos)
            
            # Plot journey for first few sequence positions
            for seq_pos in sorted(sequence_positions)[:3]:  # Show first 3 tokens
                x_coords = []
                y_coords = []
                
                for layer_id in sorted(layer_traces.keys()):
                    layer_trace_list = layer_traces[layer_id]
                    
                    # Find trace for this sequence position
                    seq_trace = None
                    for trace in layer_trace_list:
                        if f'_seq_{seq_pos}' in trace.sample_id:
                            seq_trace = trace
                            break
                    
                    if seq_trace and hasattr(seq_trace, 'target_top_k') and seq_trace.target_top_k.numel() >= 1:
                        expert_id = seq_trace.target_top_k.flatten()[0].item()
                        if expert_id < self.num_experts:
                            x_coords.append(layer_id)
                            y_coords.append(expert_id)
                
                if x_coords and y_coords:
                    plt.plot(x_coords, y_coords, 'o-', 
                            color=colors[i], alpha=0.7, markersize=8, linewidth=2,
                            label=f'Sample {i+1}, Token {seq_pos}')
        
        plt.title('Mixtral 8x7B Token Journeys Across MoE Layers')
        plt.xlabel('Layer ID')
        plt.ylabel('Expert ID')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set y-axis to show all experts
        plt.ylim(-0.5, self.num_experts - 0.5)
        plt.yticks(range(self.num_experts))
        
        plt.tight_layout()
        output_path = Path(output_dir) / "mixtral_token_journeys.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Token journey visualization saved to {output_path}")
        
    def generate_analysis_report(self, output_dir="routing_data"):
        """Generate comprehensive analysis report"""
        logger.info("Generating analysis report...")
        
        # Analyze patterns
        layer_expert_usage = self.analyze_expert_usage()
        layer_diversity = self.analyze_routing_patterns()
        
        # Calculate summary statistics
        total_traces = len(self.traces)
        unique_layers = len(set(trace.layer_id for trace in self.traces))
        unique_datasets = len(set(trace.dataset_name for trace in self.traces))
        
        # Generate report
        report = {
            "model": "Mixtral 8x7B",
            "routing_type": "top-2",
            "num_experts": self.num_experts,
            "total_traces": total_traces,
            "unique_layers": unique_layers,
            "unique_datasets": unique_datasets,
            "layer_diversity": layer_diversity,
            "summary": {
                "avg_entropy": np.mean([ld['entropy'] for ld in layer_diversity.values()]),
                "avg_normalized_entropy": np.mean([ld['normalized_entropy'] for ld in layer_diversity.values()]),
                "avg_active_experts": np.mean([ld['active_experts'] for ld in layer_diversity.values()]),
                "min_entropy": min([ld['entropy'] for ld in layer_diversity.values()]),
                "max_entropy": max([ld['entropy'] for ld in layer_diversity.values()])
            }
        }
        
        # Save report
        report_path = Path(output_dir) / "mixtral_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report saved to {report_path}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("MIXTRAL 8x7B ROUTING ANALYSIS SUMMARY")
        logger.info("="*50)
        logger.info(f"Total traces: {total_traces:,}")
        logger.info(f"Unique layers: {unique_layers}")
        logger.info(f"Unique datasets: {unique_datasets}")
        logger.info(f"Number of experts: {self.num_experts}")
        logger.info(f"Routing type: Top-2")
        logger.info(f"\nRouting Diversity:")
        logger.info(f"  Average entropy: {report['summary']['avg_entropy']:.3f}")
        logger.info(f"  Average normalized entropy: {report['summary']['avg_normalized_entropy']:.3f}")
        logger.info(f"  Average active experts: {report['summary']['avg_active_experts']:.1f}")
        logger.info(f"  Entropy range: {report['summary']['min_entropy']:.3f} - {report['summary']['max_entropy']:.3f}")
        logger.info("="*50)
        
        return report

def main():
    """Main analysis function"""
    logger.info("🚀 Starting Mixtral 8x7B Routing Analysis")
    
    # Create output directory
    output_dir = Path("routing_data")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = MixtralRoutingAnalyzer()
    
    if analyzer.traces is None:
        logger.error("No traces available for analysis")
        return
    
    # Perform analysis
    logger.info("📊 Analyzing expert usage patterns...")
    layer_expert_usage = analyzer.analyze_expert_usage()
    analyzer.plot_expert_usage_heatmap(layer_expert_usage, output_dir)
    
    logger.info("📈 Analyzing routing diversity...")
    layer_diversity = analyzer.analyze_routing_patterns()
    analyzer.plot_routing_diversity(layer_diversity, output_dir)
    
    logger.info("🎯 Creating token journey visualization...")
    analyzer.create_token_journey_visualization(output_dir)
    
    logger.info("📋 Generating analysis report...")
    report = analyzer.generate_analysis_report(output_dir)
    
    logger.info("🎉 Analysis complete!")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()