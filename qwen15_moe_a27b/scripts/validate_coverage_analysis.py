#!/usr/bin/env python3
"""
Coverage Probability Analysis for Expert Prediction Models
Tests how often all 4 target experts appear in top-k predictions (k=1,3,5,10,15,20,30)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import logging
import pickle
from tqdm import tqdm
import gc
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from simple_qwen_predictor import create_simple_qwen_predictor
from hybrid_expert_predictor import create_hybrid_predictor

# Import datapoint class for pickle loading
sys.path.append(os.path.join(os.path.dirname(__file__), 'collection'))
from collect_qwen15_moe_traces_streaming import QwenMoEGatingDataPoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """Analyze coverage probability for expert prediction"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.k_values = [1, 3, 5, 8, 10, 12, 15, 20, 25, 30]  # Extended k values
        
        # Metrics storage
        self.results = {
            'individual_accuracy': {k: [] for k in self.k_values},  # Individual expert accuracy
            'coverage_probability': {k: [] for k in self.k_values},  # All 4 experts covered
            'partial_coverage': {k: [] for k in self.k_values},     # At least 1 expert covered
            'average_coverage': {k: [] for k in self.k_values}      # Average number of experts covered
        }
        
        self.detailed_stats = {
            'position_analysis': defaultdict(list),  # Where targets appear in predictions
            'expert_frequency': defaultdict(int),    # How often each expert is predicted
            'target_frequency': defaultdict(int),    # How often each expert is target
            'confusion_matrix': np.zeros((60, 60))   # Predicted vs Target expert matrix
        }
    
    def analyze_batch(self, hidden_states, target_experts, attention_mask):
        """Analyze a single batch"""
        batch_size, seq_len = target_experts.shape[:2]
        
        # Get model predictions
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                predictions = self.model(hidden_states, attention_mask)
                if 'expert_logits' in predictions:
                    expert_logits = predictions['expert_logits']  # [batch, seq, 60]
                elif 'current_expert_probs' in predictions:
                    expert_logits = predictions['current_expert_probs']  # Hybrid model
                else:
                    raise ValueError("Unknown model output format")
            else:
                expert_logits = self.model(hidden_states, attention_mask)
        
        # Analyze each sequence position
        for b in range(batch_size):
            for s in range(seq_len):
                if not attention_mask[b, s]:
                    continue
                
                target_seq = target_experts[b, s]  # [4] target experts
                logits_seq = expert_logits[b, s]   # [60] prediction logits
                
                # Filter valid targets (< 60)
                valid_targets = target_seq[target_seq < 60].cpu().numpy()
                if len(valid_targets) == 0:
                    continue
                
                # Update target frequency
                for target in valid_targets:
                    self.detailed_stats['target_frequency'][target] += 1
                
                # Get predictions for different k values
                for k in self.k_values:
                    if k <= 60:  # Don't exceed number of experts
                        _, top_k_indices = torch.topk(logits_seq, k=k, dim=-1)
                        predicted_experts = top_k_indices.cpu().numpy()
                        
                        # Update expert frequency (only for k=10 to avoid skewing)
                        if k == 10:
                            for pred in predicted_experts:
                                self.detailed_stats['expert_frequency'][pred] += 1
                        
                        # Individual expert accuracy (how many targets are in top-k)
                        matches = sum(1 for target in valid_targets if target in predicted_experts)
                        individual_accuracy = matches / len(valid_targets)
                        self.results['individual_accuracy'][k].append(individual_accuracy)
                        
                        # Coverage probability (all targets covered)
                        full_coverage = all(target in predicted_experts for target in valid_targets)
                        self.results['coverage_probability'][k].append(float(full_coverage))
                        
                        # Partial coverage (at least one target covered)
                        partial_coverage = any(target in predicted_experts for target in valid_targets)
                        self.results['partial_coverage'][k].append(float(partial_coverage))
                        
                        # Average coverage (average number of targets found)
                        self.results['average_coverage'][k].append(matches)
                        
                        # Position analysis (where do targets appear in predictions)
                        if k == 20:  # Detailed analysis for top-20
                            for target in valid_targets:
                                if target in predicted_experts:
                                    position = np.where(predicted_experts == target)[0][0] + 1
                                    self.detailed_stats['position_analysis'][target].append(position)
                        
                        # Confusion matrix (only for top-4 predictions)
                        if k == 4:
                            for target in valid_targets:
                                for pred in predicted_experts:
                                    self.detailed_stats['confusion_matrix'][target, pred] += 1
    
    def compute_final_metrics(self):
        """Compute final aggregated metrics"""
        metrics = {}
        
        for k in self.k_values:
            if self.results['individual_accuracy'][k]:
                metrics[k] = {
                    'individual_accuracy': {
                        'mean': np.mean(self.results['individual_accuracy'][k]),
                        'std': np.std(self.results['individual_accuracy'][k]),
                        'samples': len(self.results['individual_accuracy'][k])
                    },
                    'coverage_probability': {
                        'mean': np.mean(self.results['coverage_probability'][k]),
                        'std': np.std(self.results['coverage_probability'][k]),
                        'samples': len(self.results['coverage_probability'][k])
                    },
                    'partial_coverage': {
                        'mean': np.mean(self.results['partial_coverage'][k]),
                        'std': np.std(self.results['partial_coverage'][k])
                    },
                    'average_coverage': {
                        'mean': np.mean(self.results['average_coverage'][k]),
                        'std': np.std(self.results['average_coverage'][k])
                    }
                }
        
        return metrics
    
    def print_results(self, metrics):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print("🎯 COVERAGE PROBABILITY ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\n📊 MAIN RESULTS (Total Samples: {metrics[10]['individual_accuracy']['samples']:,})")
        print("-" * 80)
        print(f"{'Top-K':<8} {'Individual':<12} {'Coverage':<12} {'Partial':<12} {'Avg Coverage':<12}")
        print(f"{'':8} {'Accuracy':<12} {'Probability':<12} {'Coverage':<12} {'(out of 4)':<12}")
        print("-" * 80)
        
        for k in self.k_values:
            if k in metrics:
                m = metrics[k]
                print(f"Top-{k:<4} {m['individual_accuracy']['mean']:<11.4f} "
                      f"{m['coverage_probability']['mean']:<11.4f} "
                      f"{m['partial_coverage']['mean']:<11.4f} "
                      f"{m['average_coverage']['mean']:<11.2f}")
        
        print("\n📈 KEY INSIGHTS")
        print("-" * 40)
        
        # Find optimal k for different coverage levels
        coverage_90 = next((k for k in self.k_values if k in metrics and 
                           metrics[k]['coverage_probability']['mean'] >= 0.90), None)
        coverage_95 = next((k for k in self.k_values if k in metrics and 
                           metrics[k]['coverage_probability']['mean'] >= 0.95), None)
        
        print(f"• For 90% coverage probability: Top-{coverage_90 or 'N/A'}")
        print(f"• For 95% coverage probability: Top-{coverage_95 or 'N/A'}")
        
        # Best individual accuracy
        best_k = max(self.k_values, key=lambda k: metrics[k]['individual_accuracy']['mean'] if k in metrics else 0)
        print(f"• Best individual accuracy: Top-{best_k} ({metrics[best_k]['individual_accuracy']['mean']:.4f})")
        
        # Memory trade-offs
        if 10 in metrics and 20 in metrics:
            coverage_10 = metrics[10]['coverage_probability']['mean']
            coverage_20 = metrics[20]['coverage_probability']['mean']
            print(f"• Memory trade-off: Top-10 ({coverage_10:.3f}) vs Top-20 ({coverage_20:.3f}) coverage")
    
    def save_detailed_analysis(self, save_dir):
        """Save detailed analysis and plots"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        metrics = self.compute_final_metrics()
        import json
        with open(save_dir / 'coverage_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot coverage probability vs k
        plt.figure(figsize=(12, 8))
        
        # Coverage probability plot
        plt.subplot(2, 2, 1)
        k_vals = [k for k in self.k_values if k in metrics]
        coverage_probs = [metrics[k]['coverage_probability']['mean'] for k in k_vals]
        individual_accs = [metrics[k]['individual_accuracy']['mean'] for k in k_vals]
        
        plt.plot(k_vals, coverage_probs, 'b-o', label='Coverage Probability', linewidth=2)
        plt.plot(k_vals, individual_accs, 'r-s', label='Individual Accuracy', linewidth=2)
        plt.axhline(y=0.90, color='g', linestyle='--', alpha=0.7, label='90% threshold')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        plt.xlabel('Top-K')
        plt.ylabel('Probability/Accuracy')
        plt.title('Coverage Probability vs Top-K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Average coverage plot
        plt.subplot(2, 2, 2)
        avg_coverage = [metrics[k]['average_coverage']['mean'] for k in k_vals]
        plt.plot(k_vals, avg_coverage, 'g-^', linewidth=2)
        plt.axhline(y=4, color='r', linestyle='--', alpha=0.7, label='Perfect (4/4)')
        plt.xlabel('Top-K')
        plt.ylabel('Average Experts Found')
        plt.title('Average Coverage vs Top-K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Expert frequency analysis
        plt.subplot(2, 2, 3)
        expert_freq = dict(sorted(self.detailed_stats['expert_frequency'].items()))
        target_freq = dict(sorted(self.detailed_stats['target_frequency'].items()))
        
        experts = list(expert_freq.keys())[:30]  # Show top 30 experts
        pred_freqs = [expert_freq.get(e, 0) for e in experts]
        targ_freqs = [target_freq.get(e, 0) for e in experts]
        
        x = np.arange(len(experts))
        plt.bar(x - 0.2, pred_freqs, 0.4, label='Predicted', alpha=0.7)
        plt.bar(x + 0.2, targ_freqs, 0.4, label='Target', alpha=0.7)
        plt.xlabel('Expert ID')
        plt.ylabel('Frequency')
        plt.title('Expert Frequency Analysis (Top 30)')
        plt.legend()
        plt.xticks(x, experts, rotation=45)
        
        # Position analysis for top-20
        plt.subplot(2, 2, 4)
        position_data = []
        for expert_positions in self.detailed_stats['position_analysis'].values():
            position_data.extend(expert_positions)
        
        if position_data:
            plt.hist(position_data, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Position in Top-20 Prediction')
            plt.ylabel('Frequency')
            plt.title('Target Expert Positions in Predictions')
            plt.axvline(x=np.mean(position_data), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(position_data):.1f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'coverage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Detailed analysis saved to: {save_dir}")


class SimpleDataLoader:
    """Simple data loader for validation"""
    
    def __init__(self, shard_files, batch_size=16):
        self.shard_files = shard_files
        self.batch_size = batch_size
        self.max_seq_len = 128
    
    def __iter__(self):
        """Process shards one by one"""
        for shard_idx, shard_file in enumerate(self.shard_files):
            logger.info(f"Processing validation shard {shard_idx}: {shard_file.name}")
            
            try:
                with open(shard_file, 'rb') as f:
                    shard_traces = pickle.load(f)
                
                # Process in batches
                for i in range(0, len(shard_traces), self.batch_size):
                    batch_traces = shard_traces[i:i + self.batch_size]
                    batch = self._collate_traces(batch_traces)
                    if batch is not None:
                        yield batch
                
                del shard_traces
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing shard {shard_idx}: {e}")
                continue
    
    def _collate_traces(self, traces):
        """Collate traces into batch"""
        try:
            batch_size = len(traces)
            hidden_states = torch.zeros(batch_size, self.max_seq_len, 2048)
            expert_indices = torch.full((batch_size, self.max_seq_len, 4), 60, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, self.max_seq_len, dtype=torch.bool)
            
            for i, trace in enumerate(traces):
                # Handle tensor shapes
                trace_hidden = trace.hidden_states
                trace_topk = trace.target_top_k
                
                # Take first sequence if batch dimension exists
                if trace_hidden.dim() > 2:
                    trace_hidden = trace_hidden[0]
                    trace_topk = trace_topk[0]
                
                seq_len = min(trace_hidden.shape[0], self.max_seq_len)
                
                hidden_states[i, :seq_len] = trace_hidden[:seq_len]
                expert_indices[i, :seq_len] = trace_topk[:seq_len]
                attention_mask[i, :seq_len] = True
            
            return {
                'hidden_states': hidden_states,
                'expert_indices': expert_indices,
                'attention_mask': attention_mask
            }
        except Exception as e:
            logger.warning(f"Error collating batch: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Coverage Probability Analysis')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--shard-dir', default='../routing_data/shards', help='Validation shard directory')
    parser.add_argument('--model-type', choices=['simple', 'hybrid'], default='simple', help='Model type')
    parser.add_argument('--batch-size', type=int, default=16, help='Validation batch size')
    parser.add_argument('--save-dir', default='../results/coverage_analysis', help='Save directory for results')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum validation samples (for quick testing)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running coverage analysis on device: {device}")
    
    # Load model
    logger.info(f"Loading {args.model_type} model from: {args.checkpoint}")
    
    if args.model_type == 'simple':
        model = create_simple_qwen_predictor()
    else:  # hybrid
        model = create_hybrid_predictor()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Best accuracy from training: {checkpoint.get('val_top1', 'N/A')}")
    
    # Load validation data
    shard_dir = Path(args.shard_dir)
    shard_files = sorted(shard_dir.glob("*.pkl"))
    
    if not shard_files:
        raise ValueError(f"No shard files found in {shard_dir}")
    
    # Use validation shards (last 20% of shards)
    val_shards = shard_files[int(0.8 * len(shard_files)):]
    logger.info(f"Using {len(val_shards)} validation shards for analysis")
    
    # Create data loader
    val_loader = SimpleDataLoader(val_shards, batch_size=args.batch_size)
    
    # Create analyzer
    analyzer = CoverageAnalyzer(model, device)
    
    # Run analysis
    logger.info("Starting coverage probability analysis...")
    sample_count = 0
    max_samples = args.max_samples or float('inf')
    
    for batch in tqdm(val_loader, desc="Analyzing coverage"):
        if sample_count >= max_samples:
            break
        
        hidden_states = batch['hidden_states'].to(device)
        expert_indices = batch['expert_indices'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        analyzer.analyze_batch(hidden_states, expert_indices, attention_mask)
        
        sample_count += hidden_states.shape[0] * hidden_states.shape[1]
        
        # Memory cleanup
        del hidden_states, expert_indices, attention_mask
        torch.cuda.empty_cache()
    
    # Compute and display results
    metrics = analyzer.compute_final_metrics()
    analyzer.print_results(metrics)
    
    # Save detailed analysis
    save_dir = Path(args.save_dir)
    analyzer.save_detailed_analysis(save_dir)
    
    logger.info(f"✅ Coverage analysis completed! Results saved to: {save_dir}")


if __name__ == "__main__":
    main()