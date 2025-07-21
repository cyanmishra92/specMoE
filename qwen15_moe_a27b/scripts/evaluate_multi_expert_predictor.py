#!/usr/bin/env python3
"""
Comprehensive Evaluation for Qwen Multi-Expert Predictor
Evaluates top-1, top-3, top-5, top-10, top-20 accuracy for multi-expert routing
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import json
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from qwen_multi_expert_predictor import create_qwen_predictor

# Import datapoint class for pickle loading
sys.path.append(os.path.join(os.path.dirname(__file__), 'collection'))
from collect_qwen15_moe_traces_streaming import QwenMoEGatingDataPoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiExpertEvaluator:
    """Comprehensive evaluator for multi-expert prediction"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.num_experts = model.num_experts
        self.experts_per_token = model.experts_per_token
        
        # Metrics to track
        self.k_values = [1, 3, 5, 10, 20]
        self.metrics = {
            'top_k_accuracy': {k: [] for k in self.k_values},
            'exact_match': [],  # All 4 experts predicted correctly
            'partial_match': [],  # At least 1 expert predicted correctly
            'position_accuracy': [[] for _ in range(self.experts_per_token)],  # Accuracy per position
            'expert_distribution': defaultdict(int),  # Which experts are predicted most
            'confidence_scores': [],
            'routing_entropy': []
        }
    
    def evaluate_batch(self, hidden_states, target_indices, target_weights, attention_mask):
        """Evaluate a single batch"""
        with torch.no_grad():
            # Get predictions
            predictions = self.model(hidden_states, attention_mask)
            expert_logits = predictions['expert_logits']  # [batch, seq, 60]
            predicted_topk = predictions['top_k_indices']  # [batch, seq, 4]
            confidence = predictions['confidence']  # [batch, seq, 1]
            
            batch_size, seq_len = target_indices.shape[:2]
            
            # Get top-k predictions for different k values
            top_k_predictions = {}
            for k in self.k_values:
                if k <= self.num_experts:
                    _, top_k_indices = torch.topk(expert_logits, k=k, dim=-1)
                    top_k_predictions[k] = top_k_indices
            
            # Calculate metrics for each sequence position
            for b in range(batch_size):
                for s in range(seq_len):
                    if not attention_mask[b, s]:
                        continue  # Skip padded positions
                    
                    target = target_indices[b, s]  # [4] - target top-4 experts
                    pred_logits = expert_logits[b, s]  # [60] - predicted logits
                    
                    # Filter valid targets (< num_experts)
                    valid_targets = target[target < self.num_experts]
                    if len(valid_targets) == 0:
                        continue
                    
                    # Top-k accuracy for different k values
                    for k in self.k_values:
                        if k in top_k_predictions:
                            predicted_k = top_k_predictions[k][b, s]  # [k]
                            
                            # Check how many target experts are in predicted top-k
                            matches = sum(1 for expert in valid_targets if expert in predicted_k)
                            accuracy = matches / len(valid_targets)
                            self.metrics['top_k_accuracy'][k].append(accuracy)
                    
                    # Exact match (all 4 experts predicted correctly in top-4)
                    pred_top4 = predicted_topk[b, s]  # [4]
                    exact_match = all(expert in pred_top4 for expert in valid_targets)
                    self.metrics['exact_match'].append(float(exact_match))
                    
                    # Partial match (at least 1 expert predicted correctly)
                    partial_match = any(expert in pred_top4 for expert in valid_targets)
                    self.metrics['partial_match'].append(float(partial_match))
                    
                    # Position-wise accuracy (how well we predict each of the 4 positions)
                    for pos in range(min(self.experts_per_token, len(valid_targets))):
                        target_expert = target[pos]
                        if target_expert < self.num_experts:
                            # Check if this expert appears in predicted top-4
                            position_correct = target_expert in pred_top4
                            self.metrics['position_accuracy'][pos].append(float(position_correct))
                    
                    # Expert distribution (track which experts are predicted)
                    for expert in pred_top4:
                        if expert < self.num_experts:
                            self.metrics['expert_distribution'][expert.item()] += 1
                    
                    # Confidence scores
                    self.metrics['confidence_scores'].append(confidence[b, s, 0].item())
                    
                    # Routing entropy (diversity of predictions)
                    probs = F.softmax(pred_logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    self.metrics['routing_entropy'].append(entropy.item())
    
    def compute_final_metrics(self):
        """Compute final aggregated metrics"""
        results = {}
        
        # Top-k accuracies
        for k in self.k_values:
            if self.metrics['top_k_accuracy'][k]:
                results[f'top_{k}_accuracy'] = np.mean(self.metrics['top_k_accuracy'][k])
        
        # Exact and partial match rates
        results['exact_match_rate'] = np.mean(self.metrics['exact_match'])
        results['partial_match_rate'] = np.mean(self.metrics['partial_match'])
        
        # Position-wise accuracies
        for pos in range(self.experts_per_token):
            if self.metrics['position_accuracy'][pos]:
                results[f'position_{pos+1}_accuracy'] = np.mean(self.metrics['position_accuracy'][pos])
        
        # Expert usage statistics
        expert_counts = np.array(list(self.metrics['expert_distribution'].values()))
        results['expert_usage'] = {
            'mean_usage': np.mean(expert_counts),
            'std_usage': np.std(expert_counts),
            'max_usage': np.max(expert_counts),
            'min_usage': np.min(expert_counts),
            'num_used_experts': len(self.metrics['expert_distribution'])
        }
        
        # Confidence and entropy statistics
        results['confidence'] = {
            'mean': np.mean(self.metrics['confidence_scores']),
            'std': np.std(self.metrics['confidence_scores'])
        }
        
        results['routing_entropy'] = {
            'mean': np.mean(self.metrics['routing_entropy']),
            'std': np.std(self.metrics['routing_entropy'])
        }
        
        return results
    
    def create_visualizations(self, results, output_dir):
        """Create evaluation visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Top-k accuracy plot
        plt.figure(figsize=(10, 6))
        k_values = [k for k in self.k_values if f'top_{k}_accuracy' in results]
        accuracies = [results[f'top_{k}_accuracy'] for k in k_values]
        
        plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('k (Top-k Prediction)')
        plt.ylabel('Accuracy')
        plt.title('Top-k Accuracy for Multi-Expert Prediction')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        
        # Add value labels
        for k, acc in zip(k_values, accuracies):
            plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'top_k_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Position-wise accuracy
        plt.figure(figsize=(8, 6))
        positions = []
        pos_accuracies = []
        
        for pos in range(self.experts_per_token):
            if f'position_{pos+1}_accuracy' in results:
                positions.append(f'Pos {pos+1}')
                pos_accuracies.append(results[f'position_{pos+1}_accuracy'])
        
        if positions:
            plt.bar(positions, pos_accuracies, alpha=0.7, color='skyblue')
            plt.xlabel('Expert Position')
            plt.ylabel('Accuracy')
            plt.title('Accuracy by Expert Position (Top-4 Routing)')
            plt.ylim(0, 1)
            
            # Add value labels
            for i, acc in enumerate(pos_accuracies):
                plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'position_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Expert usage distribution
        if self.metrics['expert_distribution']:
            plt.figure(figsize=(12, 8))
            experts = list(self.metrics['expert_distribution'].keys())
            counts = list(self.metrics['expert_distribution'].values())
            
            plt.bar(range(len(experts)), counts, alpha=0.7)
            plt.xlabel('Expert ID')
            plt.ylabel('Usage Count')
            plt.title('Expert Usage Distribution')
            plt.xticks(range(0, len(experts), max(1, len(experts)//10)), 
                      experts[::max(1, len(experts)//10)], rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'expert_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")

def evaluate_model(model_path, shard_dir, output_dir, batch_size=16):
    """Main evaluation function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Evaluating on device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    config_file = Path(shard_dir) / "training_config.json"
    model = create_qwen_predictor(config_file if config_file.exists() else None)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Create evaluator
    evaluator = MultiExpertEvaluator(model, device)
    
    # Load test shards (use last 20% for testing)
    shard_dir = Path(shard_dir)
    shard_files = sorted(shard_dir.glob("shard_*_traces.pkl"))
    test_shards = shard_files[int(0.8 * len(shard_files)):]  # Last 20% for testing
    
    if not test_shards:
        logger.warning("No test shards found, using all shards")
        test_shards = shard_files[-2:]  # Use last 2 shards
    
    logger.info(f"Evaluating on {len(test_shards)} test shards")
    
    # Evaluate on test data
    total_samples = 0
    
    for shard_file in test_shards:
        logger.info(f"Evaluating shard: {shard_file.name}")
        
        try:
            # Load shard
            with open(shard_file, 'rb') as f:
                shard_traces = pickle.load(f)
            
            # Process in batches
            for i in tqdm(range(0, len(shard_traces), batch_size), desc="Processing"):
                batch_traces = shard_traces[i:i + batch_size]
                
                # Prepare batch data
                batch_size_actual = len(batch_traces)
                max_seq_len = 256
                
                hidden_states = torch.zeros(batch_size_actual, max_seq_len, 2048)
                target_indices = torch.full((batch_size_actual, max_seq_len, 4), 60, dtype=torch.long)
                target_weights = torch.zeros(batch_size_actual, max_seq_len, 4)
                attention_masks = torch.zeros(batch_size_actual, max_seq_len, dtype=torch.bool)
                
                for j, trace in enumerate(batch_traces):
                    # Handle different tensor shapes
                    trace_hidden = trace.hidden_states
                    trace_routing = trace.target_routing  
                    trace_topk = trace.target_top_k
                    
                    # Remove any batch dimension if present
                    while trace_hidden.dim() > 2:
                        trace_hidden = trace_hidden.squeeze(0)
                    while trace_routing.dim() > 2:
                        trace_routing = trace_routing.squeeze(0)
                    while trace_topk.dim() > 2:
                        trace_topk = trace_topk.squeeze(0)
                    
                    seq_len = min(trace_hidden.shape[0], max_seq_len)
                    
                    hidden_states[j, :seq_len] = trace_hidden[:seq_len]
                    target_indices[j, :seq_len] = trace_topk[:seq_len]
                    attention_masks[j, :seq_len] = True
                    
                    # Extract weights
                    for seq_idx in range(seq_len):
                        for k_idx in range(min(4, trace_topk.shape[1])):
                            expert_idx = trace_topk[seq_idx, k_idx]
                            if expert_idx < trace_routing.shape[1]:
                                target_weights[j, seq_idx, k_idx] = trace_routing[seq_idx, expert_idx]
                
                # Move to device and evaluate
                hidden_states = hidden_states.to(device)
                target_indices = target_indices.to(device)
                target_weights = target_weights.to(device)
                attention_masks = attention_masks.to(device)
                
                evaluator.evaluate_batch(hidden_states, target_indices, target_weights, attention_masks)
                total_samples += batch_size_actual
                
                # Memory cleanup
                del hidden_states, target_indices, target_weights, attention_masks
                torch.cuda.empty_cache()
            
            del shard_traces
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error processing shard {shard_file}: {e}")
            continue
    
    # Compute final results
    logger.info("Computing final metrics...")
    results = evaluator.compute_final_metrics()
    
    # Print results
    print("\n" + "="*60)
    print("MULTI-EXPERT PREDICTION EVALUATION RESULTS")
    print("="*60)
    
    print("\nTOP-K ACCURACIES:")
    for k in [1, 3, 5, 10, 20]:
        if f'top_{k}_accuracy' in results:
            print(f"  Top-{k:2d}: {results[f'top_{k}_accuracy']:.4f}")
    
    print(f"\nMATCH RATES:")
    print(f"  Exact Match (all 4 correct): {results['exact_match_rate']:.4f}")
    print(f"  Partial Match (≥1 correct):  {results['partial_match_rate']:.4f}")
    
    print(f"\nPOSITION-WISE ACCURACY:")
    for pos in range(4):
        if f'position_{pos+1}_accuracy' in results:
            print(f"  Position {pos+1}: {results[f'position_{pos+1}_accuracy']:.4f}")
    
    print(f"\nEXPERT USAGE STATISTICS:")
    usage = results['expert_usage']
    print(f"  Used Experts: {usage['num_used_experts']}/{evaluator.num_experts}")
    print(f"  Mean Usage: {usage['mean_usage']:.2f} ± {usage['std_usage']:.2f}")
    print(f"  Usage Range: {usage['min_usage']} - {usage['max_usage']}")
    
    print(f"\nCONFIDENCE & ENTROPY:")
    print(f"  Confidence: {results['confidence']['mean']:.4f} ± {results['confidence']['std']:.4f}")
    print(f"  Routing Entropy: {results['routing_entropy']['mean']:.4f} ± {results['routing_entropy']['std']:.4f}")
    
    print(f"\nEvaluated on {total_samples:,} samples")
    print("="*60)
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    evaluator.create_visualizations(results, output_dir)
    
    logger.info(f"Evaluation complete! Results saved to {output_dir}")
    return results

def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate Multi-Expert Predictor')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--shard-dir', default='../routing_data/shards', help='Directory containing test shards')
    parser.add_argument('--output-dir', default='../results/evaluation', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=16, help='Evaluation batch size')
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    script_dir = Path(__file__).parent.absolute()
    shard_dir = (script_dir / args.shard_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    model_path = Path(args.model).resolve()
    
    # Run evaluation
    results = evaluate_model(
        model_path=str(model_path),
        shard_dir=str(shard_dir),
        output_dir=str(output_dir),
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()