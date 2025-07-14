#!/usr/bin/env python3
"""
Realistic Inference Simulator for MoE Expert Speculation
Simulates actual inference with real routing traces and trained predictor
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
import seaborn as sns

from virtual_memory_manager import VirtualMemoryManager

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.improved_speculation_training import InterLayerSpeculationModel

@dataclass
class InferenceConfig:
    """Configuration for inference simulation"""
    model_path: str = "improved_speculation_best.pt"
    traces_path: str = "routing_data/realistic_traces.pkl"
    num_simulation_tokens: int = 500
    context_length: int = 3
    prediction_horizon: int = 2
    confidence_threshold: float = 0.3
    prefetch_k: int = 10
    gpu_memory_limit_mb: float = 2048
    enable_caching: bool = True
    save_results: bool = True

class InferenceSimulator:
    """Simulates realistic MoE inference with expert speculation"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = self._load_model()
        
        # Load routing traces
        self.traces = self._load_traces()
        
        # Load predictions
        self.predictions, self.confidences = self._load_predictions()
        
        # Initialize memory manager
        self.memory_manager = VirtualMemoryManager(
            num_experts=128,
            expert_size_mb=18.0,
            gpu_memory_limit_mb=config.gpu_memory_limit_mb,
            prefetch_k=config.prefetch_k,
            enable_caching=config.enable_caching
        )
        
        # Load benchmark results
        self.memory_manager.load_benchmark_results()
        
        # Results storage
        self.simulation_results = {}
    
    def _load_model(self) -> InterLayerSpeculationModel:
        """Load trained speculation model"""
        print(f"Loading model from: {self.config.model_path}")
        
        try:
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # Initialize model with known configuration
            model = InterLayerSpeculationModel(
                num_experts=128,
                model_dim=320,
                num_heads=10,
                ff_dim=1280,
                num_attention_layers=5,
                dropout=0.12
            )
            
            # Load weights directly (checkpoint is already a state dict)
            model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            
            print(f"Model loaded successfully: {sum(p.numel() for p in model.parameters())} parameters")
            return model
            
        except FileNotFoundError:
            print(f"Model file not found: {self.config.model_path}")
            print("Creating dummy model for simulation...")
            
            # Create dummy model
            model = InterLayerSpeculationModel(
                num_experts=128,
                model_dim=320,
                num_heads=10,
                ff_dim=1280,
                num_attention_layers=5,
                dropout=0.12
            )
            model.to(self.device)
            model.eval()
            return model
    
    def _load_traces(self) -> List[List[int]]:
        """Load routing traces"""
        print(f"Loading traces from: {self.config.traces_path}")
        
        try:
            with open(self.config.traces_path, 'rb') as f:
                traces_data = pickle.load(f)
            
            # Handle different trace formats
            if isinstance(traces_data, dict):
                traces = traces_data.get('expert_selections', traces_data.get('routing_traces', []))
            else:
                traces = traces_data
            
            # Limit to simulation size
            traces = traces[:self.config.num_simulation_tokens]
            
            print(f"Loaded {len(traces)} routing traces")
            return traces
            
        except FileNotFoundError:
            print(f"Traces file not found: {self.config.traces_path}")
            print("Generating synthetic traces...")
            
            # Generate synthetic traces
            traces = []
            for _ in range(self.config.num_simulation_tokens):
                # Create routing trace with some correlation
                trace = []
                prev_expert = np.random.randint(0, 128)
                
                for layer in range(12):  # 12 layers
                    # Add some correlation with previous layer
                    if np.random.random() < 0.3:  # 30% chance to reuse
                        expert = prev_expert
                    else:
                        expert = np.random.randint(0, 128)
                    
                    trace.append(expert)
                    prev_expert = expert
                
                traces.append(trace)
            
            return traces
    
    def _load_predictions(self) -> Tuple[List[List[List[int]]], List[List[List[float]]]]:
        """Load realistic predictions"""
        predictions_path = "routing_data/realistic_predictions.pkl"
        
        try:
            with open(predictions_path, 'rb') as f:
                pred_data = pickle.load(f)
            
            predictions = pred_data['predictions']
            confidences = pred_data['confidences']
            
            # Limit to simulation size
            predictions = predictions[:self.config.num_simulation_tokens]
            confidences = confidences[:self.config.num_simulation_tokens]
            
            print(f"Loaded {len(predictions)} prediction sets")
            return predictions, confidences
            
        except FileNotFoundError:
            print(f"Predictions file not found: {predictions_path}")
            print("Using fallback predictions...")
            
            # Create fallback predictions
            predictions = []
            confidences = []
            
            for trace in self.traces:
                token_predictions = []
                token_confidences = []
                
                for layer_id in range(len(trace)):
                    # Random predictions
                    preds = np.random.choice(128, self.config.prefetch_k, replace=False).tolist()
                    confs = np.random.uniform(0.1, 0.9, self.config.prefetch_k).tolist()
                    
                    token_predictions.append(preds)
                    token_confidences.append(confs)
                
                predictions.append(token_predictions)
                confidences.append(token_confidences)
            
            return predictions, confidences
    
    def _predict_experts(self, context_experts: List[int], layer_ids: List[int]) -> Tuple[List[int], List[float]]:
        """Predict experts using trained model"""
        try:
            # Prepare input tensors
            context_tensor = torch.tensor([context_experts], dtype=torch.long, device=self.device)
            layer_tensor = torch.tensor([layer_ids], dtype=torch.long, device=self.device)
            
            # Create attention mask
            attention_mask = torch.ones(1, len(context_experts), dtype=torch.bool, device=self.device)
            
            # Forward pass
            with torch.no_grad():
                expert_logits, confidence = self.model(context_tensor, layer_tensor, attention_mask)
            
            # Get top-k predictions
            top_k_logits, top_k_indices = torch.topk(expert_logits[0], k=self.config.prefetch_k)
            top_k_probs = torch.softmax(top_k_logits, dim=0)
            
            # Convert to lists
            predicted_experts = top_k_indices.cpu().tolist()
            confidence_scores = top_k_probs.cpu().tolist()
            
            return predicted_experts, confidence_scores
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return random predictions as fallback
            predicted_experts = np.random.choice(128, self.config.prefetch_k, replace=False).tolist()
            confidence_scores = np.random.uniform(0.1, 0.9, self.config.prefetch_k).tolist()
            return predicted_experts, confidence_scores
    
    def simulate_single_token(self, routing_trace: List[int], token_id: int) -> Dict:
        """Simulate inference for a single token"""
        token_metrics = {
            'token_id': token_id,
            'total_time_ms': 0.0,
            'prefetch_time_ms': 0.0,
            'access_time_ms': 0.0,
            'predictions': [],
            'layer_metrics': []
        }
        
        # Process each layer
        for layer_id in range(len(routing_trace)):
            layer_start_time = time.time()
            actual_expert = routing_trace[layer_id]
            
            layer_metrics = {
                'layer_id': layer_id,
                'actual_expert': actual_expert,
                'predicted_experts': [],
                'confidence_scores': [],
                'prefetch_time_ms': 0.0,
                'access_time_ms': 0.0,
                'hit_status': 'miss'
            }
            
            # Make prediction if we have sufficient context
            if layer_id >= self.config.context_length and token_id < len(self.predictions) and layer_id < len(self.predictions[token_id]):
                # Use loaded predictions
                predicted_experts = self.predictions[token_id][layer_id]
                confidence_scores = self.confidences[token_id][layer_id]
                
                layer_metrics['predicted_experts'] = predicted_experts
                layer_metrics['confidence_scores'] = confidence_scores
                
                # Prefetch experts
                prefetch_time = self.memory_manager.prefetch_experts(
                    predicted_experts, confidence_scores, layer_id, token_id, layer_start_time
                )
                layer_metrics['prefetch_time_ms'] = prefetch_time
                token_metrics['prefetch_time_ms'] += prefetch_time
                
                # Record prediction
                self.memory_manager.record_prediction(
                    predicted_experts, actual_expert, confidence_scores, layer_id, token_id, layer_start_time
                )
            
            # Access actual expert
            access_time, hit_status = self.memory_manager.access_expert(
                actual_expert, layer_id, token_id, layer_start_time
            )
            layer_metrics['access_time_ms'] = access_time
            layer_metrics['hit_status'] = hit_status
            token_metrics['access_time_ms'] += access_time
            
            # Add computation time (simulated)
            computation_time = 0.8  # ms per layer
            layer_metrics['computation_time_ms'] = computation_time
            
            token_metrics['layer_metrics'].append(layer_metrics)
        
        token_metrics['total_time_ms'] = token_metrics['prefetch_time_ms'] + token_metrics['access_time_ms']
        return token_metrics
    
    def run_simulation(self) -> Dict:
        """Run complete inference simulation"""
        print("Starting inference simulation...")
        print(f"Simulating {len(self.traces)} tokens with {len(self.traces[0])} layers each")
        
        simulation_start = time.time()
        token_results = []
        
        # Process each token
        for token_id, routing_trace in enumerate(self.traces):
            if token_id % 100 == 0:
                print(f"Processing token {token_id}/{len(self.traces)}")
            
            token_metrics = self.simulate_single_token(routing_trace, token_id)
            token_results.append(token_metrics)
        
        simulation_end = time.time()
        
        # Compile results
        memory_metrics = self.memory_manager.get_performance_metrics()
        
        self.simulation_results = {
            'config': self.config.__dict__,
            'token_results': token_results,
            'memory_metrics': memory_metrics,
            'simulation_time_seconds': simulation_end - simulation_start,
            'summary': self._calculate_summary_metrics(token_results, memory_metrics)
        }
        
        return self.simulation_results
    
    def _calculate_summary_metrics(self, token_results: List[Dict], memory_metrics: Dict) -> Dict:
        """Calculate summary metrics from simulation results"""
        total_tokens = len(token_results)
        total_layers = sum(len(token['layer_metrics']) for token in token_results)
        
        # Time metrics
        total_time = sum(token['total_time_ms'] for token in token_results)
        total_prefetch_time = sum(token['prefetch_time_ms'] for token in token_results)
        total_access_time = sum(token['access_time_ms'] for token in token_results)
        
        # Hit rate analysis
        hit_counts = {'hit': 0, 'miss': 0, 'cache_hit': 0}
        for token in token_results:
            for layer in token['layer_metrics']:
                hit_counts[layer['hit_status']] += 1
        
        total_accesses = sum(hit_counts.values())
        hit_rate = (hit_counts['hit'] + hit_counts['cache_hit']) / total_accesses if total_accesses > 0 else 0
        
        # Prediction accuracy
        correct_predictions = 0
        total_predictions = 0
        
        for token in token_results:
            for layer in token['layer_metrics']:
                if layer['predicted_experts']:
                    total_predictions += 1
                    if layer['actual_expert'] in layer['predicted_experts']:
                        correct_predictions += 1
        
        prediction_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate baseline time (without speculation)
        baseline_time = total_layers * 2.4  # 2.4ms per miss
        speedup = baseline_time / total_time if total_time > 0 else 1.0
        
        return {
            'total_tokens': total_tokens,
            'total_layers': total_layers,
            'total_time_ms': total_time,
            'avg_time_per_token_ms': total_time / total_tokens if total_tokens > 0 else 0,
            'avg_time_per_layer_ms': total_time / total_layers if total_layers > 0 else 0,
            'hit_rate': hit_rate,
            'prediction_accuracy': prediction_accuracy,
            'speedup': speedup,
            'prefetch_overhead_pct': (total_prefetch_time / total_time) * 100 if total_time > 0 else 0,
            'hit_counts': hit_counts,
            'baseline_time_ms': baseline_time,
            'time_saved_ms': baseline_time - total_time,
            'efficiency_score': (speedup - 1) / (memory_metrics.get('memory_utilization', 1) + 0.1)
        }
    
    def save_results(self, filename: str = "inference_simulation_results.json"):
        """Save simulation results"""
        if not self.simulation_results:
            print("No results to save. Run simulation first.")
            return
        
        output_path = Path("simulation_results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.simulation_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def plot_results(self):
        """Create comprehensive result visualizations"""
        if not self.simulation_results:
            print("No results to plot. Run simulation first.")
            return
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Time per token distribution
        token_times = [token['total_time_ms'] for token in self.simulation_results['token_results']]
        axes[0, 0].hist(token_times, bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Time per Token Distribution')
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Hit rate over time
        hit_rates = []
        window_size = 50
        for i in range(0, len(self.simulation_results['token_results']), window_size):
            window_tokens = self.simulation_results['token_results'][i:i+window_size]
            hits = sum(1 for token in window_tokens for layer in token['layer_metrics'] 
                      if layer['hit_status'] in ['hit', 'cache_hit'])
            total = sum(len(token['layer_metrics']) for token in window_tokens)
            hit_rates.append(hits / total if total > 0 else 0)
        
        axes[0, 1].plot(range(len(hit_rates)), hit_rates, linewidth=2, color='green')
        axes[0, 1].set_title('Hit Rate Over Time')
        axes[0, 1].set_xlabel('Time Window')
        axes[0, 1].set_ylabel('Hit Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction accuracy by layer
        layer_accuracies = [[] for _ in range(12)]
        for token in self.simulation_results['token_results']:
            for layer in token['layer_metrics']:
                if layer['predicted_experts']:
                    layer_id = layer['layer_id']
                    if layer_id < len(layer_accuracies):
                        accuracy = 1 if layer['actual_expert'] in layer['predicted_experts'] else 0
                        layer_accuracies[layer_id].append(accuracy)
        
        avg_accuracies = [np.mean(acc) if acc else 0 for acc in layer_accuracies]
        axes[0, 2].bar(range(len(avg_accuracies)), avg_accuracies, color='orange')
        axes[0, 2].set_title('Prediction Accuracy by Layer')
        axes[0, 2].set_xlabel('Layer ID')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Memory usage visualization
        memory_events = self.memory_manager.memory_events
        if memory_events:
            event_times = [i for i in range(len(memory_events))]
            memory_usage = [event.memory_usage_mb for event in memory_events]
            axes[1, 0].plot(event_times, memory_usage, linewidth=2, color='red')
            axes[1, 0].axhline(y=self.config.gpu_memory_limit_mb, color='black', linestyle='--', 
                              label='Memory Limit')
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Event Index')
            axes[1, 0].set_ylabel('Memory Usage (MB)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Latency breakdown
        summary = self.simulation_results['summary']
        hit_counts = summary['hit_counts']
        
        labels = list(hit_counts.keys())
        values = list(hit_counts.values())
        colors = ['green', 'red', 'blue']
        
        axes[1, 1].pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[1, 1].set_title('Memory Access Distribution')
        
        # 6. Speedup analysis
        speedup_data = {
            'Baseline': 1.0,
            'With Speculation': summary['speedup']
        }
        
        bars = axes[1, 2].bar(speedup_data.keys(), speedup_data.values(), 
                             color=['gray', 'green'])
        axes[1, 2].set_title('Inference Speedup')
        axes[1, 2].set_ylabel('Speedup Factor')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}×', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('simulation_results/inference_simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print simulation summary"""
        if not self.simulation_results:
            print("No results to summarize. Run simulation first.")
            return
        
        summary = self.simulation_results['summary']
        memory_metrics = self.simulation_results['memory_metrics']
        
        print("\n" + "="*60)
        print("INFERENCE SIMULATION SUMMARY")
        print("="*60)
        
        print(f"Configuration:")
        print(f"  Tokens simulated: {summary['total_tokens']}")
        print(f"  Layers per token: {summary['total_layers'] // summary['total_tokens']}")
        print(f"  Prefetch K: {self.config.prefetch_k}")
        print(f"  GPU Memory Limit: {self.config.gpu_memory_limit_mb} MB")
        
        print(f"\nPerformance Metrics:")
        print(f"  Total inference time: {summary['total_time_ms']:.2f} ms")
        print(f"  Average time per token: {summary['avg_time_per_token_ms']:.2f} ms")
        print(f"  Average time per layer: {summary['avg_time_per_layer_ms']:.2f} ms")
        
        print(f"\nAccuracy Metrics:")
        print(f"  Hit rate: {summary['hit_rate']:.2%}")
        print(f"  Prediction accuracy: {summary['prediction_accuracy']:.2%}")
        
        print(f"\nSpeedup Analysis:")
        print(f"  Baseline time: {summary['baseline_time_ms']:.2f} ms")
        print(f"  Actual time: {summary['total_time_ms']:.2f} ms")
        print(f"  Speedup: {summary['speedup']:.2f}×")
        print(f"  Time saved: {summary['time_saved_ms']:.2f} ms")
        
        print(f"\nMemory Analysis:")
        print(f"  Memory utilization: {memory_metrics.get('memory_utilization', 0):.2%}")
        print(f"  Cache hits: {summary['hit_counts']['cache_hit']}")
        print(f"  Prefetch hits: {summary['hit_counts']['hit']}")
        print(f"  Cache misses: {summary['hit_counts']['miss']}")
        
        print(f"\nEfficiency Score: {summary['efficiency_score']:.2f}")
        print("="*60)

def main():
    """Run inference simulation"""
    print("=== MoE Inference Simulation ===")
    
    # Configuration
    config = InferenceConfig(
        num_simulation_tokens=500,
        prefetch_k=10,
        gpu_memory_limit_mb=2048,
        enable_caching=True
    )
    
    # Initialize simulator
    simulator = InferenceSimulator(config)
    
    # Run simulation
    results = simulator.run_simulation()
    
    # Print summary
    simulator.print_summary()
    
    # Save results
    if config.save_results:
        simulator.save_results()
    
    # Create visualizations
    simulator.plot_results()
    
    # Save memory manager results
    simulator.memory_manager.save_simulation_results("memory_manager_results.json")

if __name__ == "__main__":
    main()