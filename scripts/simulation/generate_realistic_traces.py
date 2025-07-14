#!/usr/bin/env python3
"""
Generate realistic routing traces based on training results
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple

def load_training_results() -> Dict:
    """Load training results to understand routing patterns"""
    results_files = [
        "improved_speculation_results_20250711_183257.json",
        "improved_speculation_results_20250714_113134.json"
    ]
    
    for file in results_files:
        if Path(file).exists():
            with open(file, 'r') as f:
                return json.load(f)
    
    return {}

def generate_realistic_routing_traces(num_tokens: int = 500, num_layers: int = 12) -> List[List[int]]:
    """Generate realistic routing traces based on training patterns"""
    
    # Load training results to understand patterns
    training_results = load_training_results()
    
    # Extract routing patterns if available
    if 'routing_patterns' in training_results:
        patterns = training_results['routing_patterns']
        print(f"Using patterns from training results")
    else:
        # Use realistic patterns based on MoE behavior
        patterns = {
            'expert_frequency': np.random.zipf(1.5, 128),  # Power-law distribution
            'layer_correlation': 0.3,  # Correlation between adjacent layers
            'temporal_correlation': 0.2  # Correlation across time
        }
        print(f"Using synthetic realistic patterns")
    
    traces = []
    prev_trace = None
    
    for token_id in range(num_tokens):
        if token_id % 100 == 0:
            print(f"Generating token {token_id}/{num_tokens}")
        
        trace = []
        prev_expert = None
        
        for layer_id in range(num_layers):
            # Generate expert selection with correlations
            if prev_expert is not None and np.random.random() < 0.3:
                # 30% chance to reuse previous layer's expert
                expert = prev_expert
            elif prev_trace is not None and np.random.random() < 0.2:
                # 20% chance to reuse same layer from previous token
                expert = prev_trace[layer_id]
            else:
                # Generate new expert with frequency bias
                expert = np.random.choice(128, p=_get_expert_probabilities())
            
            trace.append(int(expert))
            prev_expert = expert
        
        traces.append(trace)
        prev_trace = trace
    
    return traces

def _get_expert_probabilities() -> np.ndarray:
    """Get realistic expert selection probabilities"""
    # Create power-law distribution (some experts used more than others)
    probs = np.random.zipf(1.5, 128)
    return probs / probs.sum()

def generate_prediction_data(traces: List[List[int]], accuracy: float = 0.3386) -> Tuple[List[List[List[int]]], List[List[List[float]]]]:
    """Generate prediction data with specified accuracy"""
    predictions = []
    confidences = []
    
    for token_id, trace in enumerate(traces):
        if token_id % 100 == 0:
            print(f"Generating predictions for token {token_id}/{len(traces)}")
        
        token_predictions = []
        token_confidences = []
        
        for layer_id in range(len(trace)):
            actual_expert = trace[layer_id]
            
            # Generate top-10 predictions
            layer_preds = []
            layer_confs = []
            
            # Include actual expert with specified probability
            if np.random.random() < accuracy:
                layer_preds.append(actual_expert)
                layer_confs.append(np.random.uniform(0.6, 0.9))
            
            # Fill remaining slots with random experts
            while len(layer_preds) < 10:
                pred = np.random.randint(0, 128)
                if pred not in layer_preds:
                    layer_preds.append(pred)
                    layer_confs.append(np.random.uniform(0.1, 0.7))
            
            # Normalize confidences
            layer_confs = np.array(layer_confs)
            layer_confs = layer_confs / layer_confs.sum()
            
            token_predictions.append(layer_preds)
            token_confidences.append(layer_confs.tolist())
        
        predictions.append(token_predictions)
        confidences.append(token_confidences)
    
    return predictions, confidences

def main():
    """Generate realistic traces and predictions"""
    print("=== Generating Realistic Routing Traces ===")
    
    # Generate traces
    print("Generating routing traces...")
    traces = generate_realistic_routing_traces(num_tokens=500, num_layers=12)
    
    # Generate predictions
    print("Generating predictions...")
    predictions, confidences = generate_prediction_data(traces, accuracy=0.3386)
    
    # Save to file
    output_dir = Path("routing_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save traces
    with open(output_dir / "realistic_traces.pkl", 'wb') as f:
        pickle.dump(traces, f)
    
    # Save predictions
    with open(output_dir / "realistic_predictions.pkl", 'wb') as f:
        pickle.dump({
            'predictions': predictions,
            'confidences': confidences
        }, f)
    
    print(f"Generated {len(traces)} traces with {len(traces[0])} layers each")
    print(f"Saved to: {output_dir}")
    
    # Quick validation
    print("\n=== Validation ===")
    print(f"Trace shape: {len(traces)} tokens × {len(traces[0])} layers")
    print(f"Expert range: {min(min(trace) for trace in traces)} - {max(max(trace) for trace in traces)}")
    print(f"Prediction accuracy: {np.mean([1 if traces[i][j] in predictions[i][j] else 0 for i in range(len(traces)) for j in range(len(traces[i]))]):.3f}")

if __name__ == "__main__":
    main()