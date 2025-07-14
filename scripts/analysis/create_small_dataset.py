#!/usr/bin/env python3
"""
Create a small experimental dataset from a single document
"""

import torch
import numpy as np
import pickle
import json
from pathlib import Path
import sys
import os

# Simple dataclass to represent gating data points
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GatingDataPoint:
    """Represents a single gating data point for training"""
    layer_id: int
    hidden_states: torch.Tensor
    input_embeddings: torch.Tensor
    target_routing: torch.Tensor
    target_top_k: torch.Tensor
    prev_layer_gates: List[torch.Tensor]
    sequence_length: int
    token_ids: Optional[torch.Tensor]
    dataset_name: str
    sample_id: str

def create_small_experimental_dataset():
    """Create a small dataset from a single document"""
    print("🔬 Creating small experimental dataset...")
    
    # Sample document about machine learning
    document = """
    Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. The field has revolutionized numerous industries through its ability to identify patterns in large datasets.
    
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. Reinforcement learning involves agents learning through interaction with an environment.
    
    Deep learning, a subset of machine learning, uses neural networks with multiple layers to process information. These networks can automatically learn hierarchical representations of data, making them particularly effective for tasks like image recognition and natural language processing.
    
    The Mixture of Experts (MoE) architecture is an advanced neural network design that uses multiple specialized sub-networks, called experts, to process different types of input. A gating network decides which experts to activate for each input, leading to more efficient computation and better performance on complex tasks.
    
    Expert routing in MoE models is crucial for performance. The routing mechanism determines which experts receive which inputs, and good routing strategies can significantly improve both accuracy and computational efficiency. This is why predicting expert routing patterns is an active area of research.
    """
    
    # Split into sentences for processing
    sentences = [s.strip() for s in document.split('.') if s.strip()]
    print(f"Processing {len(sentences)} sentences...")
    
    # Create synthetic traces for each sentence
    traces = []
    num_experts = 128
    layers = [1, 3, 5, 7, 9, 11]  # MoE layers
    
    for sent_idx, sentence in enumerate(sentences):
        if not sentence:
            continue
            
        # Tokenize roughly (simple word splitting)
        tokens = sentence.split()
        seq_len = min(len(tokens), 32)  # Limit sequence length
        
        if seq_len < 3:
            continue
            
        print(f"Processing sentence {sent_idx}: {sentence[:50]}...")
        
        for layer_idx in layers:
            # Create realistic expert routing patterns
            # Different patterns for different layers
            if layer_idx == 1:
                # Early layer: more diverse routing
                expert_probs = np.random.dirichlet(np.ones(num_experts) * 0.1, seq_len)
            elif layer_idx == 3:
                # Middle layer: some specialization
                dominant_experts = np.random.choice(num_experts, 3, replace=False)
                expert_probs = np.zeros((seq_len, num_experts))
                for i in range(seq_len):
                    chosen_expert = np.random.choice(dominant_experts)
                    expert_probs[i, chosen_expert] = 0.8
                    # Add some noise to other experts
                    noise = np.random.dirichlet(np.ones(num_experts) * 0.01)
                    expert_probs[i] = expert_probs[i] * 0.8 + noise * 0.2
            elif layer_idx == 5:
                # Middle layer: moderate specialization
                dominant_experts = np.random.choice(num_experts, 5, replace=False)
                expert_probs = np.zeros((seq_len, num_experts))
                for i in range(seq_len):
                    chosen_expert = np.random.choice(dominant_experts)
                    expert_probs[i, chosen_expert] = 0.7
                    # Add some noise to other experts
                    noise = np.random.dirichlet(np.ones(num_experts) * 0.02)
                    expert_probs[i] = expert_probs[i] * 0.7 + noise * 0.3
            elif layer_idx == 7:
                # Later layer: more specialization
                dominant_experts = np.random.choice(num_experts, 8, replace=False)
                expert_probs = np.zeros((seq_len, num_experts))
                for i in range(seq_len):
                    chosen_expert = np.random.choice(dominant_experts)
                    expert_probs[i, chosen_expert] = 0.6
                    # Add some noise to other experts
                    noise = np.random.dirichlet(np.ones(num_experts) * 0.03)
                    expert_probs[i] = expert_probs[i] * 0.6 + noise * 0.4
            elif layer_idx == 9:
                # Later layer: high specialization
                dominant_experts = np.random.choice(num_experts, 10, replace=False)
                expert_probs = np.zeros((seq_len, num_experts))
                for i in range(seq_len):
                    chosen_expert = np.random.choice(dominant_experts)
                    expert_probs[i, chosen_expert] = 0.5
                    # Add some noise to other experts
                    noise = np.random.dirichlet(np.ones(num_experts) * 0.05)
                    expert_probs[i] = expert_probs[i] * 0.5 + noise * 0.5
            else:  # layer_idx == 11
                # Final layer: very high specialization
                dominant_experts = np.random.choice(num_experts, 15, replace=False)
                expert_probs = np.zeros((seq_len, num_experts))
                for i in range(seq_len):
                    chosen_expert = np.random.choice(dominant_experts)
                    expert_probs[i, chosen_expert] = 0.4
                    # Add some noise to other experts
                    noise = np.random.dirichlet(np.ones(num_experts) * 0.08)
                    expert_probs[i] = expert_probs[i] * 0.4 + noise * 0.6
            
            # Normalize probabilities
            expert_probs = expert_probs / expert_probs.sum(axis=1, keepdims=True)
            
            # Create tensors
            hidden_states = torch.randn(seq_len, 512)  # Hidden dim
            target_routing = torch.tensor(expert_probs, dtype=torch.float32)
            target_top_k = torch.argmax(target_routing, dim=1).unsqueeze(1)
            token_ids = torch.randint(0, 1000, (seq_len,))
            
            # Create trace
            trace = GatingDataPoint(
                layer_id=layer_idx,
                hidden_states=hidden_states,
                input_embeddings=token_ids,
                target_routing=target_routing,
                target_top_k=target_top_k,
                prev_layer_gates=[],
                sequence_length=seq_len,
                token_ids=token_ids,
                dataset_name="small_experimental",
                sample_id=f"sent_{sent_idx}_layer_{layer_idx}"
            )
            traces.append(trace)
    
    print(f"Created {len(traces)} traces")
    return traces

def save_small_dataset(traces, output_file="routing_data/small_experimental_traces.pkl"):
    """Save small dataset"""
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"💾 Saving {len(traces)} traces to {output_file}")
    
    # Convert to serializable format
    serializable_traces = []
    for trace in traces:
        trace_dict = {
            'layer_id': trace.layer_id,
            'hidden_states': trace.hidden_states.numpy(),
            'input_embeddings': trace.input_embeddings.numpy(),
            'target_routing': trace.target_routing.numpy(),
            'target_top_k': trace.target_top_k.numpy(),
            'prev_layer_gates': [g.numpy() for g in trace.prev_layer_gates] if trace.prev_layer_gates else [],
            'sequence_length': trace.sequence_length,
            'token_ids': trace.token_ids.numpy() if trace.token_ids is not None else None,
            'dataset_name': trace.dataset_name,
            'sample_id': trace.sample_id
        }
        serializable_traces.append(trace_dict)
    
    with open(output_file, 'wb') as f:
        pickle.dump(serializable_traces, f)
    
    # Analyze and save metadata
    expert_usage = {}
    layer_distribution = {}
    
    for trace in traces:
        layer_distribution[trace.layer_id] = layer_distribution.get(trace.layer_id, 0) + 1
        
        top_experts = torch.argmax(trace.target_routing, dim=-1)
        for expert in top_experts.flatten():
            expert_usage[expert.item()] = expert_usage.get(expert.item(), 0) + 1
    
    metadata = {
        'total_traces': len(traces),
        'num_experts': 128,
        'expert_distribution': expert_usage,
        'experts_used': len(expert_usage),
        'diversity_percentage': len(expert_usage) / 128 * 100,
        'layer_distribution': layer_distribution,
        'dataset_type': 'small_experimental'
    }
    
    metadata_file = output_path.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved small dataset")
    print(f"   - Expert diversity: {len(expert_usage)}/128 experts ({len(expert_usage)/128*100:.1f}%)")
    print(f"   - Layer distribution: {layer_distribution}")
    print(f"   - File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    return output_file

def main():
    """Main function"""
    print("🚀 Creating Small Experimental Dataset")
    print("=" * 50)
    
    # Create traces
    traces = create_small_experimental_dataset()
    
    # Save dataset
    output_file = save_small_dataset(traces)
    
    print(f"\n✅ Small dataset created: {output_file}")
    print("Now you can analyze this smaller dataset for experiments!")

if __name__ == "__main__":
    main()