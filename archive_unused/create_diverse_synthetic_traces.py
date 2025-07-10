#!/usr/bin/env python3
"""
Create Diverse Synthetic MoE Traces for Testing
Generate realistic diverse routing patterns without needing huge models
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_diverse_routing_patterns(num_experts=16, num_samples=5000):
    """Create diverse realistic routing patterns"""
    
    logger.info(f"🎯 Creating diverse routing for {num_experts} experts")
    
    traces = []
    
    # Different routing strategies to simulate real MoE behavior
    routing_strategies = [
        'balanced',      # Equal expert usage
        'skewed',        # Some experts more popular
        'task_specific', # Different tasks use different experts
        'layer_dependent', # Different layers prefer different experts
        'sequence_aware'   # Position in sequence affects routing
    ]
    
    samples_per_strategy = num_samples // len(routing_strategies)
    
    for strategy_idx, strategy in enumerate(routing_strategies):
        logger.info(f"Generating {samples_per_strategy} samples with {strategy} routing")
        
        for sample_idx in tqdm(range(samples_per_strategy), desc=f"{strategy} routing"):
            
            # Create sample metadata
            layer_id = np.random.randint(0, 6)  # 6 layers like before
            seq_len = np.random.randint(8, 32)  # Variable sequence lengths
            hidden_size = 512
            
            # Generate hidden states (realistic)
            hidden_states = torch.randn(seq_len, hidden_size) * 0.5
            
            # Generate diverse routing based on strategy
            if strategy == 'balanced':
                # Roughly equal expert usage
                routing_logits = torch.randn(seq_len, num_experts)
                
            elif strategy == 'skewed':
                # Some experts much more popular
                routing_logits = torch.randn(seq_len, num_experts)
                # Make experts 0, 1, 2 more likely
                routing_logits[:, :3] += 2.0
                
            elif strategy == 'task_specific':
                # Different "tasks" use different expert subsets
                task_type = sample_idx % 4
                routing_logits = torch.randn(seq_len, num_experts) - 1.0  # Start negative
                
                if task_type == 0:  # "Language" task - experts 0-3
                    routing_logits[:, 0:4] += 3.0
                elif task_type == 1:  # "Math" task - experts 4-7  
                    routing_logits[:, 4:8] += 3.0
                elif task_type == 2:  # "Code" task - experts 8-11
                    routing_logits[:, 8:12] += 3.0
                else:  # "General" task - experts 12-15
                    routing_logits[:, 12:16] += 3.0
                    
            elif strategy == 'layer_dependent':
                # Different layers prefer different experts
                routing_logits = torch.randn(seq_len, num_experts) - 0.5
                preferred_experts = list(range(layer_id * 2, (layer_id + 1) * 2 + 2))
                for expert in preferred_experts:
                    if expert < num_experts:
                        routing_logits[:, expert] += 2.0
                        
            elif strategy == 'sequence_aware':
                # Position in sequence affects expert choice
                routing_logits = torch.randn(seq_len, num_experts)
                for pos in range(seq_len):
                    # Early positions prefer certain experts, later positions prefer others
                    if pos < seq_len // 3:
                        routing_logits[pos, :4] += 1.5  # Early experts
                    elif pos < 2 * seq_len // 3:
                        routing_logits[pos, 4:12] += 1.5  # Middle experts
                    else:
                        routing_logits[pos, 12:] += 1.5  # Late experts
            
            # Convert to probabilities
            target_routing = torch.softmax(routing_logits, dim=-1)
            
            # Get top-k indices
            top_k_results = torch.topk(target_routing, k=1, dim=-1)
            target_top_k = top_k_results.indices
            
            # Create previous layer gates (context)
            prev_layer_gates = []
            if layer_id > 0:
                # Generate 1-3 previous layers
                num_prev = min(layer_id, 3)
                for prev_idx in range(num_prev):
                    # Previous layer routing should be somewhat correlated but different
                    prev_routing_logits = routing_logits + torch.randn(seq_len, num_experts) * 0.5
                    prev_routing = torch.softmax(prev_routing_logits, dim=-1)
                    prev_layer_gates.append(prev_routing)
            
            # Create trace object
            from training.gating_data_collector import GatingDataPoint
            
            trace = GatingDataPoint(
                layer_id=layer_id,
                hidden_states=hidden_states,
                input_embeddings=torch.randint(0, 1000, (seq_len,)),  # Dummy token IDs
                target_routing=target_routing,
                target_top_k=target_top_k,
                prev_layer_gates=prev_layer_gates,
                sequence_length=seq_len,
                token_ids=torch.randint(0, 1000, (seq_len,)),
                dataset_name=f"{strategy}_routing",
                sample_id=f"{strategy}_{sample_idx}"
            )
            
            traces.append(trace)
    
    return traces

def analyze_routing_diversity(traces):
    """Analyze the diversity of routing patterns"""
    
    expert_usage = {}
    layer_expert_usage = {}
    
    for trace in traces:
        # Get expert choices
        routing = trace.target_routing
        expert_choices = torch.argmax(routing, dim=-1).tolist()
        
        # Count expert usage
        for expert in expert_choices:
            expert_usage[expert] = expert_usage.get(expert, 0) + 1
            
        # Count by layer
        layer_id = trace.layer_id
        if layer_id not in layer_expert_usage:
            layer_expert_usage[layer_id] = {}
        
        for expert in expert_choices:
            layer_expert_usage[layer_id][expert] = layer_expert_usage[layer_id].get(expert, 0) + 1
    
    return expert_usage, layer_expert_usage

def save_diverse_traces(traces, output_file="routing_data/diverse_synthetic_traces.pkl"):
    """Save diverse traces"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    logger.info(f"💾 Saving {len(traces)} diverse traces to {output_file}")
    
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
    expert_usage, layer_expert_usage = analyze_routing_diversity(traces)
    
    metadata = {
        'total_traces': len(traces),
        'num_experts': 16,
        'experts_used': len(expert_usage),
        'expert_distribution': expert_usage,
        'layer_expert_distribution': layer_expert_usage,
        'collection_time': time.time(),
        'routing_strategies': [
            'balanced', 'skewed', 'task_specific', 
            'layer_dependent', 'sequence_aware'
        ]
    }
    
    metadata_file = output_path.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Diverse traces and metadata saved")
    return output_file, expert_usage

def main():
    """Create diverse synthetic traces"""
    
    logger.info("🚀 Creating Diverse Synthetic MoE Traces")
    logger.info("=" * 50)
    
    # Create diverse traces
    num_experts = 16  # More experts for diversity
    num_samples = 8000  # More samples
    
    traces = create_diverse_routing_patterns(num_experts, num_samples)
    
    # Analyze diversity
    expert_usage, layer_expert_usage = analyze_routing_diversity(traces)
    
    logger.info(f"\\n📊 Routing Diversity Analysis:")
    logger.info(f"   Total traces: {len(traces)}")
    logger.info(f"   Experts used: {len(expert_usage)}/{num_experts}")
    logger.info(f"   Expert distribution:")
    
    # Show expert usage
    sorted_experts = sorted(expert_usage.items())
    for expert, count in sorted_experts[:8]:  # Show first 8
        percentage = count / sum(expert_usage.values()) * 100
        logger.info(f"     Expert {expert}: {count:,} ({percentage:.1f}%)")
    
    if len(sorted_experts) > 8:
        logger.info(f"     ... and {len(sorted_experts) - 8} more experts")
    
    # Check diversity quality
    diversity_ratio = len(expert_usage) / num_experts
    entropy = -sum((count/sum(expert_usage.values())) * np.log2(count/sum(expert_usage.values())) 
                   for count in expert_usage.values())
    max_entropy = np.log2(num_experts)
    normalized_entropy = entropy / max_entropy
    
    logger.info(f"\\n🎯 Diversity Metrics:")
    logger.info(f"   Expert coverage: {diversity_ratio:.1%}")
    logger.info(f"   Routing entropy: {entropy:.3f}/{max_entropy:.3f} ({normalized_entropy:.1%})")
    
    if diversity_ratio > 0.8 and normalized_entropy > 0.6:
        logger.info("✅ Excellent routing diversity!")
    elif diversity_ratio > 0.5 and normalized_entropy > 0.4:
        logger.info("✅ Good routing diversity!")
    else:
        logger.warning("⚠️  Limited routing diversity")
    
    # Save traces
    output_file, _ = save_diverse_traces(traces)
    
    logger.info(f"\\n🎉 SUCCESS!")
    logger.info(f"📁 File: {output_file}")
    logger.info(f"🚀 Ready for training diverse speculation models!")
    logger.info(f"   Next: python train_on_diverse_traces.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n✅ Diverse synthetic traces created!")
        print("🎯 Ready for realistic speculation training")
    else:
        print("\\n❌ Failed to create traces")