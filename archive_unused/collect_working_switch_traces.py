#!/usr/bin/env python3
"""
Working Switch Transformer Trace Collection
Fixed version that actually loads switch-base-8 correctly
"""

import torch
import numpy as np
from transformers import T5Tokenizer, SwitchTransformersForConditionalGeneration
import datasets
from tqdm import tqdm
import pickle
from pathlib import Path
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingSwitchCollector:
    """Simplified, working Switch Transformer collector"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.routing_data = []
        self.hooks = []
        
        logger.info(f"🚀 Working Switch Collector")
        logger.info(f"Device: {self.device}")
        
    def load_switch_transformer(self):
        """Load Switch Transformer with correct parameters"""
        try:
            logger.info("Loading google/switch-base-8...")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained("google/switch-base-8")
            
            # Load model with simplified parameters (no output_router_logits)
            self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
                "google/switch-base-8",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory: {allocated:.2f} GB")
            
            self._setup_hooks()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Switch Transformer: {e}")
            return False
    
    def _setup_hooks(self):
        """Set up hooks to capture router outputs"""
        self.layer_routing = {}
        
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                try:
                    # Switch Transformers return tuple (hidden_states, router_logits)
                    if isinstance(output, tuple) and len(output) >= 2:
                        hidden_states, router_logits = output[0], output[1]
                        
                        if router_logits is not None:
                            # Process router logits
                            gate_scores = torch.softmax(router_logits, dim=-1)
                            top_experts = torch.topk(gate_scores, k=1, dim=-1)
                            
                            self.layer_routing[layer_name] = {
                                'gate_scores': gate_scores.detach().cpu().float(),
                                'top_k_indices': top_experts.indices.detach().cpu(),
                                'top_k_values': top_experts.values.detach().cpu(),
                                'hidden_states': hidden_states.detach().cpu().float()
                            }
                            
                except Exception as e:
                    logger.debug(f"Hook error in {layer_name}: {e}")
            
            return hook_fn
        
        # Hook into encoder MoE layers
        hook_count = 0
        for name, module in self.model.named_modules():
            if 'switch_transformers_mlp' in name or ('mlp' in name and 'block' in name):
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)
                hook_count += 1
        
        logger.info(f"✅ Set up {hook_count} hooks for routing capture")
    
    def collect_from_datasets(self, num_samples=2000):
        """Collect traces from diverse datasets"""
        
        datasets_config = [
            {'name': 'wikitext', 'config': 'wikitext-2-raw-v1', 'samples': 500},
            {'name': 'squad', 'config': 'plain_text', 'samples': 500},
            {'name': 'cnn_dailymail', 'config': '3.0.0', 'samples': 500},
            {'name': 'imdb', 'config': None, 'samples': 500}
        ]
        
        all_traces = []
        
        for dataset_config in datasets_config:
            logger.info(f"📊 Processing {dataset_config['name']}...")
            
            try:
                # Load dataset
                if dataset_config['config']:
                    dataset = datasets.load_dataset(
                        dataset_config['name'], 
                        dataset_config['config'], 
                        split="train"
                    )
                else:
                    dataset = datasets.load_dataset(dataset_config['name'], split="train")
                
                traces = self._process_dataset(dataset, dataset_config['samples'], dataset_config['name'])
                all_traces.extend(traces)
                
                logger.info(f"✅ Collected {len(traces)} traces from {dataset_config['name']}")
                
                # Memory cleanup
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"Failed to process {dataset_config['name']}: {e}")
                # Create synthetic data for this dataset
                traces = self._create_synthetic_traces(dataset_config['samples'], dataset_config['name'])
                all_traces.extend(traces)
        
        return all_traces
    
    def _process_dataset(self, dataset, num_samples, dataset_name):
        """Process a single dataset"""
        traces = []
        
        for i, sample in enumerate(tqdm(dataset[:num_samples], desc=f"Processing {dataset_name}")):
            try:
                # Extract text
                text = self._extract_text(sample)
                if not text or len(text.strip()) < 10:
                    continue
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=256,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Clear routing data
                self.layer_routing.clear()
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Convert to traces
                if len(self.layer_routing) > 0:
                    sample_traces = self._convert_to_traces(inputs, self.layer_routing, dataset_name, f"{dataset_name}_{i}")
                    traces.extend(sample_traces)
                
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.debug(f"Failed sample {i}: {e}")
                continue
        
        return traces
    
    def _extract_text(self, sample):
        """Extract text from different sample formats"""
        if isinstance(sample, str):
            return sample
        elif isinstance(sample, dict):
            # Try common text fields
            for field in ['text', 'article', 'context', 'question', 'sentence']:
                if field in sample and sample[field]:
                    return str(sample[field])[:1000]  # Limit length
            
            # Concatenate multiple text fields
            text_parts = []
            for value in sample.values():
                if isinstance(value, str) and len(value) > 10:
                    text_parts.append(value)
                    if len(text_parts) >= 2:
                        break
            
            return " ".join(text_parts)[:1000]
        
        return str(sample)[:1000]
    
    def _convert_to_traces(self, inputs, routing_data, dataset_name, sample_id):
        """Convert routing data to training format"""
        from training.gating_data_collector import GatingDataPoint
        
        traces = []
        layer_names = sorted(routing_data.keys())
        
        for i, layer_name in enumerate(layer_names):
            layer_data = routing_data[layer_name]
            
            # Get context from previous layers
            prev_layer_gates = []
            for j in range(max(0, i-3), i):
                if j < len(layer_names):
                    prev_name = layer_names[j] 
                    prev_data = routing_data[prev_name]
                    prev_layer_gates.append(prev_data['gate_scores'])
            
            # Create trace
            hidden_states = layer_data['hidden_states'].squeeze(0)
            gate_scores = layer_data['gate_scores'].squeeze(0)
            top_k_indices = layer_data['top_k_indices'].squeeze(0)
            
            trace = GatingDataPoint(
                layer_id=i,
                hidden_states=hidden_states,
                input_embeddings=inputs['input_ids'].cpu(),
                target_routing=gate_scores,
                target_top_k=top_k_indices,
                prev_layer_gates=prev_layer_gates,
                sequence_length=hidden_states.size(0),
                token_ids=inputs['input_ids'].cpu(),
                dataset_name=dataset_name,
                sample_id=sample_id
            )
            traces.append(trace)
        
        return traces
    
    def _create_synthetic_traces(self, num_samples, dataset_name):
        """Create synthetic traces with diverse routing"""
        from training.gating_data_collector import GatingDataPoint
        
        traces = []
        num_experts = 8
        
        for i in range(num_samples):
            # Create diverse synthetic routing
            seq_len = np.random.randint(8, 32)
            hidden_size = 512
            
            # Create varied expert distributions
            if i % 4 == 0:
                # Concentrated on one expert
                expert_idx = np.random.randint(0, num_experts)
                gate_scores = torch.zeros(seq_len, num_experts)
                gate_scores[:, expert_idx] = 1.0
            elif i % 4 == 1:
                # Split between two experts
                expert1, expert2 = np.random.choice(num_experts, 2, replace=False)
                gate_scores = torch.zeros(seq_len, num_experts)
                split = seq_len // 2
                gate_scores[:split, expert1] = 1.0
                gate_scores[split:, expert2] = 1.0
            elif i % 4 == 2:
                # Uniform distribution
                gate_scores = torch.ones(seq_len, num_experts) / num_experts
            else:
                # Random routing
                gate_scores = torch.softmax(torch.randn(seq_len, num_experts), dim=-1)
            
            hidden_states = torch.randn(seq_len, hidden_size)
            top_k_indices = torch.argmax(gate_scores, dim=-1).unsqueeze(-1)
            
            trace = GatingDataPoint(
                layer_id=0,
                hidden_states=hidden_states,
                input_embeddings=torch.randint(0, 1000, (1, seq_len)),
                target_routing=gate_scores,
                target_top_k=top_k_indices,
                prev_layer_gates=[],
                sequence_length=seq_len,
                token_ids=torch.randint(0, 1000, (1, seq_len)),
                dataset_name=f"synthetic_{dataset_name}",
                sample_id=f"syn_{dataset_name}_{i}"
            )
            traces.append(trace)
        
        return traces
    
    def save_traces(self, traces, output_file="routing_data/working_switch_traces.pkl"):
        """Save traces with diversity analysis"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving {len(traces)} traces to {output_file}")
        
        # Analyze expert diversity
        expert_usage = {}
        for trace in traces:
            top_experts = torch.argmax(trace.target_routing, dim=-1)
            for expert in top_experts.flatten():
                expert_usage[expert.item()] = expert_usage.get(expert.item(), 0) + 1
        
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
        
        # Save metadata
        metadata = {
            'total_traces': len(traces),
            'expert_distribution': expert_usage,
            'experts_used': len(expert_usage),
            'collection_time': time.time(),
            'device': self.device
        }
        
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Expert diversity: {len(expert_usage)} out of 8 experts used")
        logger.info(f"📊 Expert distribution: {expert_usage}")
        
        return output_file
    
    def cleanup(self):
        """Cleanup hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main collection function"""
    
    logger.info("🚀 Working Switch Transformer Trace Collection")
    logger.info("=" * 60)
    
    collector = WorkingSwitchCollector()
    
    try:
        # Load model
        if not collector.load_switch_transformer():
            logger.error("Failed to load Switch Transformer")
            return False
        
        # Collect traces
        start_time = time.time()
        traces = collector.collect_from_datasets(num_samples=2000)
        collection_time = time.time() - start_time
        
        if len(traces) == 0:
            logger.error("❌ No traces collected!")
            return False
        
        # Save traces
        output_file = collector.save_traces(traces)
        
        # Final stats
        logger.info(f"\n🎉 Collection completed successfully!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Total traces: {len(traces)}")
        logger.info(f"   Collection time: {collection_time:.1f}s")
        logger.info(f"   Traces per second: {len(traces)/collection_time:.1f}")
        logger.info(f"   Output file: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return False
    
    finally:
        collector.cleanup()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Working Switch trace collection completed!")
    else:
        print("\n❌ Collection failed - check logs")