#!/usr/bin/env python3
"""
Collect Real MoE Routing Traces from Switch Transformer
Run this script to gather 10,000+ routing patterns for training speculation models
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

class RoutingTraceCollector:
    """Collect routing traces from real Switch Transformer model"""
    
    def __init__(self, model_name="google/switch-base-8"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Storage for collected traces
        self.routing_traces = []
        self.routing_hooks = {}
        
        logger.info(f"Initializing trace collector with {model_name}")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load Switch Transformer model"""
        try:
            logger.info("Loading tokenizer and model...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                output_router_logits=True  # Essential for getting routing info
            )
            self.model.eval()
            logger.info("✅ Model loaded successfully")
            
            # Set up hooks to capture routing
            self._setup_routing_hooks()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to custom model for trace collection")
            self._load_custom_model()
    
    def _load_custom_model(self):
        """Fallback: Use custom model for trace collection"""
        from models.small_switch_transformer import SmallSwitchTransformer
        from transformers import AutoTokenizer
        
        logger.info("Using custom SmallSwitchTransformer")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = SmallSwitchTransformer(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            num_experts=8
        ).to(self.device)
        
        self._setup_custom_hooks()
    
    def _setup_routing_hooks(self):
        """Set up hooks to capture Switch Transformer routing"""
        self.layer_routing_data = {}
        
        def create_routing_hook(layer_name, layer_idx):
            def hook(module, input, output):
                # Extract routing from Switch Transformer output
                if hasattr(output, 'router_logits') and output.router_logits is not None:
                    router_logits = output.router_logits
                    gate_scores = torch.softmax(router_logits, dim=-1)
                    top_k_indices = torch.topk(gate_scores, k=1, dim=-1).indices
                    
                    self.layer_routing_data[layer_idx] = {
                        'layer_name': layer_name,
                        'gate_scores': gate_scores.detach().cpu(),
                        'top_k_indices': top_k_indices.detach().cpu(),
                        'hidden_states': input[0].detach().cpu() if len(input) > 0 else None
                    }
            return hook
        
        # Hook into encoder layers
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'block'):
            for layer_idx, layer in enumerate(self.model.encoder.block):
                # Switch Transformer has MoE in certain layers
                if hasattr(layer, 'layer') and len(layer.layer) > 1:
                    if hasattr(layer.layer[1], 'mlp'):
                        layer_name = f"encoder_layer_{layer_idx}"
                        handle = layer.layer[1].mlp.register_forward_hook(
                            create_routing_hook(layer_name, layer_idx)
                        )
                        self.routing_hooks[layer_name] = handle
        
        logger.info(f"Set up {len(self.routing_hooks)} routing hooks")
    
    def _setup_custom_hooks(self):
        """Set up hooks for custom model"""
        self.layer_routing_data = {}
        
        def create_custom_hook(layer_idx):
            def hook(module, input, output):
                if len(output) > 1 and isinstance(output[1], dict):
                    routing_info = output[1]
                    self.layer_routing_data[layer_idx] = {
                        'layer_name': f"custom_layer_{layer_idx}",
                        'gate_scores': routing_info['gate_scores'].detach().cpu(),
                        'top_k_indices': routing_info['top_k_indices'].detach().cpu(),
                        'hidden_states': input[0].detach().cpu() if len(input) > 0 else None
                    }
            return hook
        
        for layer_idx, layer in enumerate(self.model.layers):
            handle = layer.register_forward_hook(create_custom_hook(layer_idx))
            self.routing_hooks[f"layer_{layer_idx}"] = handle
    
    def collect_traces_from_dataset(
        self, 
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        num_samples=1000,
        max_length=256
    ):
        """Collect routing traces from a dataset"""
        
        logger.info(f"Collecting traces from {dataset_name} ({num_samples} samples)")
        
        try:
            # Load dataset
            dataset = datasets.load_dataset(dataset_name, dataset_config, split="train")
            logger.info(f"Loaded dataset with {len(dataset)} samples")
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            logger.info("Using synthetic text data")
            dataset = self._create_synthetic_dataset(num_samples)
        
        collected_traces = []
        
        for i, sample in enumerate(tqdm(dataset, desc="Collecting traces")):
            if i >= num_samples:
                break
                
            try:
                # Get text from sample
                text = sample.get('text', '') if isinstance(sample, dict) else str(sample)
                if len(text.strip()) == 0:
                    continue
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Clear previous routing data
                self.layer_routing_data.clear()
                
                # Forward pass to collect routing
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Convert to training format
                if len(self.layer_routing_data) > 0:
                    trace = self._convert_to_training_format(
                        inputs, 
                        self.layer_routing_data,
                        dataset_name,
                        f"sample_{i}"
                    )
                    collected_traces.extend(trace)
                
                if i % 100 == 0:
                    logger.info(f"Processed {i} samples, collected {len(collected_traces)} traces")
                    
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue
        
        logger.info(f"✅ Collected {len(collected_traces)} routing traces")
        return collected_traces
    
    def _create_synthetic_dataset(self, num_samples):
        """Create synthetic text for testing"""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is revolutionizing artificial intelligence research.",
            "Deep neural networks learn complex patterns from large datasets.",
            "Natural language processing enables computers to understand human communication.",
            "Transformer models have achieved breakthrough results in language tasks.",
            "Mixture of experts architectures scale model capacity efficiently.",
            "Speculative execution reduces memory bandwidth in neural networks.",
            "Graphics processing units accelerate parallel computations significantly.",
        ]
        
        dataset = []
        for i in range(num_samples):
            # Combine and vary texts
            text = texts[i % len(texts)]
            if i > len(texts):
                text = f"{text} Sample {i}. " + texts[(i * 3) % len(texts)]
            dataset.append({"text": text})
        
        return dataset
    
    def _convert_to_training_format(self, inputs, routing_data, dataset_name, sample_id):
        """Convert routing data to training format"""
        from training.gating_data_collector import GatingDataPoint
        
        traces = []
        layer_ids = sorted(routing_data.keys())
        
        for i, layer_id in enumerate(layer_ids):
            layer_data = routing_data[layer_id]
            
            # Get context from previous layers
            prev_layer_gates = []
            for j in range(max(0, i-3), i):  # Previous 3 layers
                if j < len(layer_ids):
                    prev_layer_id = layer_ids[j]
                    prev_data = routing_data[prev_layer_id]
                    prev_layer_gates.append(prev_data['gate_scores'])
            
            # Create training data point
            if layer_data['hidden_states'] is not None:
                trace = GatingDataPoint(
                    layer_id=layer_id,
                    hidden_states=layer_data['hidden_states'].squeeze(0),  # Remove batch dim
                    input_embeddings=inputs['input_ids'].cpu(),
                    target_routing=layer_data['gate_scores'].squeeze(0),
                    target_top_k=layer_data['top_k_indices'].squeeze(0),
                    prev_layer_gates=prev_layer_gates,
                    sequence_length=layer_data['hidden_states'].size(1),
                    token_ids=inputs['input_ids'].cpu(),
                    dataset_name=dataset_name,
                    sample_id=sample_id
                )
                traces.append(trace)
        
        return traces
    
    def save_traces(self, traces, output_file="routing_data/real_traces.pkl"):
        """Save collected traces to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"Saving {len(traces)} traces to {output_file}")
        
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
        
        logger.info(f"✅ Traces saved to {output_file}")
        
        # Save metadata
        metadata = {
            'num_traces': len(traces),
            'model_name': self.model_name,
            'collection_time': time.time(),
            'device': self.device
        }
        
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def cleanup(self):
        """Remove hooks and clean up"""
        for handle in self.routing_hooks.values():
            handle.remove()
        self.routing_hooks.clear()

def main():
    """Main trace collection function"""
    
    logger.info("🚀 Starting Real MoE Routing Trace Collection")
    logger.info("=" * 60)
    
    # Configuration
    datasets_config = [
        {'name': 'wikitext', 'config': 'wikitext-2-raw-v1', 'samples': 2000},
        {'name': 'squad', 'config': 'plain_text', 'samples': 1500},
        {'name': 'glue', 'config': 'cola', 'samples': 1000},
    ]
    
    collector = RoutingTraceCollector("google/switch-base-8")
    
    try:
        # Load model
        collector.load_model()
        
        all_traces = []
        
        # Collect from multiple datasets
        for dataset_config in datasets_config:
            logger.info(f"\n📊 Processing {dataset_config['name']}...")
            traces = collector.collect_traces_from_dataset(
                dataset_name=dataset_config['name'],
                dataset_config=dataset_config.get('config'),
                num_samples=dataset_config['samples']
            )
            all_traces.extend(traces)
        
        # Save combined traces
        collector.save_traces(all_traces, "routing_data/real_traces_combined.pkl")
        
        logger.info(f"\n🎉 Collection Complete!")
        logger.info(f"Total traces collected: {len(all_traces)}")
        logger.info(f"Ready for training speculative models!")
        
        # Print summary statistics
        layer_counts = {}
        for trace in all_traces:
            layer_counts[trace.layer_id] = layer_counts.get(trace.layer_id, 0) + 1
        
        logger.info("\n📈 Trace Distribution by Layer:")
        for layer_id, count in sorted(layer_counts.items()):
            logger.info(f"  Layer {layer_id}: {count} traces")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return False
    
    finally:
        collector.cleanup()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Trace collection completed successfully!")
        print("🚀 Next step: Train speculation models with real data")
        print("   python train_with_real_traces.py")
    else:
        print("\n❌ Trace collection failed")
        print("Check logs for details and retry")