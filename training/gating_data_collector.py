"""
Gating Data Collector
Collects real MoE routing patterns from trained models to create training data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pickle
from tqdm import tqdm
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GatingDataPoint:
    """Single training data point for gating prediction"""
    
    # Input features
    layer_id: int
    hidden_states: torch.Tensor          # [seq_len, hidden_size]
    input_embeddings: torch.Tensor       # [seq_len, hidden_size] 
    target_routing: torch.Tensor         # [seq_len, num_experts] - gate scores
    target_top_k: torch.Tensor           # [seq_len, top_k] - selected experts
    
    # Optional features with defaults
    attention_scores: Optional[torch.Tensor] = None  # [num_heads, seq_len, seq_len]
    prev_layer_routing: Optional[List[torch.Tensor]] = None  # Previous layer routing decisions
    prev_layer_gates: Optional[List[torch.Tensor]] = None    # Previous layer gate scores
    routing_history: Optional[List[int]] = None              # Sequence of expert choices
    sequence_length: int = 0
    token_ids: Optional[torch.Tensor] = None
    dataset_name: str = ""
    sample_id: str = ""

class GatingDataCollector:
    """
    Collects routing patterns from trained MoE models to create gating training data
    """
    
    def __init__(self, model_name: str = "google/switch-base-8", max_seq_len: int = 512):
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.routing_hooks = {}
        self.collected_data = []
        
        # Hook storage
        self.layer_activations = {}
        self.layer_routing_data = {}
        
        logger.info(f"Initializing GatingDataCollector for {model_name}")
        
    def load_model(self):
        """Load the MoE model and set up routing hooks"""
        try:
            from transformers import T5Tokenizer, SwitchTransformersForConditionalGeneration
            
            logger.info(f"Loading {self.model_name}...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                output_router_logits=True  # This is key for getting routing info
            )
            
            self.model.eval()
            logger.info("✅ Model loaded successfully")
            
            # Set up routing hooks
            self._setup_routing_hooks()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to custom model
            self._load_custom_model()
    
    def _load_custom_model(self):
        """Load our custom Switch Transformer as fallback"""
        logger.info("Loading custom SmallSwitchTransformer as fallback...")
        
        from ..models.small_switch_transformer import SmallSwitchTransformer
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = SmallSwitchTransformer(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            num_experts=8
        ).to(self.device)
        
        self._setup_custom_routing_hooks()
        logger.info("✅ Custom model loaded as fallback")
    
    def _setup_routing_hooks(self):
        """Set up hooks to capture routing information from Switch Transformers"""
        
        def create_routing_hook(layer_idx):
            def hook(module, input, output):
                # Extract routing information
                if hasattr(output, 'router_logits') and output.router_logits is not None:
                    router_logits = output.router_logits
                    
                    # Store routing data
                    self.layer_routing_data[layer_idx] = {
                        'router_logits': router_logits.detach().cpu(),
                        'hidden_states': input[0].detach().cpu(),
                        'layer_id': layer_idx
                    }
                    
                    # Also store gate scores (softmax of router logits)
                    gate_scores = F.softmax(router_logits, dim=-1)
                    top_k_indices = torch.topk(gate_scores, k=1, dim=-1).indices
                    
                    self.layer_routing_data[layer_idx].update({
                        'gate_scores': gate_scores.detach().cpu(),
                        'top_k_indices': top_k_indices.detach().cpu()
                    })
            
            return hook
        
        # Register hooks on encoder and decoder layers
        for layer_idx, layer in enumerate(self.model.encoder.block):
            if hasattr(layer, 'layer') and len(layer.layer) > 1:
                # Switch Transformer has MoE in second sub-layer
                if hasattr(layer.layer[1], 'mlp'):
                    handle = layer.layer[1].mlp.register_forward_hook(create_routing_hook(f"encoder_{layer_idx}"))
                    self.routing_hooks[f"encoder_{layer_idx}"] = handle
        
        # Similar for decoder if exists
        if hasattr(self.model, 'decoder'):
            for layer_idx, layer in enumerate(self.model.decoder.block):
                if hasattr(layer, 'layer') and len(layer.layer) > 2:
                    if hasattr(layer.layer[2], 'mlp'):
                        handle = layer.layer[2].mlp.register_forward_hook(create_routing_hook(f"decoder_{layer_idx}"))
                        self.routing_hooks[f"decoder_{layer_idx}"] = handle
    
    def _setup_custom_routing_hooks(self):
        """Set up hooks for our custom model"""
        
        def create_custom_hook(layer_idx):
            def hook(module, input, output):
                # For our custom model, we need to extract routing info differently
                hidden_states = input[0]
                
                # Get routing from the MoE layer
                if hasattr(module, 'last_routing_info'):
                    routing_info = module.last_routing_info
                    
                    self.layer_routing_data[layer_idx] = {
                        'gate_scores': routing_info['gate_scores'].detach().cpu(),
                        'top_k_indices': routing_info['top_k_indices'].detach().cpu(),
                        'hidden_states': hidden_states.detach().cpu(),
                        'layer_id': layer_idx
                    }
            
            return hook
        
        # Register hooks on our custom model layers
        for layer_idx, layer in enumerate(self.model.layers):
            if hasattr(layer, 'moe_mlp'):
                handle = layer.moe_mlp.register_forward_hook(create_custom_hook(layer_idx))
                self.routing_hooks[layer_idx] = handle
    
    def collect_routing_data_from_dataset(
        self, 
        dataset_name: str = "wikitext", 
        dataset_config: str = "wikitext-2-raw-v1",
        num_samples: int = 1000,
        split: str = "train"
    ) -> List[GatingDataPoint]:
        """
        Collect routing data from a benchmark dataset
        """
        logger.info(f"Collecting routing data from {dataset_name}...")
        
        # Load dataset
        try:
            dataset = datasets.load_dataset(dataset_name, dataset_config, split=split)
            logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            # Fallback to synthetic data
            dataset = self._create_synthetic_dataset(num_samples)
        
        collected_data = []
        
        for i, sample in enumerate(tqdm(dataset, desc="Collecting routing data")):
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
                    max_length=self.max_seq_len, 
                    truncation=True, 
                    padding=True, 
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Clear previous routing data
                self.layer_routing_data.clear()
                
                # Forward pass to trigger hooks
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Convert routing data to GatingDataPoint objects
                data_points = self._convert_routing_to_datapoints(
                    inputs, 
                    dataset_name, 
                    f"sample_{i}"
                )
                
                collected_data.extend(data_points)
                
                if i % 100 == 0:
                    logger.info(f"Processed {i} samples, collected {len(collected_data)} data points")
                    
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue
        
        logger.info(f"✅ Collected {len(collected_data)} routing data points")
        return collected_data
    
    def _create_synthetic_dataset(self, num_samples: int) -> List[Dict]:
        """Create synthetic text data for testing"""
        logger.info("Creating synthetic dataset...")
        
        synthetic_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of artificial intelligence.",
            "Deep neural networks can learn complex patterns from data.",
            "Natural language processing enables computers to understand human language.",
            "Transformer models have revolutionized the field of NLP.",
        ]
        
        dataset = []
        for i in range(num_samples):
            text = synthetic_texts[i % len(synthetic_texts)]
            # Add some variation
            text = f"{text} Sample {i}. " + text
            dataset.append({"text": text})
        
        return dataset
    
    def _convert_routing_to_datapoints(
        self, 
        inputs: Dict[str, torch.Tensor], 
        dataset_name: str, 
        sample_id: str
    ) -> List[GatingDataPoint]:
        """Convert captured routing data to GatingDataPoint objects"""
        
        data_points = []
        layer_ids = sorted(self.layer_routing_data.keys())
        
        for i, layer_id in enumerate(layer_ids):
            routing_data = self.layer_routing_data[layer_id]
            
            # Get previous layer data for context
            prev_layer_routing = []
            prev_layer_gates = []
            routing_history = []
            
            for j in range(max(0, i-3), i):  # Use up to 3 previous layers
                if j < len(layer_ids):
                    prev_layer_id = layer_ids[j]
                    prev_data = self.layer_routing_data[prev_layer_id]
                    prev_layer_routing.append(prev_data['top_k_indices'])
                    prev_layer_gates.append(prev_data['gate_scores'])
                    
                    # Extract routing history
                    top_experts = prev_data['top_k_indices'].flatten()
                    routing_history.extend(top_experts.tolist())
            
            # Create data point
            data_point = GatingDataPoint(
                layer_id=layer_id,
                hidden_states=routing_data['hidden_states'].squeeze(0),  # Remove batch dim
                input_embeddings=inputs['input_ids'].cpu(),
                prev_layer_routing=prev_layer_routing,
                prev_layer_gates=prev_layer_gates,
                routing_history=routing_history,
                target_routing=routing_data['gate_scores'].squeeze(0),  # Remove batch dim
                target_top_k=routing_data['top_k_indices'].squeeze(0),  # Remove batch dim
                sequence_length=routing_data['hidden_states'].size(1),
                token_ids=inputs['input_ids'].cpu(),
                dataset_name=dataset_name,
                sample_id=sample_id
            )
            
            data_points.append(data_point)
        
        return data_points
    
    def save_collected_data(self, data: List[GatingDataPoint], output_path: str):
        """Save collected routing data to disk"""
        logger.info(f"Saving {len(data)} data points to {output_path}")
        
        # Convert to serializable format
        serializable_data = []
        for dp in data:
            data_dict = asdict(dp)
            # Convert tensors to numpy arrays
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.numpy()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    data_dict[key] = [v.numpy() for v in value]
            
            serializable_data.append(data_dict)
        
        # Save as pickle for efficient loading
        with open(output_path, 'wb') as f:
            pickle.dump(serializable_data, f)
        
        logger.info(f"✅ Data saved to {output_path}")
    
    def load_collected_data(self, input_path: str) -> List[GatingDataPoint]:
        """Load previously collected routing data"""
        logger.info(f"Loading routing data from {input_path}")
        
        with open(input_path, 'rb') as f:
            serializable_data = pickle.load(f)
        
        # Convert back to GatingDataPoint objects
        data_points = []
        for data_dict in serializable_data:
            # Convert numpy arrays back to tensors
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    data_dict[key] = torch.from_numpy(value)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    data_dict[key] = [torch.from_numpy(v) for v in value]
            
            data_points.append(GatingDataPoint(**data_dict))
        
        logger.info(f"✅ Loaded {len(data_points)} data points")
        return data_points
    
    def cleanup(self):
        """Remove hooks and clean up"""
        for handle in self.routing_hooks.values():
            handle.remove()
        self.routing_hooks.clear()
        logger.info("✅ Cleanup completed")

def collect_multiple_datasets(
    datasets_config: List[Dict],
    model_name: str = "google/switch-base-8",
    output_dir: str = "routing_data"
) -> str:
    """
    Collect routing data from multiple datasets
    
    Args:
        datasets_config: List of dicts with 'name', 'config', 'samples' keys
        model_name: MoE model to use
        output_dir: Directory to save data
    
    Returns:
        Path to combined dataset file
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    collector = GatingDataCollector(model_name)
    
    try:
        collector.load_model()
        
        all_data = []
        
        for dataset_config in datasets_config:
            logger.info(f"Processing {dataset_config['name']}...")
            
            data = collector.collect_routing_data_from_dataset(
                dataset_name=dataset_config['name'],
                dataset_config=dataset_config.get('config', None),
                num_samples=dataset_config.get('samples', 1000),
                split=dataset_config.get('split', 'train')
            )
            
            all_data.extend(data)
            
            # Save individual dataset
            individual_path = output_path / f"{dataset_config['name']}_routing_data.pkl"
            collector.save_collected_data(data, str(individual_path))
        
        # Save combined dataset
        combined_path = output_path / "combined_routing_data.pkl"
        collector.save_collected_data(all_data, str(combined_path))
        
        logger.info(f"✅ Total collected: {len(all_data)} data points")
        
        return str(combined_path)
    
    finally:
        collector.cleanup()

if __name__ == "__main__":
    # Example usage
    datasets_config = [
        {'name': 'wikitext', 'config': 'wikitext-2-raw-v1', 'samples': 500},
        {'name': 'squad', 'config': 'plain_text', 'samples': 300},
        {'name': 'glue', 'config': 'cola', 'samples': 200}
    ]
    
    try:
        combined_path = collect_multiple_datasets(
            datasets_config,
            model_name="google/switch-base-8",
            output_dir="routing_data"
        )
        
        print(f"🎉 Routing data collection completed!")
        print(f"📁 Combined dataset saved to: {combined_path}")
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        # Try with custom model
        logger.info("Retrying with custom model...")
        combined_path = collect_multiple_datasets(
            datasets_config,
            model_name="custom",
            output_dir="routing_data"
        )
        print(f"🎉 Routing data collection completed with custom model!")
        print(f"📁 Combined dataset saved to: {combined_path}")