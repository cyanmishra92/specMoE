#!/usr/bin/env python3
"""
Collect DeepSeek-V3 MoE Traces for Expert Speculation Training
DeepSeek-V3: 671B parameters, 37B activated per token, DeepSeekMoE architecture
"""

# Fix threading issue
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from tqdm import tqdm
import pickle
from pathlib import Path
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeepSeekGatingDataPoint:
    """Represents a single DeepSeek-V3 gating data point for training"""
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

class DeepSeekV3TraceCollector:
    """Collector for DeepSeek-V3 MoE traces"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        logger.info(f"🚀 DeepSeek-V3 MoE Trace Collector")
        logger.info(f"Device: {self.device}")
        
    def load_model(self):
        """Load DeepSeek-V3 model and tokenizer"""
        try:
            logger.info("Loading DeepSeek-V3 model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-V3-Base",
                trust_remote_code=True
            )
            
            # Load model with specific configuration for MoE tracing
            self.model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/DeepSeek-V3-Base",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True,  # Use 8-bit to handle large model
                output_hidden_states=True,
                output_attentions=True
            )
            
            logger.info(f"✅ DeepSeek-V3 model loaded successfully")
            logger.info(f"Model config: {self.model.config}")
            
            # Check for MoE layers
            if hasattr(self.model.config, 'moe_intermediate_size'):
                logger.info(f"MoE intermediate size: {self.model.config.moe_intermediate_size}")
            if hasattr(self.model.config, 'num_experts'):
                logger.info(f"Number of experts: {self.model.config.num_experts}")
                
        except Exception as e:
            logger.error(f"❌ Error loading DeepSeek-V3 model: {e}")
            raise

    def get_moe_layers(self):
        """Identify MoE layers in DeepSeek-V3"""
        moe_layers = []
        
        # DeepSeek-V3 has MoE layers integrated into the transformer blocks
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                # This is a MoE layer
                moe_layers.append(i)
                
        logger.info(f"Found {len(moe_layers)} MoE layers: {moe_layers}")
        return moe_layers

    def extract_expert_routing(self, outputs, inputs):
        """Extract expert routing information from DeepSeek-V3 outputs"""
        routing_data = []
        
        # DeepSeek-V3 uses different MoE structure
        # Need to extract from the model's internal states
        hidden_states = outputs.hidden_states
        
        moe_layers = self.get_moe_layers()
        
        for layer_idx in moe_layers:
            if layer_idx < len(hidden_states):
                layer_hidden = hidden_states[layer_idx]
                
                # For DeepSeek-V3, we need to access the MoE routing information
                # This might require model modifications or hooking into forward pass
                layer_routing = {
                    'layer_id': layer_idx,
                    'hidden_states': layer_hidden,
                    'sequence_length': layer_hidden.shape[1]
                }
                
                routing_data.append(layer_routing)
                
        return routing_data

    def collect_traces_from_text(self, text: str, dataset_name: str, sample_id: str):
        """Collect traces from a single text sample"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Forward pass with output collection
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
            # Extract routing information
            routing_data = self.extract_expert_routing(outputs, inputs)
            
            # Convert to our data format
            traces = []
            for route_info in routing_data:
                trace = DeepSeekGatingDataPoint(
                    layer_id=route_info['layer_id'],
                    hidden_states=route_info['hidden_states'],
                    input_embeddings=outputs.hidden_states[0] if outputs.hidden_states else None,
                    target_routing=torch.zeros(128),  # Placeholder - need actual routing
                    target_top_k=torch.zeros(10),    # Placeholder - need actual top-k
                    prev_layer_gates=[],
                    sequence_length=route_info['sequence_length'],
                    token_ids=inputs['input_ids'],
                    dataset_name=dataset_name,
                    sample_id=sample_id
                )
                traces.append(trace)
                
            return traces
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return []

    def collect_from_datasets(self, num_samples_per_dataset=100):
        """Collect traces from multiple datasets"""
        
        # Dataset selection - same as original but optimized for DeepSeek-V3
        dataset_configs = [
            ("wikitext", "wikitext-2-raw-v1", "train"),
            ("bookcorpus", None, "train"),
            ("openwebtext", None, "train"),
            ("squad", None, "train"),
            ("cnn_dailymail", "3.0.0", "train"),
            ("xsum", None, "train"),
            ("multi_news", None, "train"),
            ("reddit_tifu", "short", "train"),
            ("writingprompts", None, "train"),
            ("imdb", None, "train")
        ]
        
        all_traces = []
        
        for dataset_name, config_name, split in dataset_configs:
            logger.info(f"\n📊 Processing {dataset_name} ({config_name or 'default'})...")
            
            try:
                # Load dataset
                if config_name:
                    dataset = datasets.load_dataset(dataset_name, config_name, split=split)
                else:
                    dataset = datasets.load_dataset(dataset_name, split=split)
                
                # Sample data
                sample_size = min(num_samples_per_dataset, len(dataset))
                indices = np.random.choice(len(dataset), sample_size, replace=False)
                
                # Process samples
                for i, idx in enumerate(tqdm(indices, desc=f"Processing {dataset_name}")):
                    sample = dataset[idx]
                    
                    # Extract text based on dataset format
                    if dataset_name == "wikitext":
                        text = sample['text']
                    elif dataset_name == "squad":
                        text = sample['context']
                    elif dataset_name == "cnn_dailymail":
                        text = sample['article']
                    elif dataset_name == "xsum":
                        text = sample['document']
                    elif dataset_name == "multi_news":
                        text = sample['document']
                    elif dataset_name == "reddit_tifu":
                        text = sample['documents']
                    elif dataset_name == "writingprompts":
                        text = sample['story']
                    elif dataset_name == "imdb":
                        text = sample['text']
                    else:
                        text = str(sample)
                    
                    if text and len(text.strip()) > 50:
                        traces = self.collect_traces_from_text(
                            text, 
                            dataset_name, 
                            f"{dataset_name}_{i}"
                        )
                        all_traces.extend(traces)
                        
                        # Memory management
                        if len(all_traces) > 5000:
                            logger.info(f"Collected {len(all_traces)} traces, continuing...")
                            
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                continue
                
        return all_traces

    def save_traces(self, traces, output_dir="routing_data"):
        """Save collected traces"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save as pickle
        pickle_path = output_dir / "deepseek_v3_traces.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(traces, f)
        
        # Save metadata
        metadata = {
            'model': 'deepseek-ai/DeepSeek-V3-Base',
            'num_traces': len(traces),
            'collection_time': time.time(),
            'moe_layers': self.get_moe_layers(),
            'total_parameters': '671B',
            'activated_parameters': '37B'
        }
        
        json_path = output_dir / "deepseek_v3_traces.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"✅ Saved {len(traces)} traces to {pickle_path}")
        logger.info(f"📊 Metadata saved to {json_path}")

def main():
    """Main collection function"""
    logger.info("🚀 Starting DeepSeek-V3 MoE Trace Collection")
    
    collector = DeepSeekV3TraceCollector()
    collector.load_model()
    
    # Collect traces
    traces = collector.collect_from_datasets(num_samples_per_dataset=200)
    
    # Save results
    collector.save_traces(traces)
    
    logger.info(f"🎉 Collection complete! Total traces: {len(traces)}")

if __name__ == "__main__":
    main()