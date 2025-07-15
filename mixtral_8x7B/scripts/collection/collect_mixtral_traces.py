#!/usr/bin/env python3
"""
Collect Mixtral 8x7B MoE Traces for Expert Speculation Training
Optimized for RTX 3090 with 8-bit loading and efficient memory management
"""

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datasets
from tqdm import tqdm
import pickle
from pathlib import Path
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Optional
import gc
import subprocess
import GPUtil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MixtralGatingDataPoint:
    """Represents a Mixtral 8x7B gating data point for training"""
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

class MixtralTraceCollector:
    """Collector for Mixtral 8x7B MoE traces - Multi-GPU optimized"""
    
    def __init__(self):
        self.device, self.gpu_info = self._select_best_gpu()
        self.model = None
        self.tokenizer = None
        self.num_experts = 8  # Default for Mixtral, will be updated based on model
        self.is_moe = False  # Will be set during model loading
        
        logger.info(f"🚀 Mixtral 8x7B MoE Trace Collector (Multi-GPU Optimized)")
        logger.info(f"Selected GPU: {self.gpu_info['name']} ({self.gpu_info['memory_total']}GB)")
        logger.info(f"Device: {self.device}")
        
    def _select_best_gpu(self):
        """Select the best available GPU from the pool"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return "cpu", {"name": "CPU", "memory_total": 0, "memory_free": 0}
        
        try:
            # Get all GPUs
            gpus = GPUtil.getGPUs()
            if not gpus:
                logger.warning("No GPUs detected, using CPU")
                return "cpu", {"name": "CPU", "memory_total": 0, "memory_free": 0}
            
            # Sort GPUs by available memory (descending)
            sorted_gpus = sorted(gpus, key=lambda x: x.memoryFree, reverse=True)
            
            logger.info("📊 Available GPUs:")
            for i, gpu in enumerate(sorted_gpus):
                memory_used_pct = (gpu.memoryUsed / gpu.memoryTotal) * 100
                logger.info(f"  GPU {gpu.id}: {gpu.name} - {gpu.memoryFree:.1f}GB free / {gpu.memoryTotal:.1f}GB total ({memory_used_pct:.1f}% used)")
            
            # Select best GPU based on memory and model requirements
            selected_gpu = None
            for gpu in sorted_gpus:
                # Check if GPU has enough memory (minimum thresholds)
                if gpu.memoryFree >= 20:  # At least 20GB free for Mixtral
                    selected_gpu = gpu
                    break
                elif gpu.memoryFree >= 15:  # 15GB+ for Switch Transformer
                    selected_gpu = gpu
                    break
                elif gpu.memoryFree >= 10:  # 10GB+ minimum
                    selected_gpu = gpu
                    break
            
            if selected_gpu is None:
                # Fall back to GPU with most free memory
                selected_gpu = sorted_gpus[0]
                logger.warning(f"No GPU with sufficient memory found, using GPU {selected_gpu.id} with {selected_gpu.memoryFree:.1f}GB free")
            
            # Set CUDA device
            device = f"cuda:{selected_gpu.id}"
            torch.cuda.set_device(selected_gpu.id)
            # Don't set CUDA_VISIBLE_DEVICES as it interferes with device mapping
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu.id)
            
            gpu_info = {
                "id": selected_gpu.id,
                "name": selected_gpu.name,
                "memory_total": selected_gpu.memoryTotal,
                "memory_free": selected_gpu.memoryFree,
                "memory_used": selected_gpu.memoryUsed,
                "load": selected_gpu.load
            }
            
            logger.info(f"✅ Selected GPU {selected_gpu.id}: {selected_gpu.name}")
            logger.info(f"   Memory: {selected_gpu.memoryFree:.1f}GB free / {selected_gpu.memoryTotal:.1f}GB total")
            
            return device, gpu_info
            
        except Exception as e:
            logger.error(f"Error selecting GPU: {e}")
            logger.warning("Falling back to default CUDA device")
            return "cuda", {"name": "CUDA", "memory_total": 0, "memory_free": 0}
    
    def _get_optimal_config_for_gpu(self):
        """Get optimal configuration based on GPU memory"""
        memory_gb = self.gpu_info['memory_total']
        
        gpu_id = self.gpu_info['id']
        
        if memory_gb >= 80:  # A100 80GB
            return {
                "quantization": "4bit",
                "device_map": "auto",
                "max_memory": {gpu_id: f"{int(memory_gb * 0.9)}GB"},
                "cpu_offload": False
            }
        elif memory_gb >= 40:  # A100 40GB, A6000 48GB
            return {
                "quantization": "4bit",
                "device_map": "auto", 
                "max_memory": {gpu_id: f"{int(memory_gb * 0.85)}GB", "cpu": "20GB"},
                "cpu_offload": True
            }
        elif memory_gb >= 24:  # RTX 3090, RTX 4090
            return {
                "quantization": "4bit",
                "device_map": "sequential",
                "max_memory": {gpu_id: f"{int(memory_gb * 0.8)}GB", "cpu": "40GB"},
                "cpu_offload": True
            }
        else:  # Smaller GPUs
            return {
                "quantization": "8bit",
                "device_map": "auto",
                "max_memory": {gpu_id: f"{int(memory_gb * 0.7)}GB", "cpu": "20GB"},
                "cpu_offload": True
            }
        
    def load_model(self):
        """Load Mixtral 8x7B model with optimal GPU configuration"""
        try:
            logger.info("Loading Mixtral 8x7B model with optimal GPU configuration...")
            
            # Get optimal config for this GPU
            optimal_config = self._get_optimal_config_for_gpu()
            logger.info(f"Optimal config for {self.gpu_info['name']}: {optimal_config}")
            
            # Focus on real MoE models only - no synthetic traces
            model_options = [
                {
                    "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "is_moe": True,
                    **optimal_config
                },
                {
                    "name": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", 
                    "is_moe": True,
                    **optimal_config
                },
                {
                    "name": "google/switch-base-128",
                    "quantization": "8bit",  # Switch works well with 8bit
                    "is_moe": True,
                    "device_map": "auto",
                    "max_memory": {self.gpu_info['id']: f"{int(self.gpu_info['memory_total'] * 0.9)}GB"},
                    "cpu_offload": False
                }
            ]
            
            model_loaded = False
            for model_option in model_options:
                model_name = model_option["name"]
                quantization = model_option["quantization"]
                is_moe = model_option["is_moe"]
                device_map = model_option["device_map"]
                max_memory = model_option["max_memory"]
                cpu_offload = model_option.get("cpu_offload", False)
                
                try:
                    logger.info(f"Trying to load {model_name} with {quantization} quantization...")
                    logger.info(f"  Device map: {device_map}")
                    logger.info(f"  Max memory: {max_memory}")
                    logger.info(f"  CPU offload: {cpu_offload}")
                    
                    # Load tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Configure quantization based on model
                    if quantization == "4bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            llm_int8_enable_fp32_cpu_offload=cpu_offload,
                        )
                    else:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16,
                            bnb_8bit_use_double_quant=True,
                            llm_int8_enable_fp32_cpu_offload=cpu_offload,
                        )
                    
                    # Load model with optimal GPU configuration
                    model_kwargs = {
                        "torch_dtype": torch.float16,
                        "device_map": device_map,
                        "quantization_config": quantization_config,
                        "output_hidden_states": True,
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                        "max_memory": max_memory
                    }
                    
                    # Only add output_router_logits for MoE models
                    if is_moe:
                        model_kwargs["output_router_logits"] = True
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **model_kwargs
                    )
                    
                    logger.info(f"✅ {model_name} loaded successfully")
                    logger.info(f"Model config: {self.model.config}")
                    logger.info(f"Number of layers: {self.model.config.num_hidden_layers}")
                    logger.info(f"Is MoE model: {is_moe}")
                    
                    # Update expert count based on model
                    if "switch" in model_name.lower():
                        self.num_experts = 128  # Switch has 128 experts
                    elif "mixtral" in model_name.lower():
                        self.num_experts = 8   # Mixtral has 8 experts
                    
                    logger.info(f"Number of experts: {self.num_experts}")
                    self.is_moe = is_moe
                    model_loaded = True
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if not model_loaded:
                raise RuntimeError("Failed to load any Mixtral MoE model. RTX 3090 may not have sufficient memory for Mixtral 8x7B. Please try with a larger GPU or request access to gated models.")
            
        except Exception as e:
            logger.error(f"❌ Error loading Mixtral model: {e}")
            raise

    def get_moe_layers(self):
        """Get MoE layer indices for Mixtral 8x7B"""
        # In Mixtral, MoE layers are interspersed with regular layers
        # Typically every few layers has MoE
        moe_layers = []
        
        for i in range(self.model.config.num_hidden_layers):
            # Check if layer has MoE components
            layer = self.model.model.layers[i]
            if hasattr(layer, 'block_sparse_moe') or hasattr(layer, 'mlp'):
                # This is a MoE layer
                moe_layers.append(i)
                
        logger.info(f"Found {len(moe_layers)} MoE layers: {moe_layers}")
        return moe_layers

    def extract_expert_routing(self, outputs, inputs):
        """Extract expert routing information from Mixtral outputs"""
        routing_data = []
        
        # Handle MoE models with router logits
        if self.is_moe and hasattr(outputs, 'router_logits') and outputs.router_logits:
            router_logits = outputs.router_logits
            hidden_states = outputs.hidden_states
            
            for layer_idx, router_logit in enumerate(router_logits):
                if layer_idx < len(hidden_states):
                    # Handle different router logit shapes
                    if len(router_logit.shape) == 3:
                        # Expected shape: [batch_size, seq_len, num_experts]
                        batch_size, seq_len, num_experts = router_logit.shape
                    elif len(router_logit.shape) == 2:
                        # Alternative shape: [batch_size, num_experts] - expand to include seq_len
                        batch_size, num_experts = router_logit.shape
                        seq_len = hidden_states[layer_idx].shape[1]  # Get seq_len from hidden states
                        router_logit = router_logit.unsqueeze(1).expand(batch_size, seq_len, num_experts)
                    else:
                        logger.warning(f"Unexpected router logit shape: {router_logit.shape}")
                        continue
                    
                    # Handle different routing strategies
                    if "mixtral" in type(self.model).__name__.lower():
                        # Mixtral uses top-2 routing
                        top_k = 2
                        top_k_logits, top_k_indices = torch.topk(router_logit, k=top_k, dim=-1)
                    else:
                        # Switch Transformer uses top-1 routing
                        top_k = 1
                        top_k_logits, top_k_indices = torch.topk(router_logit, k=top_k, dim=-1)
                    
                    # Convert to routing probabilities
                    routing_probs = torch.softmax(router_logit, dim=-1)
                    
                    # Create target routing tensor
                    target_routing = torch.zeros(batch_size, seq_len, self.num_experts)
                    for b in range(batch_size):
                        for s in range(seq_len):
                            for k in range(top_k):
                                expert_idx = top_k_indices[b, s, k]
                                target_routing[b, s, expert_idx] = routing_probs[b, s, expert_idx]
                    
                    layer_routing = {
                        'layer_id': layer_idx,
                        'hidden_states': hidden_states[layer_idx],
                        'target_routing': target_routing,
                        'top_k_indices': top_k_indices,
                        'router_logits': router_logit,
                        'sequence_length': seq_len
                    }
                    
                    routing_data.append(layer_routing)
        
        else:
            # No synthetic routing - we only want real MoE traces
            logger.warning("No router logits found - this is not a proper MoE model")
            return []
        
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
            
            # Forward pass - only for MoE models 
            if not self.is_moe:
                logger.warning("Non-MoE model detected, skipping trace collection")
                return []
                
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    output_router_logits=True
                )
                
            # Extract routing information
            routing_data = self.extract_expert_routing(outputs, inputs)
            
            # Convert to our data format
            traces = []
            for route_info in routing_data:
                # Flatten for sequence processing
                batch_size, seq_len, hidden_size = route_info['hidden_states'].shape
                
                for seq_idx in range(seq_len):
                    trace = MixtralGatingDataPoint(
                        layer_id=route_info['layer_id'],
                        hidden_states=route_info['hidden_states'][:, seq_idx, :],
                        input_embeddings=outputs.hidden_states[0][:, seq_idx, :] if outputs.hidden_states else None,
                        target_routing=route_info['target_routing'][:, seq_idx, :],
                        target_top_k=route_info['top_k_indices'][:, seq_idx, :],
                        prev_layer_gates=[],
                        sequence_length=seq_len,
                        token_ids=inputs['input_ids'][:, seq_idx] if seq_idx < inputs['input_ids'].shape[1] else None,
                        dataset_name=dataset_name,
                        sample_id=f"{sample_id}_seq_{seq_idx}"
                    )
                    traces.append(trace)
                    
            return traces
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return []

    def collect_from_datasets(self, num_samples_per_dataset=150):
        """Collect traces from multiple datasets"""
        
        # Dataset selection optimized for Mixtral - using working datasets only
        dataset_configs = [
            ("wikitext", "wikitext-2-raw-v1", "train"),
            ("squad", None, "train"), 
            ("imdb", None, "train"),
            ("yelp_review_full", None, "train"),
            ("ag_news", None, "train"),
            ("dbpedia_14", None, "train"),
            ("amazon_polarity", None, "train"),
            ("yahoo_answers_topics", None, "train")
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
                
                # Sample data safely
                dataset_size = len(dataset)
                sample_size = min(num_samples_per_dataset, dataset_size)
                indices = np.random.choice(dataset_size, sample_size, replace=False)
                
                # Convert indices to regular Python integers
                indices = [int(idx) for idx in indices]
                
                # Process samples
                for i, idx in enumerate(tqdm(indices, desc=f"Processing {dataset_name}")):
                    sample = dataset[idx]
                    
                    # Extract text based on dataset format
                    try:
                        if dataset_name == "wikitext":
                            text = sample['text']
                        elif dataset_name == "squad":
                            text = sample['context']
                        elif dataset_name == "imdb":
                            text = sample['text']
                        elif dataset_name == "yelp_review_full":
                            text = sample['text']
                        elif dataset_name == "ag_news":
                            text = sample['text']
                        elif dataset_name == "dbpedia_14":
                            text = sample['content']
                        elif dataset_name == "amazon_polarity":
                            text = sample['content']
                        elif dataset_name == "yahoo_answers_topics":
                            text = sample['question_content']
                        else:
                            # Try common text fields
                            for field in ['text', 'content', 'article', 'document']:
                                if field in sample:
                                    text = sample[field]
                                    break
                            else:
                                text = str(sample)
                    except Exception as e:
                        logger.warning(f"Error extracting text from {dataset_name}: {e}")
                        text = None
                    
                    if text and len(text.strip()) > 50:
                        traces = self.collect_traces_from_text(
                            text, 
                            dataset_name, 
                            f"{dataset_name}_{i}"
                        )
                        all_traces.extend(traces)
                        
                        # Debug: print progress
                        if i % 10 == 0:
                            logger.info(f"Processed {i+1}/{sample_size} samples from {dataset_name}, collected {len(traces)} traces")
                    else:
                        logger.debug(f"Skipping sample {i} from {dataset_name}: text too short or empty")
                        
                        # Memory management for RTX 3090
                        if len(all_traces) > 3000:  # Lower threshold for RTX 3090
                            logger.info(f"Collected {len(all_traces)} traces, cleaning memory...")
                            torch.cuda.empty_cache()
                            gc.collect()
                            
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                continue
                
        return all_traces

    def save_traces(self, traces, output_dir="routing_data"):
        """Save collected traces"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save as pickle
        pickle_path = output_dir / "mixtral_8x7b_traces.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(traces, f)
        
        # Calculate file size
        file_size_mb = pickle_path.stat().st_size / (1024 * 1024)
        
        # Save metadata
        model_name = getattr(self.model, 'name_or_path', 'unknown_mixtral_model')
        metadata = {
            'model': model_name,
            'num_traces': len(traces),
            'collection_time': time.time(),
            'num_experts': self.num_experts,
            'routing_type': 'top-2',
            'total_parameters': '45B',
            'active_parameters': '14B',
            'file_size_mb': file_size_mb,
            'gpu_optimized': 'RTX 3090'
        }
        
        json_path = output_dir / "mixtral_8x7b_traces.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"✅ Saved {len(traces)} traces to {pickle_path}")
        logger.info(f"📊 File size: {file_size_mb:.1f} MB")
        logger.info(f"📄 Metadata saved to {json_path}")

def main():
    """Main collection function"""
    logger.info("🚀 Starting Mixtral 8x7B MoE Trace Collection")
    
    # Check HuggingFace authentication
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"🤗 Authenticated as: {user_info['name']}")
    except Exception as e:
        logger.error("❌ HuggingFace authentication failed")
        logger.error("Please run: huggingface-cli login")
        return
    
    collector = MixtralTraceCollector()
    collector.load_model()
    
    # Collect traces
    traces = collector.collect_from_datasets(num_samples_per_dataset=150)
    
    # Save results
    collector.save_traces(traces)
    
    logger.info(f"🎉 Collection complete! Total traces: {len(traces)}")

if __name__ == "__main__":
    main()