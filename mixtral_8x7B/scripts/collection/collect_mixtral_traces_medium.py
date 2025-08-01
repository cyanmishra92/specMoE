#!/usr/bin/env python3
"""
Collect Medium Mixtral-8x7B Traces (Target < 20GB) for Training
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
import GPUtil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MixtralGatingDataPoint:
    """Represents a Mixtral-8x7B gating data point for training"""
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
    """Collector for Mixtral-8x7B traces - Medium size for training"""
    
    def __init__(self, target_traces: int = 1500):
        self.device, self.gpu_info = self._select_best_gpu()
        self.model = None
        self.tokenizer = None
        self.num_experts = None
        self.num_layers = None
        self.traces = []
        self.processed_count = 0
        self.target_traces = target_traces  # Configurable target traces
        
    def _select_best_gpu(self):
        """Select the best available GPU"""
        try:
            # Initialize CUDA if available
            if not torch.cuda.is_available():
                return "cpu", {"name": "CPU", "memory": 0}
                
            gpus = GPUtil.getGPUs()
            if not gpus:
                return "cpu", {"name": "CPU", "memory": 0}
            
            # Sort by available memory
            best_gpu = max(gpus, key=lambda g: g.memoryFree)
            gpu_info = {
                "id": best_gpu.id,
                "name": best_gpu.name,
                "memory": best_gpu.memoryTotal,
                "memory_free": best_gpu.memoryFree
            }
            
            device = f"cuda:{best_gpu.id}"
            
            # Test device accessibility
            try:
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()
                logger.info(f"Selected GPU: {best_gpu.name} with {best_gpu.memoryFree}MB free memory")
                return device, gpu_info
            except Exception as device_error:
                logger.warning(f"Device {device} not accessible: {device_error}")
                # Fallback to default CUDA device
                if torch.cuda.is_available():
                    device = "cuda:0"
                    gpu_info = {"id": 0, "name": "CUDA Device 0", "memory": 0, "memory_free": 0}
                    logger.info(f"Falling back to {device}")
                    return device, gpu_info
                else:
                    return "cpu", {"name": "CPU", "memory": 0}
            
        except Exception as e:
            logger.warning(f"GPU selection failed: {e}, using CPU")
            return "cpu", {"name": "CPU", "memory": 0}
    
    def _get_gpu_config(self):
        """Get optimized config for current GPU"""
        if self.device == "cpu":
            return {"quantization": None, "device_map": "cpu"}
        
        gpu_id = int(self.device.split(':')[1])
        memory_gb = self.gpu_info.get('memory', 0) / 1024
        
        logger.info(f"Configuring for GPU with {memory_gb:.1f}GB memory")
        
        # A100 optimized config
        if memory_gb >= 75:
            return {
                "quantization": "8bit",
                "device_map": "auto",
                "max_memory": {gpu_id: f"{int(memory_gb * 0.9)}GB"},
                "cpu_offload": False
            }
        # A6000 config  
        elif memory_gb >= 40:
            return {
                "quantization": "4bit",
                "device_map": "auto", 
                "max_memory": {gpu_id: f"{int(memory_gb * 0.8)}GB"},
                "cpu_offload": True
            }
        else:
            return {
                "quantization": "4bit",
                "device_map": "auto",
                "cpu_offload": True
            }
    
    def _get_batch_size(self):
        """Get optimal batch size for current GPU"""
        memory_gb = self.gpu_info.get('memory', 0) / 1024
        
        if memory_gb >= 75:  # A100
            return 12  # Medium batch for training data
        elif memory_gb >= 40:  # A6000
            return 6   # Good batch for A6000
        else:
            return 4   # Conservative batch
    
    def _setup_model(self):
        """Initialize model and tokenizer"""
        model_name = "mistralai/Mixtral-8x7B-v0.1"
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get GPU configuration
        gpu_config = self._get_gpu_config()
        
        # Configure quantization
        quantization_config = None
        if gpu_config["quantization"] == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif gpu_config["quantization"] == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        logger.info(f"Loading Mixtral-8x7B model with {gpu_config['quantization']} quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=gpu_config["device_map"],
            max_memory=gpu_config.get("max_memory"),
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Get model info
        self.num_experts = 8  # Mixtral-8x7B has 8 experts
        self.num_layers = len(self.model.model.layers)
        
        logger.info(f"Model loaded: {self.num_layers} layers, {self.num_experts} experts per layer")
        
        # Add hooks for trace collection
        self._add_hooks()
    
    def _add_hooks(self):
        """Add forward hooks to collect routing information"""
        def hook_fn(module, input, output):
            # Skip if we have enough traces
            if len(self.traces) >= self.target_traces:
                return
                
            try:
                # Get layer index
                layer_idx = None
                for i, layer in enumerate(self.model.model.layers):
                    if hasattr(layer, 'block_sparse_moe') and layer.block_sparse_moe is module:
                        layer_idx = i
                        break
                
                if layer_idx is None:
                    return
                
                # Extract routing information
                hidden_states = input[0]
                
                # Get router logits
                if hasattr(module, 'gate'):
                    router_logits = module.gate(hidden_states)
                    
                    # Handle different tensor shapes
                    if router_logits.dim() == 2:
                        router_logits = router_logits.unsqueeze(1)
                    
                    # Get top-k routing
                    top_k_logits, top_k_indices = torch.topk(router_logits, k=2, dim=-1)
                    
                    # Create trace
                    trace = MixtralGatingDataPoint(
                        layer_id=layer_idx,
                        hidden_states=hidden_states.detach().cpu(),
                        input_embeddings=hidden_states.detach().cpu(),
                        target_routing=router_logits.detach().cpu(),
                        target_top_k=top_k_indices.detach().cpu(),
                        prev_layer_gates=[],
                        sequence_length=hidden_states.shape[1],
                        token_ids=None,
                        dataset_name=getattr(self, 'current_dataset', 'unknown'),
                        sample_id=f"sample_{len(self.traces)}"
                    )
                    
                    self.traces.append(trace)
                    
            except Exception as e:
                logger.warning(f"Hook failed: {e}")
        
        # Add hooks to MoE layers
        for layer in self.model.model.layers:
            if hasattr(layer, 'block_sparse_moe'):
                layer.block_sparse_moe.register_forward_hook(hook_fn)
    
    def collect_traces_from_text_batch(self, texts: List[str], dataset_name: str) -> int:
        """Collect traces from a batch of text samples"""
        if len(self.traces) >= self.target_traces:
            return 0
            
        self.current_dataset = dataset_name
        collected_before = len(self.traces)
        
        logger.info(f"🚀 Starting GPU inference for batch of {len(texts)} samples...")
        
        # Tokenize batch
        start_time = time.time()
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256  # Medium sequence length
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        logger.info(f"🔥 Running model inference on {len(texts)} samples...")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        inference_time = time.time() - start_time
        logger.info(f"✅ Model inference completed in {inference_time:.2f}s ({inference_time/len(texts):.2f}s per sample)")
        
        collected_new = len(self.traces) - collected_before
        return collected_new
    
    def collect_traces(self):
        """Collect medium-sized traces from datasets"""
        self._setup_model()
        
        # Balanced dataset selection for medium traces
        datasets_config = [
            ("imdb", "imdb", {"split": "train"}),
            ("ag_news", "ag_news", {"split": "train"}),
            ("yelp_review_full", "yelp_review_full", {"split": "train"}),
        ]
        
        batch_size = self._get_batch_size()
        logger.info(f"Using batch size: {batch_size} for {self.gpu_info['name']}")
        
        traces_per_dataset = self.target_traces // len(datasets_config)
        
        for dataset_name, dataset_id, dataset_config in datasets_config:
            if len(self.traces) >= self.target_traces:
                break
                
            logger.info(f"📊 Processing {dataset_name} (medium)...")
            
            # Load dataset
            dataset = datasets.load_dataset(dataset_id, **dataset_config)
            
            # Process in batches with progress bar
            batch_texts = []
            dataset_traces = 0
            
            with tqdm(total=traces_per_dataset, desc=f"Collecting {dataset_name} traces") as pbar:
                for i, sample in enumerate(dataset):
                    if len(self.traces) >= self.target_traces or dataset_traces >= traces_per_dataset:
                        break
                        
                    # Get text
                    text = sample.get('text', sample.get('sentence', ''))
                    if text:
                        batch_texts.append(text)
                    
                    # Process batch
                    if len(batch_texts) >= batch_size:
                        collected = self.collect_traces_from_text_batch(batch_texts, dataset_name)
                        dataset_traces += collected
                        pbar.update(collected)
                        batch_texts = []
                        
                        if len(self.traces) >= self.target_traces or dataset_traces >= traces_per_dataset:
                            break
                
                # Process remaining texts
                if batch_texts and len(self.traces) < self.target_traces and dataset_traces < traces_per_dataset:
                    collected = self.collect_traces_from_text_batch(batch_texts, dataset_name)
                    dataset_traces += collected
                    pbar.update(collected)
            
            logger.info(f"✅ {dataset_name}: collected {dataset_traces} traces")
            
            # Clean memory
            if len(self.traces) > 500:
                gc.collect()
                torch.cuda.empty_cache()
        
        logger.info(f"🎉 Collection complete! Total traces: {len(self.traces)}")
        return self.traces
    
    def save_traces(self, filepath: str):
        """Save collected traces to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.traces, f)
        logger.info(f"💾 Saved {len(self.traces)} traces to {filepath}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect Medium Mixtral-8x7B Traces')
    parser.add_argument('--target_traces', type=int, default=1500, 
                        help='Number of traces to collect (default: 1500)')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Suffix to add to output filename (default: none)')
    parser.add_argument('--shard_data', action='store_true',
                        help='Automatically shard data for memory-efficient training')
    parser.add_argument('--shard_size_mb', type=int, default=500,
                        help='Target shard size in MB (default: 500)')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Mixtral-8x7B Medium Trace Collection...")
    logger.info(f"Target traces: {args.target_traces}")
    
    collector = MixtralTraceCollector(target_traces=args.target_traces)
    
    # Collect traces
    traces = collector.collect_traces()
    
    # Save traces
    output_dir = Path("routing_data")
    output_dir.mkdir(exist_ok=True)
    
    # Create output filename with optional suffix
    if args.output_suffix:
        output_file = output_dir / f"mixtral_8x7b_traces_medium_{args.output_suffix}.pkl"
    else:
        output_file = output_dir / "mixtral_8x7b_traces_medium.pkl"
    
    collector.save_traces(output_file)
    
    # Optionally shard the data
    if args.shard_data:
        logger.info("📁 Sharding data for memory-efficient training...")
        
        # Import sharding utilities
        import sys
        sys.path.append('../../qwen15_moe_a27b/scripts/utils')
        from data_sharding import shard_trace_file
        
        # Create shard directory
        shard_dir = output_file.parent / (output_file.stem + "_shards")
        shard_files = shard_trace_file(str(output_file), str(shard_dir), args.shard_size_mb)
        
        logger.info(f"📁 Created {len(shard_files)} shards in {shard_dir}")
        logger.info(f"📁 Each shard ~{args.shard_size_mb}MB, suitable for RTX 3090 training")
        
        # Optionally remove original file to save space
        # output_file.unlink()  # Uncomment if you want to remove the original
    
    logger.info(f"✅ Medium trace collection complete! {len(traces)} traces saved to {output_file}")

if __name__ == "__main__":
    main()