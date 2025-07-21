#!/usr/bin/env python3
"""
Streaming Qwen Trace Collection - Build Shards Incrementally
Never keeps large amounts of data in memory - perfect for RTX 3090
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
class QwenMoEGatingDataPoint:
    """Represents a Qwen1.5-MoE-A2.7B gating data point for training"""
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

class StreamingQwenTraceCollector:
    """Streaming collector that builds shards incrementally"""
    
    def __init__(self, target_traces: int = 5000, shard_size: int = 500):
        self.device, self.gpu_info = self._select_best_gpu()
        self.model = None
        self.tokenizer = None
        self.num_experts = None
        self.num_layers = None
        self.num_experts_per_tok = None
        
        # Streaming configuration
        self.target_traces = target_traces
        self.shard_size = shard_size
        self.current_shard_traces = []
        self.current_shard_id = 0
        self.total_collected = 0
        
        # Create shard directory
        self.shard_dir = Path("routing_data/shards")
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Streaming Qwen Trace Collector")
        logger.info(f"Target: {target_traces} traces in shards of {shard_size}")
        logger.info(f"GPU: {self.gpu_info['name']} ({self.gpu_info['memory_total']}GB)")
        
    def _select_best_gpu(self):
        """Select the best available GPU"""
        if not torch.cuda.is_available():
            return "cpu", {"name": "CPU", "memory_total": 0, "memory_free": 0}
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return "cpu", {"name": "CPU", "memory_total": 0, "memory_free": 0}
            
            # Sort by available memory
            sorted_gpus = sorted(gpus, key=lambda x: x.memoryFree, reverse=True)
            selected_gpu = sorted_gpus[0]
            
            gpu_info = {
                "id": selected_gpu.id,
                "name": selected_gpu.name,
                "memory_total": selected_gpu.memoryTotal,
                "memory_free": selected_gpu.memoryFree,
            }
            
            device = f"cuda:{selected_gpu.id}"
            torch.cuda.set_device(selected_gpu.id)
            
            return device, gpu_info
            
        except Exception as e:
            logger.warning(f"GPU selection failed: {e}, using default")
            return "cuda", {"name": "CUDA", "memory_total": 24, "memory_free": 24}
    
    def _setup_model(self):
        """Load model with optimal configuration"""
        model_name = "Qwen/Qwen1.5-MoE-A2.7B"
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # RTX 3090 optimized quantization
        logger.info("Loading model with 4-bit quantization for RTX 3090...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Get model config info
        if hasattr(self.model.config, 'num_experts'):
            self.num_experts = self.model.config.num_experts
        else:
            self.num_experts = 60  # Default for this model
        
        self.num_experts_per_tok = getattr(self.model.config, 'num_experts_per_tok', 4)
        self.num_layers = len(self.model.model.layers)
        
        logger.info(f"Model loaded: {self.num_layers} layers, {self.num_experts} experts, top-{self.num_experts_per_tok}")
        
        # Add hooks for streaming trace collection
        self._add_hooks()
    
    def _add_hooks(self):
        """Add hooks that immediately process and shard traces"""
        def hook_fn(module, input, output):
            if self.total_collected >= self.target_traces:
                return
                
            try:
                hidden_states = input[0]
                
                if hasattr(module, 'gate'):
                    router_logits = module.gate(hidden_states)
                    
                    if router_logits.dim() == 2:
                        router_logits = router_logits.unsqueeze(1)
                    
                    # Get top-k routing
                    k = self.num_experts_per_tok
                    top_k_logits, top_k_indices = torch.topk(router_logits, k=k, dim=-1)
                    top_k_indices = torch.clamp(top_k_indices, 0, self.num_experts - 1)
                    
                    # Create trace and immediately move to CPU
                    trace = QwenMoEGatingDataPoint(
                        layer_id=len(self.current_shard_traces),
                        hidden_states=hidden_states.detach().cpu(),
                        input_embeddings=hidden_states.detach().cpu(), 
                        target_routing=router_logits.detach().cpu(),
                        target_top_k=top_k_indices.detach().cpu(),
                        prev_layer_gates=[],
                        sequence_length=hidden_states.shape[1],
                        token_ids=None,
                        dataset_name=getattr(self, 'current_dataset', 'unknown'),
                        sample_id=f"trace_{self.total_collected}"
                    )
                    
                    # Add to current shard
                    self.current_shard_traces.append(trace)
                    self.total_collected += 1
                    
                    # Immediately clear GPU tensors
                    del hidden_states, router_logits, top_k_logits, top_k_indices
                    torch.cuda.empty_cache()
                    
                    # Save shard when full
                    if len(self.current_shard_traces) >= self.shard_size:
                        self._save_current_shard()
                        
            except Exception as e:
                logger.warning(f"Hook failed: {e}")
        
        # Add hooks to MLP layers
        for layer in self.model.model.layers:
            if hasattr(layer, 'mlp'):
                layer.mlp.register_forward_hook(hook_fn)
    
    def _save_current_shard(self):
        """Save current shard to disk and clear from memory"""
        if not self.current_shard_traces:
            return
            
        # Save shard file
        shard_file = self.shard_dir / f"shard_{self.current_shard_id:03d}_{len(self.current_shard_traces)}_traces.pkl"
        with open(shard_file, 'wb') as f:
            pickle.dump(self.current_shard_traces, f)
        
        file_size_mb = shard_file.stat().st_size / (1024 * 1024)
        
        # Save metadata
        metadata = {
            'shard_id': self.current_shard_id,
            'num_traces': len(self.current_shard_traces),
            'file_size_mb': file_size_mb,
            'rtx3090_optimized': True,
            'created_streaming': True
        }
        
        metadata_file = self.shard_dir / f"shard_{self.current_shard_id:03d}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Shard {self.current_shard_id} saved: {len(self.current_shard_traces)} traces ({file_size_mb:.1f}MB)")
        
        # Clear current shard from memory
        self.current_shard_traces.clear()
        self.current_shard_id += 1
        
        # Aggressive cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    def collect_from_text_batch(self, texts: List[str], dataset_name: str):
        """Process a batch of texts"""
        if self.total_collected >= self.target_traces:
            return
            
        self.current_dataset = dataset_name
        
        # Use shorter sequences for 60-expert model
        max_length = 256
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)
        
        # Run inference (hooks will collect traces)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Immediate cleanup
        del inputs, outputs
        torch.cuda.empty_cache()
    
    def collect_traces(self):
        """Main collection loop with streaming"""
        self._setup_model()
        
        # Dataset configuration
        datasets_config = [
            ("imdb", "train"),
            ("yelp_review_full", "train"), 
            ("ag_news", "train"),
        ]
        
        batch_size = 8  # Conservative for 60-expert model
        traces_per_dataset = self.target_traces // len(datasets_config)
        
        for dataset_name, split in datasets_config:
            if self.total_collected >= self.target_traces:
                break
                
            logger.info(f"📊 Processing {dataset_name}...")
            
            try:
                dataset = datasets.load_dataset(dataset_name, split=split)
                dataset_traces = 0
                
                with tqdm(total=min(traces_per_dataset, len(dataset)), desc=f"Collecting {dataset_name}") as pbar:
                    batch_texts = []
                    
                    for i, sample in enumerate(dataset):
                        if self.total_collected >= self.target_traces or dataset_traces >= traces_per_dataset:
                            break
                        
                        # Extract text
                        if dataset_name == "imdb":
                            text = sample['text']
                        elif dataset_name == "yelp_review_full":
                            text = sample['text']
                        elif dataset_name == "ag_news":
                            text = sample['text']
                        else:
                            text = str(sample)
                        
                        if text and len(text.strip()) > 50:
                            batch_texts.append(text)
                        
                        # Process batch
                        if len(batch_texts) >= batch_size:
                            traces_before = self.total_collected
                            self.collect_from_text_batch(batch_texts, dataset_name)
                            new_traces = self.total_collected - traces_before
                            dataset_traces += new_traces
                            pbar.update(new_traces)
                            batch_texts.clear()
                    
                    # Process remaining
                    if batch_texts and self.total_collected < self.target_traces:
                        traces_before = self.total_collected
                        self.collect_from_text_batch(batch_texts, dataset_name)
                        new_traces = self.total_collected - traces_before
                        dataset_traces += new_traces
                        pbar.update(new_traces)
                
                logger.info(f"✅ {dataset_name}: {dataset_traces} traces")
                
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                continue
        
        # Save final partial shard
        if self.current_shard_traces:
            self._save_current_shard()
        
        # Create training config
        self._create_training_config()
        
        logger.info(f"🎉 Streaming collection complete! {self.total_collected} traces in {self.current_shard_id} shards")
    
    def _create_training_config(self):
        """Create RTX 3090 training configuration"""
        config = {
            'total_traces': self.total_collected,
            'num_shards': self.current_shard_id,
            'shard_size': self.shard_size,
            'rtx3090_optimized': True,
            'model_info': {
                'num_experts': self.num_experts,
                'experts_per_token': self.num_experts_per_tok,
                'num_layers': self.num_layers
            },
            'training_settings': {
                'batch_size': 16,
                'max_seq_length': 256,
                'fp16': True,
                'gradient_accumulation': 4,
                'shard_based_training': True
            }
        }
        
        config_file = self.shard_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"📋 Training config saved: {config_file}")

def main():
    """Main streaming collection"""
    import argparse
    parser = argparse.ArgumentParser(description='Streaming Qwen trace collection')
    parser.add_argument('--traces', type=int, default=5000, help='Total traces to collect')
    parser.add_argument('--shard-size', type=int, default=500, help='Traces per shard')
    args = parser.parse_args()
    
    logger.info("🚀 Starting streaming trace collection...")
    logger.info(f"Target: {args.traces} traces in shards of {args.shard_size}")
    
    collector = StreamingQwenTraceCollector(target_traces=args.traces, shard_size=args.shard_size)
    collector.collect_traces()
    
    logger.info("✅ Collection complete! Ready for RTX 3090 training.")

if __name__ == "__main__":
    main()