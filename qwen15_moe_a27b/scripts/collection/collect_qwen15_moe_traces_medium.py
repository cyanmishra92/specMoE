#!/usr/bin/env python3
"""
Collect Medium Qwen1.5-MoE-A2.7B Traces (Target < 20GB) for RTX 3090 Training
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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

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

class QwenMoETraceCollector:
    """Collector for Qwen1.5-MoE-A2.7B traces - Medium size for RTX 3090"""
    
    def __init__(self, target_traces: int = 2000, checkpoint_interval: int = 500, shard_size: int = 500):
        self.device, self.gpu_info = self._select_best_gpu()
        self.model = None
        self.tokenizer = None
        self.num_experts = None
        self.num_layers = None
        self.traces = []
        self.processed_count = 0
        self.total_collected = 0  # Total traces collected (including streamed)
        self.target_traces = target_traces  # Configurable target traces
        self.checkpoint_interval = checkpoint_interval  # Store checkpoint interval
        self.shard_size = shard_size  # For RTX 3090 memory-efficient training
        self.current_shard = 0
        self.memory_limit = 2000  # Keep max 2000 traces in memory to prevent OOM
        
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
        
        # RTX 3090 optimized config
        if 10 <= memory_gb <= 26:
            return {
                "quantization": "4bit",
                "device_map": "auto",
                "max_memory": {gpu_id: f"{int(memory_gb * 0.8)}GB"},
                "cpu_offload": True
            }
        # A6000 optimized config  
        elif memory_gb >= 40:
            return {
                "quantization": "8bit",
                "device_map": "auto", 
                "max_memory": {gpu_id: f"{int(memory_gb * 0.9)}GB"},
                "cpu_offload": False
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
        
        if memory_gb >= 40:  # A6000
            return 12  # Medium batch for training data
        elif memory_gb >= 20:  # RTX 3090
            return 8   # Good batch for RTX 3090
        else:
            return 4   # Conservative batch
    
    def _setup_model(self):
        """Initialize model and tokenizer"""
        model_name = "Qwen/Qwen1.5-MoE-A2.7B"
        
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
        logger.info(f"Loading Qwen1.5-MoE-A2.7B model with {gpu_config['quantization']} quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=gpu_config["device_map"],
            max_memory=gpu_config.get("max_memory"),
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Get model info - dynamically read from config
        # Get number of experts from model config
        if hasattr(self.model.config, 'num_experts'):
            self.num_experts = self.model.config.num_experts
        elif hasattr(self.model.config, 'num_experts_per_tok'):
            # Sometimes stored differently
            self.num_experts = getattr(self.model.config, 'num_experts', 8)
        else:
            self.num_experts = 8  # Default fallback
        
        # Get routing strategy
        self.num_experts_per_tok = getattr(self.model.config, 'num_experts_per_tok', 2)
        self.num_layers = len(self.model.model.layers)
        
        logger.info(f"Model loaded: {self.num_layers} layers, {self.num_experts} experts per layer")
        logger.info(f"Experts per token: {self.num_experts_per_tok}")
        
        # Add hooks for trace collection
        self._add_hooks()
    
    def _add_hooks(self):
        """Add forward hooks to collect routing information"""
        def hook_fn(module, input, output):
            # Skip if we have enough traces
            if self.total_collected >= self.target_traces:
                return
                
            try:
                # Get layer index
                layer_idx = None
                for i, layer in enumerate(self.model.model.layers):
                    if hasattr(layer, 'mlp') and layer.mlp is module:
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
                    
                    # Get top-k routing - use dynamic k from config
                    k = self.num_experts_per_tok if hasattr(self, 'num_experts_per_tok') else 2
                    top_k_logits, top_k_indices = torch.topk(router_logits, k=k, dim=-1)
                    
                    # Clamp indices to valid range to prevent routing errors
                    top_k_indices = torch.clamp(top_k_indices, 0, self.num_experts - 1)
                    
                    # Create trace
                    trace = QwenMoEGatingDataPoint(
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
                    self.total_collected += 1  # Keep accurate count including streamed
                    
                    # Memory management - prevent OOM on RTX 3090
                    if len(self.traces) >= self.memory_limit:
                        logger.info(f"🧹 Memory limit reached ({len(self.traces)} traces), saving to disk...")
                        self._emergency_save_and_clear()
                    
                    # Aggressive GPU memory cleanup every 100 traces  
                    if self.total_collected % 100 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
            except Exception as e:
                logger.warning(f"Hook failed: {e}")
        
        # Add hooks to MLP layers
        for layer in self.model.model.layers:
            if hasattr(layer, 'mlp'):
                layer.mlp.register_forward_hook(hook_fn)
    
    def collect_traces_from_text_batch(self, texts: List[str], dataset_name: str) -> int:
        """Collect traces from a batch of text samples"""
        if self.total_collected >= self.target_traces:
            return 0
            
        self.current_dataset = dataset_name
        collected_before = self.total_collected
        
        logger.info(f"🚀 Starting GPU inference for batch of {len(texts)} samples...")
        
        # Tokenize batch
        start_time = time.time()
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256 if self.num_experts >= 60 else 512  # Adjust based on model size
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        logger.info(f"🔥 Running model inference on {len(texts)} samples...")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Immediately clear outputs from GPU memory
        del outputs
        torch.cuda.empty_cache()
        
        inference_time = time.time() - start_time
        logger.info(f"✅ Model inference completed in {inference_time:.2f}s ({inference_time/len(texts):.2f}s per sample)")
        
        # Clear inputs from GPU
        del inputs
        torch.cuda.empty_cache()
        
        collected_new = self.total_collected - collected_before
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
            if self.total_collected >= self.target_traces:
                break
                
            logger.info(f"📊 Processing {dataset_name} (medium)...")
            
            # Load dataset
            dataset = datasets.load_dataset(dataset_id, **dataset_config)
            
            # Process in batches with progress bar
            batch_texts = []
            dataset_traces = 0
            
            with tqdm(total=traces_per_dataset, desc=f"Collecting {dataset_name} traces") as pbar:
                for i, sample in enumerate(dataset):
                    if self.total_collected >= self.target_traces or dataset_traces >= traces_per_dataset:
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
                        
                        # Save checkpoint based on interval
                        self.save_checkpoint("routing_data/checkpoints", self.checkpoint_interval)
                        
                        if self.total_collected >= self.target_traces or dataset_traces >= traces_per_dataset:
                            break
                
                # Process remaining texts
                if batch_texts and self.total_collected < self.target_traces and dataset_traces < traces_per_dataset:
                    collected = self.collect_traces_from_text_batch(batch_texts, dataset_name)
                    dataset_traces += collected
                    pbar.update(collected)
            
            logger.info(f"✅ {dataset_name}: collected {dataset_traces} traces")
            
            # Clean memory more aggressively
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"🎉 Collection complete! Total traces: {self.total_collected} (in memory: {len(self.traces)})")
        return self.traces
    
    def save_traces(self, filepath: str, backup: bool = True):
        """Save collected traces to file with backup and checkpointing"""
        logger.info(f"💾 Starting to save {len(self.traces)} traces to {filepath}")
        logger.info(f"💾 Estimated file size: ~{len(self.traces) * 10 / 1024:.1f} MB")
        
        # Create backup if file exists
        filepath_obj = Path(filepath)
        if backup and filepath_obj.exists():
            backup_path = filepath_obj.with_suffix(f'.backup_{int(time.time())}.pkl')
            logger.info(f"💾 Creating backup: {backup_path}")
            filepath_obj.rename(backup_path)
        
        # Load all traces from streaming files if they exist
        logger.info("💾 Collecting all traces (including streaming files)...")
        all_traces = self._load_all_streaming_traces()
        
        # Save with error handling
        temp_filepath = filepath_obj.with_suffix('.tmp')
        try:
            # Save to temporary file first
            logger.info(f"💾 Writing {len(all_traces)} traces to temporary file: {temp_filepath}")
            with open(temp_filepath, 'wb') as f:
                pickle.dump(all_traces, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Verify the temporary file
            logger.info(f"💾 Verifying temporary file...")
            with open(temp_filepath, 'rb') as f:
                test_traces = pickle.load(f)
                assert len(test_traces) == len(all_traces), f"Trace count mismatch: {len(test_traces)} vs {len(all_traces)}"
            
            # Move temp file to final location
            temp_filepath.rename(filepath_obj)
            logger.info(f"💾 Successfully moved temp file to final location")
            
        except Exception as e:
            logger.error(f"💾 Failed to save traces: {e}")
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise
        
        # Check final file
        file_size = filepath_obj.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"💾 Successfully saved {len(self.traces)} traces to {filepath}")
        logger.info(f"💾 File size: {file_size:.1f} MB")
    
    def _emergency_save_and_clear(self):
        """Emergency save current traces to disk and clear from memory (RTX 3090 OOM prevention)"""
        if not self.traces:
            return
            
        # Create streaming save directory
        streaming_dir = Path("routing_data/streaming")
        streaming_dir.mkdir(exist_ok=True)
        
        # Save current batch to disk
        batch_file = streaming_dir / f"batch_{self.processed_count // self.memory_limit:03d}.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(self.traces, f)
        
        file_size_mb = batch_file.stat().st_size / (1024 * 1024)
        logger.info(f"💾 Saved {len(self.traces)} traces to {batch_file} ({file_size_mb:.1f}MB)")
        
        # Clear traces from memory but keep count
        trace_count = len(self.traces)
        self.traces.clear()
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"🧹 Memory cleared, {trace_count} traces moved to disk")
    
    def _load_all_streaming_traces(self):
        """Load all streaming traces back into memory for final save"""
        streaming_dir = Path("routing_data/streaming")
        if not streaming_dir.exists():
            return self.traces
            
        all_traces = list(self.traces)  # Start with current traces
        
        # Load all batch files
        batch_files = sorted(streaming_dir.glob("batch_*.pkl"))
        for batch_file in batch_files:
            logger.info(f"Loading {batch_file}...")
            with open(batch_file, 'rb') as f:
                batch_traces = pickle.load(f)
                all_traces.extend(batch_traces)
        
        logger.info(f"Loaded {len(all_traces)} total traces from {len(batch_files)} batches")
        return all_traces

    def save_checkpoint(self, checkpoint_dir: str, checkpoint_interval: int = 500):
        """Save periodic checkpoints during collection"""
        if len(self.traces) % checkpoint_interval == 0 and len(self.traces) > 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{len(self.traces)}_traces.pkl"
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"💾 Saving checkpoint: {checkpoint_path}")
            try:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(self.traces, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"💾 Checkpoint saved: {len(self.traces)} traces")
            except Exception as e:
                logger.warning(f"💾 Checkpoint save failed: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect Medium Qwen1.5-MoE-A2.7B Traces')
    parser.add_argument('--target_traces', type=int, default=1000, 
                        help='Number of traces to collect (default: 1000, reduced for stability)')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Suffix to add to output filename (default: none)')
    parser.add_argument('--shard_data', action='store_true',
                        help='Automatically shard data for memory-efficient training')
    parser.add_argument('--shard_size_mb', type=int, default=200,
                        help='Target shard size in MB (default: 200 for RTX 3090)')
    parser.add_argument('--checkpoint_interval', type=int, default=250,
                        help='Save checkpoint every N traces (default: 250)')
    parser.add_argument('--resume_from_checkpoint', type=str, default='',
                        help='Resume collection from checkpoint file')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Qwen1.5-MoE-A2.7B Medium Trace Collection...")
    logger.info(f"Target traces: {args.target_traces}")
    
    collector = QwenMoETraceCollector(target_traces=args.target_traces, checkpoint_interval=args.checkpoint_interval)
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint and Path(args.resume_from_checkpoint).exists():
        logger.info(f"📥 Resuming from checkpoint: {args.resume_from_checkpoint}")
        try:
            with open(args.resume_from_checkpoint, 'rb') as f:
                collector.traces = pickle.load(f)
            logger.info(f"📥 Loaded {len(collector.traces)} traces from checkpoint")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.info("🚀 Starting fresh collection...")
    
    # Collect traces
    try:
        traces = collector.collect_traces()
        
        # Save final checkpoint
        logger.info("💾 Saving final checkpoint...")
        collector.save_checkpoint("routing_data/checkpoints", 1)  # Force save final checkpoint
        
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}")
        logger.info("💾 Saving emergency checkpoint...")
        collector.save_checkpoint("routing_data/checkpoints", 1)  # Emergency save
        raise
    
    # Save traces
    output_dir = Path("routing_data")
    output_dir.mkdir(exist_ok=True)
    
    # Create output filename with optional suffix
    if args.output_suffix:
        output_file = output_dir / f"qwen15_moe_a27b_traces_medium_{args.output_suffix}.pkl"
    else:
        output_file = output_dir / "qwen15_moe_a27b_traces_medium.pkl"
    
    collector.save_traces(output_file)
    
    # Optionally shard the data
    if args.shard_data:
        logger.info("📁 Sharding data for memory-efficient training...")
        
        # Import sharding utilities
        import sys
        sys.path.append('../utils')
        from data_sharding import shard_trace_file
        
        # Create shard directory
        shard_dir = output_file.parent / (output_file.stem + "_shards")
        shard_files = shard_trace_file(str(output_file), str(shard_dir), args.shard_size_mb)
        
        logger.info(f"📁 Created {len(shard_files)} shards in {shard_dir}")
        logger.info(f"📁 Each shard ~{args.shard_size_mb}MB, suitable for RTX 3090 training")
        
        # Optionally remove original file to save space
        # output_file.unlink()  # Uncomment if you want to remove the original
    
    logger.info(f"✅ Medium trace collection complete! {len(traces)} traces saved to {output_file}")

def create_training_shards(collector_or_traces, shard_size=500):
    """Create training shards optimized for RTX 3090"""
    shard_dir = Path("routing_data/shards")
    shard_dir.mkdir(exist_ok=True)
    
    # Handle both collector object and traces list
    if hasattr(collector_or_traces, '_load_all_streaming_traces'):
        logger.info(f"🧩 Loading all traces from collector (including streaming files)...")
        traces = collector_or_traces._load_all_streaming_traces()
    else:
        traces = collector_or_traces
    
    logger.info(f"🧩 Creating training shards for RTX 3090 (shard_size={shard_size}, total_traces={len(traces)})")
    
    for i in range(0, len(traces), shard_size):
        shard_traces = traces[i:i+shard_size]
        shard_id = i // shard_size
        
        # Save shard
        shard_file = shard_dir / f"shard_{shard_id:03d}_{len(shard_traces)}_traces.pkl"
        with open(shard_file, 'wb') as f:
            pickle.dump(shard_traces, f)
        
        file_size_mb = shard_file.stat().st_size / (1024 * 1024)
        
        # Save metadata
        metadata = {
            'shard_id': shard_id,
            'num_traces': len(shard_traces),
            'file_size_mb': file_size_mb,
            'rtx3090_optimized': True,
            'created_from': 'medium_collection'
        }
        
        metadata_file = shard_dir / f"shard_{shard_id:03d}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Shard {shard_id}: {len(shard_traces)} traces, {file_size_mb:.1f}MB")
    
    # Create training config
    total_shards = (len(traces) + shard_size - 1) // shard_size
    config = {
        'total_traces': len(traces),
        'shard_size': shard_size,
        'num_shards': total_shards,
        'rtx3090_settings': {
            'batch_size': 16,
            'max_seq_length': 256,
            'fp16': True,
            'gradient_accumulation': 4
        }
    }
    
    config_file = shard_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Created {total_shards} shards for RTX 3090 training in {shard_dir}")
    return total_shards

if __name__ == "__main__":
    main()