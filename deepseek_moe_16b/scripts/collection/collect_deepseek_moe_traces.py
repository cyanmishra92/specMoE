#!/usr/bin/env python3
"""
Collect DeepSeek-MoE-16B Traces for Expert Speculation Training
Optimized for RTX 3090 and A6000 GPUs - 2.8B active parameters
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
class DeepSeekMoEGatingDataPoint:
    """Represents a DeepSeek-MoE-16B gating data point for training"""
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

class DeepSeekMoETraceCollector:
    """Collector for DeepSeek-MoE-16B traces - RTX 3090/A6000 optimized"""
    
    def __init__(self):
        self.device, self.gpu_info = self._select_best_gpu()
        self.model = None
        self.tokenizer = None
        self.num_experts = 64  # DeepSeek-MoE uses 64 experts
        self.is_moe = False
        
        logger.info(f"🚀 DeepSeek-MoE-16B Trace Collector")
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
            
            # Select best GPU - DeepSeek-MoE-16B needs more memory than Qwen1.5-MoE
            selected_gpu = None
            for gpu in sorted_gpus:
                if gpu.memoryFree >= 20:  # 20GB needed for DeepSeek-MoE
                    selected_gpu = gpu
                    break
                elif gpu.memoryFree >= 16:  # 16GB minimum
                    selected_gpu = gpu
                    break
            
            if selected_gpu is None:
                selected_gpu = sorted_gpus[0]
                logger.warning(f"No GPU with 16GB+ free memory found, using GPU {selected_gpu.id} with {selected_gpu.memoryFree:.1f}GB free")
            
            # Set CUDA device with error handling
            try:
                device = f"cuda:{selected_gpu.id}"
                torch.cuda.set_device(selected_gpu.id)
                torch.cuda.get_device_properties(selected_gpu.id)
            except Exception as device_error:
                logger.warning(f"Failed to set CUDA device {selected_gpu.id}: {device_error}")
                device = "cuda"
                selected_gpu.id = 0
            
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
            return "cuda", {"id": 0, "name": "CUDA", "memory_total": 24, "memory_free": 24}
    
    def _get_optimal_config_for_gpu(self):
        """Get optimal configuration based on GPU memory"""
        memory_gb = self.gpu_info['memory_total']
        gpu_id = self.gpu_info.get('id', 0)
        
        if memory_gb >= 40:  # A6000 48GB or A100
            return {
                "quantization": "4bit",
                "device_map": "auto",
                "max_memory": {gpu_id: f"{int(memory_gb * 0.9)}GB"},
                "cpu_offload": False
            }
        elif memory_gb >= 24:  # RTX 3090, RTX 4090
            return {
                "quantization": "4bit",
                "device_map": "auto",
                "max_memory": {gpu_id: f"{int(memory_gb * 0.8)}GB", "cpu": "20GB"},
                "cpu_offload": True
            }
        else:  # Smaller GPUs
            return {
                "quantization": "8bit",
                "device_map": "auto",
                "max_memory": {gpu_id: f"{int(memory_gb * 0.7)}GB", "cpu": "15GB"},
                "cpu_offload": True
            }
        
    def load_model(self):
        """Load DeepSeek-MoE-16B model"""
        try:
            logger.info("Loading DeepSeek-MoE-16B model...")
            
            # Get optimal config for this GPU
            optimal_config = self._get_optimal_config_for_gpu()
            logger.info(f"Optimal config for {self.gpu_info['name']}: {optimal_config}")
            
            # DeepSeek-MoE model options
            model_options = [
                {
                    "name": "deepseek-ai/deepseek-moe-16b-base",
                    "is_moe": True,
                    **optimal_config
                },
                {
                    "name": "deepseek-ai/deepseek-moe-16b-chat",
                    "is_moe": True,
                    **optimal_config
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
                    
                    # Load tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Configure quantization
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
                    
                    # Load model
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        quantization_config=quantization_config,
                        output_hidden_states=True,
                        output_router_logits=True,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory=max_memory
                    )
                    
                    logger.info(f"✅ {model_name} loaded successfully")
                    logger.info(f"Model config: {self.model.config}")
                    logger.info(f"Number of layers: {self.model.config.num_hidden_layers}")
                    logger.info(f"Is MoE model: {is_moe}")
                    logger.info(f"Number of experts: {self.num_experts}")
                    
                    self.is_moe = is_moe
                    model_loaded = True
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if not model_loaded:
                raise RuntimeError("Failed to load any DeepSeek-MoE model. Please check model access or GPU memory.")
            
        except Exception as e:
            logger.error(f"❌ Error loading DeepSeek-MoE model: {e}")
            raise

    def extract_expert_routing(self, outputs, inputs):
        """Extract expert routing information from DeepSeek-MoE outputs"""
        routing_data = []
        
        # Handle MoE models with router logits
        if self.is_moe and hasattr(outputs, 'router_logits') and outputs.router_logits:
            router_logits = outputs.router_logits
            hidden_states = outputs.hidden_states
            
            for layer_idx, router_logit in enumerate(router_logits):
                if layer_idx < len(hidden_states):
                    # Handle different router logit shapes
                    if len(router_logit.shape) == 3:
                        batch_size, seq_len, num_experts = router_logit.shape
                    elif len(router_logit.shape) == 2:
                        batch_size, num_experts = router_logit.shape
                        seq_len = hidden_states[layer_idx].shape[1]
                        router_logit = router_logit.unsqueeze(1).expand(batch_size, seq_len, num_experts)
                    else:
                        logger.warning(f"Unexpected router logit shape: {router_logit.shape}")
                        continue
                    
                    # DeepSeek-MoE uses top-2 routing but with more fine-grained experts
                    top_k = 2
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
            logger.warning("No router logits found - this is not a proper MoE model")
            return []
        
        return routing_data

    def collect_traces_from_text_batch(self, texts: List[str], dataset_names: List[str], sample_ids: List[str], batch_size: int = 8):
        """Collect traces from multiple text samples in batches"""
        all_traces = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_dataset_names = dataset_names[i:i + batch_size]
            batch_sample_ids = sample_ids[i:i + batch_size]
            
            try:
                # Tokenize batch - shorter sequences for faster processing
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                if not self.is_moe:
                    logger.warning("Non-MoE model detected, skipping trace collection")
                    continue
                    
                with torch.no_grad():
                    start_time = time.time()
                    logger.info(f"🔥 Running model inference on {len(batch_texts)} samples...")
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        output_router_logits=True
                    )
                    inference_time = time.time() - start_time
                    logger.info(f"✅ Model inference completed in {inference_time:.2f}s ({inference_time/len(batch_texts):.2f}s per sample)")
                    
                # Extract routing information
                routing_data = self.extract_expert_routing(outputs, inputs)
                
                # Convert to our data format
                for route_info in routing_data:
                    batch_size_actual, seq_len, hidden_size = route_info['hidden_states'].shape
                    
                    for batch_idx in range(batch_size_actual):
                        if batch_idx < len(batch_sample_ids):
                            dataset_name = batch_dataset_names[batch_idx]
                            sample_id = batch_sample_ids[batch_idx]
                            
                            for seq_idx in range(seq_len):
                                trace = DeepSeekMoEGatingDataPoint(
                                    layer_id=route_info['layer_id'],
                                    hidden_states=route_info['hidden_states'][batch_idx:batch_idx+1, seq_idx, :],
                                    input_embeddings=outputs.hidden_states[0][batch_idx:batch_idx+1, seq_idx, :] if outputs.hidden_states else None,
                                    target_routing=route_info['target_routing'][batch_idx:batch_idx+1, seq_idx, :],
                                    target_top_k=route_info['top_k_indices'][batch_idx:batch_idx+1, seq_idx, :],
                                    prev_layer_gates=[],
                                    sequence_length=seq_len,
                                    token_ids=inputs['input_ids'][batch_idx, seq_idx] if seq_idx < inputs['input_ids'].shape[1] else None,
                                    dataset_name=dataset_name,
                                    sample_id=f"{sample_id}_seq_{seq_idx}"
                                )
                                all_traces.append(trace)
                                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                continue
                
        return all_traces

    def collect_from_datasets(self, target_total_traces=20000, max_traces_per_sample=150):
        """Collect traces from multiple datasets with balanced sampling"""
        
        # Dataset selection optimized for DeepSeek-MoE - clean datasets
        dataset_configs = [
            ("imdb", None, "train"),
            ("yelp_review_full", None, "train"),
            ("ag_news", None, "train"),
            ("squad", None, "train"),
            ("amazon_polarity", None, "train"),
            ("dbpedia_14", None, "train"),
        ]
        
        # Calculate target traces per dataset
        traces_per_dataset = target_total_traces // len(dataset_configs)
        logger.info(f"Target: {target_total_traces} total traces, ~{traces_per_dataset} per dataset")
        
        all_traces = []
        
        for dataset_name, config_name, split in dataset_configs:
            logger.info(f"\n📊 Processing {dataset_name} ({config_name or 'default'})...")
            dataset_traces = []
            
            try:
                # Load dataset
                if config_name:
                    dataset = datasets.load_dataset(dataset_name, config_name, split=split)
                else:
                    dataset = datasets.load_dataset(dataset_name, split=split)
                
                # Determine optimal batch size based on GPU memory
                memory_gb = self.gpu_info['memory_total']
                if memory_gb >= 40:  # A6000 48GB
                    batch_size = 12
                elif memory_gb >= 24:  # RTX 3090
                    batch_size = 8
                else:
                    batch_size = 4
                
                logger.info(f"Using batch size: {batch_size} for {self.gpu_info['name']}")
                
                # Start with reasonable samples
                dataset_size = len(dataset)
                initial_samples = min(25, dataset_size)
                indices = np.random.choice(dataset_size, initial_samples, replace=False)
                indices = [int(idx) for idx in indices]
                
                # Process samples until we have enough traces
                sample_idx = 0
                dataset_progress = tqdm(
                    total=traces_per_dataset,
                    desc=f"Collecting {dataset_name} traces",
                    unit="traces"
                )
                
                while len(dataset_traces) < traces_per_dataset and sample_idx < len(indices):
                    current_batch_indices = indices[sample_idx:sample_idx + batch_size]
                    batch_samples = []
                    
                    for idx in current_batch_indices:
                        sample = dataset[idx]
                        
                        # Extract text based on dataset format
                        try:
                            if dataset_name == "imdb":
                                text = sample['text']
                            elif dataset_name == "yelp_review_full":
                                text = sample['text']
                            elif dataset_name == "ag_news":
                                text = sample['text']
                            elif dataset_name == "squad":
                                text = sample['context']
                            elif dataset_name == "amazon_polarity":
                                text = sample['content']
                            elif dataset_name == "dbpedia_14":
                                text = sample['content']
                            else:
                                text = str(sample)
                        except Exception as e:
                            logger.warning(f"Error extracting text from {dataset_name}: {e}")
                            text = None
                        
                        if text and len(text.strip()) > 50:
                            batch_samples.append((text, dataset_name, f"{dataset_name}_{sample_idx}"))
                    
                    if batch_samples:
                        texts, names, ids = zip(*batch_samples)
                        logger.info(f"🚀 Starting GPU inference for batch of {len(batch_samples)} samples...")
                        start_batch_time = time.time()
                        traces = self.collect_traces_from_text_batch(
                            list(texts), 
                            list(names), 
                            list(ids),
                            len(batch_samples)
                        )
                        batch_time = time.time() - start_batch_time
                        
                        # Limit traces per sample
                        limited_traces = traces[:max_traces_per_sample * len(batch_samples)]
                        dataset_traces.extend(limited_traces)
                        
                        # Update progress bar
                        dataset_progress.update(len(limited_traces))
                        dataset_progress.set_postfix({
                            'batch_size': len(batch_samples),
                            'traces_collected': len(limited_traces),
                            'rate': f"{len(limited_traces)/batch_time:.1f} traces/s"
                        })
                        
                        # Memory management
                        torch.cuda.empty_cache()
                    
                    sample_idx += batch_size
                    
                    # Expand sample set if needed
                    if sample_idx >= len(indices) and len(dataset_traces) < traces_per_dataset:
                        additional_samples = min(25, dataset_size - len(indices))
                        if additional_samples > 0:
                            new_indices = np.random.choice(
                                [i for i in range(dataset_size) if i not in indices],
                                additional_samples,
                                replace=False
                            )
                            indices.extend([int(idx) for idx in new_indices])
                            logger.info(f"Expanding sample set for {dataset_name}: added {additional_samples} samples")
                
                dataset_progress.close()
                all_traces.extend(dataset_traces)
                logger.info(f"✅ {dataset_name}: collected {len(dataset_traces)} traces")
                
                # Memory management
                if len(all_traces) > 4000:
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
        pickle_path = output_dir / "deepseek_moe_16b_traces.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(traces, f)
        
        # Calculate file size
        file_size_mb = pickle_path.stat().st_size / (1024 * 1024)
        
        # Save metadata
        metadata = {
            'model': 'deepseek-ai/deepseek-moe-16b-base',
            'num_traces': len(traces),
            'collection_time': time.time(),
            'num_experts': self.num_experts,
            'routing_type': 'top-2',
            'total_parameters': '16.4B',
            'active_parameters': '2.8B',
            'file_size_mb': file_size_mb,
            'gpu_optimized': self.gpu_info['name']
        }
        
        json_path = output_dir / "deepseek_moe_16b_traces.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"✅ Saved {len(traces)} traces to {pickle_path}")
        logger.info(f"📊 File size: {file_size_mb:.1f} MB")
        logger.info(f"📄 Metadata saved to {json_path}")

def main():
    """Main collection function"""
    logger.info("🚀 Starting DeepSeek-MoE-16B Trace Collection")
    
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
    
    collector = DeepSeekMoETraceCollector()
    collector.load_model()
    
    # Collect traces
    traces = collector.collect_from_datasets(target_total_traces=20000, max_traces_per_sample=150)
    
    # Save results
    collector.save_traces(traces)
    
    logger.info(f"🎉 Collection complete! Total traces: {len(traces)}")

if __name__ == "__main__":
    main()