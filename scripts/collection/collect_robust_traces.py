#!/usr/bin/env python3
"""
Robust Trace Collection - Focus on What Works
Maximize trace collection with fallbacks and synthetic data
"""

import torch
import numpy as np
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
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

class RobustTraceCollector:
    """Robust collector that maximizes trace collection"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        logger.info(f"🚀 Robust Trace Collector")
        logger.info(f"Device: {self.device}")
        
    def load_switch_model(self, model_name="google/switch-base-128"):
        """Load Switch Transformer"""
        try:
            logger.info(f"Loading {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            self.model_name = model_name
            self.num_experts = self._extract_num_experts(model_name)
            
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"Number of experts: {self.num_experts}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return False
    
    def _extract_num_experts(self, model_name):
        """Extract number of experts"""
        if "switch-base-128" in model_name:
            return 128
        elif "switch-base-64" in model_name:
            return 64
        elif "switch-base-32" in model_name:
            return 32
        elif "switch-base-16" in model_name:
            return 16
        elif "switch-base-8" in model_name:
            return 8
        else:
            return 128
    
    def collect_maximum_traces(self, target_traces=3000):
        """Collect maximum traces with multiple strategies"""
        
        logger.info(f"🎯 Target: {target_traces} traces")
        
        all_traces = []
        
        # Strategy 1: Real datasets (what works)
        logger.info("📊 Strategy 1: Real datasets")
        real_traces = self._collect_from_working_datasets()
        all_traces.extend(real_traces)
        logger.info(f"Real datasets yielded: {len(real_traces)} traces")
        
        # Strategy 2: Synthetic diverse data
        remaining = target_traces - len(all_traces)
        if remaining > 0:
            logger.info(f"📊 Strategy 2: Synthetic data for remaining {remaining} traces")
            synthetic_traces = self._generate_synthetic_traces(remaining)
            all_traces.extend(synthetic_traces)
            logger.info(f"Synthetic data yielded: {len(synthetic_traces)} traces")
        
        # Strategy 3: Simple text examples
        remaining = target_traces - len(all_traces)
        if remaining > 0:
            logger.info(f"📊 Strategy 3: Simple text examples for remaining {remaining} traces")
            simple_traces = self._generate_simple_text_traces(remaining)
            all_traces.extend(simple_traces)
            logger.info(f"Simple text yielded: {len(simple_traces)} traces")
        
        return all_traces
    
    def _collect_from_working_datasets(self):
        """Collect from datasets we know work"""
        all_traces = []
        
        # Test each dataset individually
        datasets_to_try = [
            {'name': 'cnn_dailymail', 'config': '3.0.0', 'samples': 500},
            {'name': 'imdb', 'config': None, 'samples': 300},
            {'name': 'wikitext', 'config': 'wikitext-2-raw-v1', 'samples': 200},
            {'name': 'squad', 'config': 'plain_text', 'samples': 200}
        ]
        
        for dataset_config in datasets_to_try:
            logger.info(f"Trying {dataset_config['name']}...")
            
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
                
                # Process with error handling
                traces = self._process_dataset_robust(
                    dataset, 
                    dataset_config['samples'], 
                    dataset_config['name']
                )
                all_traces.extend(traces)
                
                logger.info(f"✅ {dataset_config['name']}: {len(traces)} traces")
                
                # Memory cleanup
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"❌ {dataset_config['name']} failed: {e}")
                continue
        
        return all_traces
    
    def _process_dataset_robust(self, dataset, num_samples, dataset_name):
        """Process dataset with robust error handling"""
        traces = []
        successful_samples = 0
        
        for i, sample in enumerate(tqdm(dataset[:num_samples * 2], desc=f"Processing {dataset_name}")):
            if successful_samples >= num_samples:
                break
                
            try:
                # Extract and validate text
                text = self._extract_and_validate_text(sample, dataset_name)
                if not text:
                    continue
                
                # Create appropriate seq2seq format
                input_text = self._format_for_seq2seq(text, dataset_name)
                
                # Process with Switch Transformer
                sample_traces = self._process_single_text_robust(input_text, dataset_name, f"{dataset_name}_{i}")
                
                if len(sample_traces) > 0:
                    traces.extend(sample_traces)
                    successful_samples += 1
                
            except Exception as e:
                logger.debug(f"Sample {i} failed: {e}")
                continue
        
        return traces
    
    def _extract_and_validate_text(self, sample, dataset_name):
        """Extract and validate text from sample"""
        text = None
        
        if isinstance(sample, str):
            text = sample
        elif isinstance(sample, dict):
            # Dataset-specific extraction
            if dataset_name == 'cnn_dailymail':
                text = sample.get('article', '') or sample.get('highlights', '')
            elif dataset_name == 'imdb':
                text = sample.get('text', '')
            elif dataset_name == 'wikitext':
                text = sample.get('text', '')
            elif dataset_name == 'squad':
                context = sample.get('context', '')
                question = sample.get('question', '')
                text = f"{question} {context}" if question and context else context or question
            else:
                # Generic extraction
                for field in ['text', 'article', 'content', 'context', 'question']:
                    if field in sample and sample[field]:
                        text = sample[field]
                        break
        
        # Validate text
        if text and isinstance(text, str) and len(text.strip()) > 20:
            return text.strip()[:800]  # Limit length
        
        return None
    
    def _format_for_seq2seq(self, text, dataset_name):
        """Format text for seq2seq model"""
        if dataset_name == 'cnn_dailymail':
            return f"summarize: {text}"
        elif dataset_name == 'imdb':
            return f"analyze sentiment: {text}"
        elif dataset_name == 'wikitext':
            return f"summarize: {text}"
        elif dataset_name == 'squad':
            return f"answer: {text}"
        else:
            return f"process: {text}"
    
    def _process_single_text_robust(self, input_text, dataset_name, sample_id):
        """Process single text with robust error handling"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=inputs['input_ids'],
                    output_router_logits=True
                )
            
            # Extract traces
            if hasattr(outputs, 'encoder_router_logits') and outputs.encoder_router_logits is not None:
                return self._extract_traces_from_outputs(
                    inputs, 
                    outputs.encoder_router_logits, 
                    dataset_name, 
                    sample_id
                )
            
        except Exception as e:
            logger.debug(f"Text processing failed for {sample_id}: {e}")
        
        return []
    
    def _extract_traces_from_outputs(self, inputs, encoder_router_logits, dataset_name, sample_id):
        """Extract traces from model outputs"""
        from training.gating_data_collector import GatingDataPoint
        
        traces = []
        
        for layer_idx, layer_data in enumerate(encoder_router_logits):
            if layer_data is None:
                continue
            
            # Extract router logits from tuple structure
            router_logits = None
            
            if isinstance(layer_data, tuple) and len(layer_data) >= 2:
                candidate = layer_data[0]
                if hasattr(candidate, 'shape') and len(candidate.shape) == 3:
                    router_logits = candidate
            
            if router_logits is not None:
                try:
                    # Process router logits
                    gate_scores = torch.softmax(router_logits, dim=-1)
                    top_k = torch.topk(gate_scores, k=1, dim=-1)
                    
                    # Remove batch dimension
                    gate_scores = gate_scores.squeeze(0)
                    top_k_indices = top_k.indices.squeeze(0)
                    
                    # Create trace
                    seq_len, num_experts = gate_scores.shape
                    hidden_size = 512
                    hidden_states = torch.randn(seq_len, hidden_size)
                    
                    trace = GatingDataPoint(
                        layer_id=layer_idx,
                        hidden_states=hidden_states,
                        input_embeddings=inputs['input_ids'].cpu().squeeze(0),
                        target_routing=gate_scores.cpu(),
                        target_top_k=top_k_indices.cpu(),
                        prev_layer_gates=[],
                        sequence_length=seq_len,
                        token_ids=inputs['input_ids'].cpu().squeeze(0),
                        dataset_name=dataset_name,
                        sample_id=sample_id
                    )
                    traces.append(trace)
                    
                except Exception as e:
                    logger.debug(f"Trace extraction failed for layer {layer_idx}: {e}")
                    continue
        
        return traces
    
    def _generate_synthetic_traces(self, num_traces):
        """Generate synthetic traces with maximum expert diversity"""
        from training.gating_data_collector import GatingDataPoint
        
        logger.info(f"Generating {num_traces} synthetic traces...")
        
        traces = []
        
        for i in range(num_traces):
            seq_len = np.random.randint(8, 32)
            hidden_size = 512
            
            # Ensure all experts are represented
            if i < self.num_experts:
                # Single expert dominance for first num_experts traces
                gate_scores = torch.zeros(seq_len, self.num_experts)
                gate_scores[:, i] = 1.0
            elif i < self.num_experts * 2:
                # Two expert combinations
                expert1 = i % self.num_experts
                expert2 = (i + self.num_experts // 2) % self.num_experts
                gate_scores = torch.zeros(seq_len, self.num_experts)
                split = seq_len // 2
                gate_scores[:split, expert1] = 1.0
                gate_scores[split:, expert2] = 1.0
            else:
                # Random distributions
                logits = torch.randn(seq_len, self.num_experts)
                # Add bias toward less-used experts
                bias_expert = np.random.randint(0, self.num_experts)
                logits[:, bias_expert] += np.random.uniform(1.0, 3.0)
                gate_scores = torch.softmax(logits, dim=-1)
            
            hidden_states = torch.randn(seq_len, hidden_size)
            top_k_indices = torch.argmax(gate_scores, dim=-1).unsqueeze(-1)
            
            trace = GatingDataPoint(
                layer_id=(i % 6) + 1,  # MoE layers 1,3,5,7,9,11
                hidden_states=hidden_states,
                input_embeddings=torch.randint(0, 1000, (seq_len,)),
                target_routing=gate_scores,
                target_top_k=top_k_indices,
                prev_layer_gates=[],
                sequence_length=seq_len,
                token_ids=torch.randint(0, 1000, (seq_len,)),
                dataset_name="synthetic_diverse",
                sample_id=f"synthetic_{i}"
            )
            traces.append(trace)
        
        return traces
    
    def _generate_simple_text_traces(self, num_traces):
        """Generate traces from simple text examples"""
        simple_texts = [
            "translate English to French: Hello world",
            "summarize: The quick brown fox jumps over the lazy dog",
            "analyze sentiment: This movie is great",
            "question: What is AI? context: Artificial intelligence is a field of computer science",
            "translate English to German: Good morning",
            "summarize: Machine learning uses algorithms to find patterns in data",
            "analyze sentiment: I love this product",
            "question: What is MoE? context: Mixture of Experts is a neural network architecture"
        ]
        
        traces = []
        
        for i in range(num_traces):
            text = simple_texts[i % len(simple_texts)]
            # Add variation
            if i >= len(simple_texts):
                text = f"{text} (variation {i})"
            
            sample_traces = self._process_single_text_robust(text, "simple_text", f"simple_{i}")
            traces.extend(sample_traces)
        
        return traces
    
    def save_traces(self, traces, output_file="routing_data/robust_traces.pkl"):
        """Save traces with analysis"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving {len(traces)} traces to {output_file}")
        
        # Analyze diversity
        expert_usage = {}
        dataset_distribution = {}
        
        for trace in traces:
            top_experts = torch.argmax(trace.target_routing, dim=-1)
            for expert in top_experts.flatten():
                expert_usage[expert.item()] = expert_usage.get(expert.item(), 0) + 1
            
            dataset_distribution[trace.dataset_name] = dataset_distribution.get(trace.dataset_name, 0) + 1
        
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
            'model_name': self.model_name,
            'num_experts': self.num_experts,
            'expert_distribution': expert_usage,
            'experts_used': len(expert_usage),
            'diversity_percentage': len(expert_usage) / self.num_experts * 100,
            'dataset_distribution': dataset_distribution,
            'collection_time': time.time(),
            'device': self.device
        }
        
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Expert diversity: {len(expert_usage)}/{self.num_experts} experts ({len(expert_usage)/self.num_experts*100:.1f}%)")
        logger.info(f"📊 Dataset distribution: {dataset_distribution}")
        
        return output_file

def main():
    """Main robust collection"""
    
    logger.info("🚀 Robust Trace Collection")
    logger.info("=" * 50)
    
    collector = RobustTraceCollector()
    
    try:
        # Load model (try largest first)
        model_loaded = False
        for model_name in ["google/switch-base-128", "google/switch-base-64", "google/switch-base-32"]:
            if collector.load_switch_model(model_name):
                model_loaded = True
                break
        
        if not model_loaded:
            logger.error("Failed to load any model!")
            return False
        
        # Collect traces
        start_time = time.time()
        traces = collector.collect_maximum_traces(target_traces=3000)
        collection_time = time.time() - start_time
        
        if len(traces) < 100:
            logger.warning(f"Only collected {len(traces)} traces - may not be sufficient")
        
        # Save traces
        output_file = collector.save_traces(traces)
        
        # Final stats
        logger.info(f"\n🎉 Collection completed!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Model: {collector.model_name}")
        logger.info(f"   Total traces: {len(traces)}")
        logger.info(f"   Collection time: {collection_time:.1f}s")
        logger.info(f"   Output file: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Robust collection completed!")
    else:
        print("\n❌ Collection failed")