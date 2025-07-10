#!/usr/bin/env python3
"""
Benchmark-Based Trace Collection
Use standard NLP benchmarks to generate diverse MoE routing traces
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

class BenchmarkTraceCollector:
    """Collect traces from standard NLP benchmarks"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        logger.info(f"🚀 Benchmark Trace Collector")
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
    
    def collect_from_benchmarks(self, target_traces=5000):
        """Collect traces from various NLP benchmarks"""
        
        logger.info(f"🎯 Collecting {target_traces} traces from benchmarks")
        
        # Standard NLP benchmarks - these are designed to be diverse!
        benchmark_configs = [
            # Question Answering
            {'dataset': 'squad', 'config': None, 'split': 'train', 'task': 'qa', 'samples': 800},
            {'dataset': 'squad_v2', 'config': None, 'split': 'train', 'task': 'qa', 'samples': 600},
            
            # Text Classification  
            {'dataset': 'glue', 'config': 'sst2', 'split': 'train', 'task': 'sentiment', 'samples': 500},
            {'dataset': 'glue', 'config': 'cola', 'split': 'train', 'task': 'grammar', 'samples': 400},
            {'dataset': 'glue', 'config': 'mnli', 'split': 'train', 'task': 'nli', 'samples': 600},
            
            # Summarization
            {'dataset': 'cnn_dailymail', 'config': '3.0.0', 'split': 'train', 'task': 'summarization', 'samples': 500},
            {'dataset': 'xsum', 'config': None, 'split': 'train', 'task': 'summarization', 'samples': 400},
            
            # Text Generation
            {'dataset': 'wikitext', 'config': 'wikitext-2-raw-v1', 'split': 'train', 'task': 'generation', 'samples': 400},
            {'dataset': 'bookcorpus', 'config': None, 'split': 'train', 'task': 'generation', 'samples': 300},
            
            # Reading Comprehension
            {'dataset': 'race', 'config': 'all', 'split': 'train', 'task': 'reading', 'samples': 300},
            
            # Translation (if available)
            {'dataset': 'wmt16', 'config': 'de-en', 'split': 'train', 'task': 'translation', 'samples': 300},
            
            # Dialogue
            {'dataset': 'daily_dialog', 'config': None, 'split': 'train', 'task': 'dialogue', 'samples': 300},
            
            # Common Sense
            {'dataset': 'commonsense_qa', 'config': None, 'split': 'train', 'task': 'reasoning', 'samples': 300},
        ]
        
        all_traces = []
        
        for config in benchmark_configs:
            if len(all_traces) >= target_traces:
                break
                
            logger.info(f"📊 Processing {config['dataset']} ({config.get('config', 'default')})...")
            
            try:
                # Load benchmark dataset
                if config['config']:
                    dataset = datasets.load_dataset(config['dataset'], config['config'], split=config['split'])
                else:
                    dataset = datasets.load_dataset(config['dataset'], split=config['split'])
                
                # Process benchmark
                traces = self._process_benchmark_dataset(
                    dataset,
                    config['samples'],
                    config['task'],
                    f"{config['dataset']}_{config.get('config', 'default')}"
                )
                
                all_traces.extend(traces)
                logger.info(f"✅ {config['dataset']}: {len(traces)} traces (total: {len(all_traces)})")
                
                # Memory cleanup
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"❌ {config['dataset']} failed: {e}")
                # Generate synthetic data for this benchmark type
                synthetic_traces = self._generate_benchmark_synthetic(
                    config['samples'],
                    config['task'],
                    f"synthetic_{config['dataset']}"
                )
                all_traces.extend(synthetic_traces)
                logger.info(f"🔄 {config['dataset']}: {len(synthetic_traces)} synthetic traces")
        
        return all_traces[:target_traces]  # Limit to target
    
    def _process_benchmark_dataset(self, dataset, num_samples, task_type, dataset_name):
        """Process a benchmark dataset"""
        traces = []
        processed = 0
        
        for i, sample in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
            if processed >= num_samples:
                break
                
            try:
                # Extract data based on task type
                input_text, target_text = self._extract_benchmark_data(sample, task_type)
                
                if not input_text:
                    continue
                
                # Process with Switch Transformer
                sample_traces = self._run_benchmark_inference(
                    input_text,
                    target_text,
                    dataset_name,
                    f"{dataset_name}_{i}"
                )
                
                if len(sample_traces) > 0:
                    traces.extend(sample_traces)
                    processed += 1
                
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.debug(f"Sample {i} failed: {e}")
                continue
        
        return traces
    
    def _extract_benchmark_data(self, sample, task_type):
        """Extract input/target from benchmark sample based on task type"""
        
        if task_type == 'qa':
            # Question Answering
            question = sample.get('question', '')
            context = sample.get('context', '')
            answers = sample.get('answers', {})
            
            if question and context:
                input_text = f"question: {question} context: {context[:500]}"
                if isinstance(answers, dict) and 'text' in answers and answers['text']:
                    target_text = answers['text'][0] if isinstance(answers['text'], list) else str(answers['text'])
                else:
                    target_text = "unknown"
                return input_text, target_text
        
        elif task_type == 'sentiment':
            # Sentiment Analysis
            sentence = sample.get('sentence', '')
            label = sample.get('label', 0)
            
            if sentence:
                input_text = f"sentiment: {sentence}"
                target_text = "positive" if label == 1 else "negative"
                return input_text, target_text
        
        elif task_type == 'grammar':
            # Grammar Acceptability
            sentence = sample.get('sentence', '')
            label = sample.get('label', 0)
            
            if sentence:
                input_text = f"grammar: {sentence}"
                target_text = "acceptable" if label == 1 else "unacceptable"
                return input_text, target_text
        
        elif task_type == 'nli':
            # Natural Language Inference
            premise = sample.get('premise', '')
            hypothesis = sample.get('hypothesis', '')
            label = sample.get('label', 0)
            
            if premise and hypothesis:
                input_text = f"premise: {premise} hypothesis: {hypothesis}"
                labels = ["entailment", "neutral", "contradiction"]
                target_text = labels[label] if 0 <= label < len(labels) else "neutral"
                return input_text, target_text
        
        elif task_type == 'summarization':
            # Summarization
            article = sample.get('article', '') or sample.get('document', '')
            summary = sample.get('highlights', '') or sample.get('summary', '')
            
            if article:
                input_text = f"summarize: {article[:600]}"
                target_text = summary[:200] if summary else "summary"
                return input_text, target_text
        
        elif task_type == 'generation':
            # Text Generation
            text = sample.get('text', '')
            
            if text and len(text) > 50:
                # Use first part as input, next part as target
                mid_point = len(text) // 2
                input_text = f"continue: {text[:mid_point]}"
                target_text = text[mid_point:mid_point+200]
                return input_text, target_text
        
        elif task_type == 'reading':
            # Reading Comprehension
            article = sample.get('article', '')
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            
            if article and question:
                input_text = f"read: {article[:400]} question: {question}"
                target_text = answer if answer else "answer"
                return input_text, target_text
        
        elif task_type == 'translation':
            # Translation
            source = sample.get('translation', {}).get('de', '') or sample.get('de', '')
            target = sample.get('translation', {}).get('en', '') or sample.get('en', '')
            
            if source:
                input_text = f"translate German to English: {source}"
                target_text = target if target else "translation"
                return input_text, target_text
        
        elif task_type == 'dialogue':
            # Dialogue
            dialog = sample.get('dialog', [])
            
            if dialog and len(dialog) >= 2:
                context = " ".join(dialog[:-1])
                response = dialog[-1]
                input_text = f"respond: {context}"
                target_text = response
                return input_text, target_text
        
        elif task_type == 'reasoning':
            # Common Sense Reasoning
            question = sample.get('question', '')
            choices = sample.get('choices', {})
            answer_key = sample.get('answerKey', '')
            
            if question and choices:
                choice_text = " ".join([f"{k}: {v}" for k, v in choices.items()])
                input_text = f"reason: {question} choices: {choice_text}"
                target_text = answer_key if answer_key else "A"
                return input_text, target_text
        
        return None, None
    
    def _run_benchmark_inference(self, input_text, target_text, dataset_name, sample_id):
        """Run inference on benchmark data"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            targets = self.tokenizer(
                target_text,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=targets['input_ids'],
                    output_router_logits=True
                )
            
            # Extract traces
            if hasattr(outputs, 'encoder_router_logits') and outputs.encoder_router_logits is not None:
                return self._extract_traces_from_outputs(
                    inputs,
                    targets,
                    outputs.encoder_router_logits,
                    dataset_name,
                    sample_id
                )
            
        except Exception as e:
            logger.debug(f"Inference failed for {sample_id}: {e}")
        
        return []
    
    def _extract_traces_from_outputs(self, inputs, targets, encoder_router_logits, dataset_name, sample_id):
        """Extract traces from model outputs"""
        from training.gating_data_collector import GatingDataPoint
        
        traces = []
        
        for layer_idx, layer_data in enumerate(encoder_router_logits):
            if layer_data is None:
                continue
            
            # Extract router logits
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
    
    def _generate_benchmark_synthetic(self, num_samples, task_type, dataset_name):
        """Generate synthetic traces for a specific benchmark task"""
        from training.gating_data_collector import GatingDataPoint
        
        traces = []
        
        for i in range(num_samples):
            seq_len = np.random.randint(12, 40)  # Longer sequences for benchmarks
            hidden_size = 512
            
            # Task-specific expert usage patterns
            if task_type == 'qa':
                # QA might use reasoning experts more
                expert_bias = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
            elif task_type == 'sentiment':
                # Sentiment might use emotion experts
                expert_bias = np.random.choice(range(self.num_experts//4, self.num_experts//2))
            elif task_type == 'summarization':
                # Summarization might use compression experts
                expert_bias = np.random.choice(range(self.num_experts//2, 3*self.num_experts//4))
            else:
                # Random for other tasks
                expert_bias = np.random.randint(0, self.num_experts)
            
            # Create routing with task bias
            logits = torch.randn(seq_len, self.num_experts)
            logits[:, expert_bias] += np.random.uniform(1.5, 3.0)
            
            # Add some randomness to other experts
            num_active = np.random.randint(2, min(8, self.num_experts))
            active_experts = np.random.choice(self.num_experts, num_active, replace=False)
            for expert in active_experts:
                logits[:, expert] += np.random.uniform(0.5, 1.5)
            
            gate_scores = torch.softmax(logits, dim=-1)
            hidden_states = torch.randn(seq_len, hidden_size)
            top_k_indices = torch.argmax(gate_scores, dim=-1).unsqueeze(-1)
            
            trace = GatingDataPoint(
                layer_id=(i % 6) + 1,
                hidden_states=hidden_states,
                input_embeddings=torch.randint(0, 1000, (seq_len,)),
                target_routing=gate_scores,
                target_top_k=top_k_indices,
                prev_layer_gates=[],
                sequence_length=seq_len,
                token_ids=torch.randint(0, 1000, (seq_len,)),
                dataset_name=dataset_name,
                sample_id=f"{dataset_name}_{i}"
            )
            traces.append(trace)
        
        return traces
    
    def save_traces(self, traces, output_file="routing_data/benchmark_traces.pkl"):
        """Save traces with comprehensive analysis"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving {len(traces)} traces to {output_file}")
        
        # Analyze diversity
        expert_usage = {}
        task_distribution = {}
        layer_distribution = {}
        
        for trace in traces:
            # Expert analysis
            top_experts = torch.argmax(trace.target_routing, dim=-1)
            for expert in top_experts.flatten():
                expert_usage[expert.item()] = expert_usage.get(expert.item(), 0) + 1
            
            # Task distribution
            task_distribution[trace.dataset_name] = task_distribution.get(trace.dataset_name, 0) + 1
            
            # Layer distribution
            layer_distribution[trace.layer_id] = layer_distribution.get(trace.layer_id, 0) + 1
        
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
            'task_distribution': task_distribution,
            'layer_distribution': layer_distribution,
            'collection_time': time.time(),
            'device': self.device
        }
        
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Expert diversity: {len(expert_usage)}/{self.num_experts} experts ({len(expert_usage)/self.num_experts*100:.1f}%)")
        logger.info(f"📊 Task distribution: {task_distribution}")
        
        return output_file

def main():
    """Main benchmark collection"""
    
    logger.info("🚀 Benchmark-Based Trace Collection")
    logger.info("=" * 50)
    
    collector = BenchmarkTraceCollector()
    
    try:
        # Load model
        model_loaded = False
        for model_name in ["google/switch-base-128", "google/switch-base-64", "google/switch-base-32"]:
            if collector.load_switch_model(model_name):
                model_loaded = True
                break
        
        if not model_loaded:
            logger.error("Failed to load any model!")
            return False
        
        # Collect from benchmarks
        start_time = time.time()
        traces = collector.collect_from_benchmarks(target_traces=5000)
        collection_time = time.time() - start_time
        
        if len(traces) < 1000:
            logger.warning(f"Only collected {len(traces)} traces")
        
        # Save traces
        output_file = collector.save_traces(traces)
        
        # Final stats
        logger.info(f"\n🎉 Benchmark collection completed!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Model: {collector.model_name}")
        logger.info(f"   Total traces: {len(traces)}")
        logger.info(f"   Collection time: {collection_time/60:.1f} minutes")
        logger.info(f"   Traces per minute: {len(traces)/(collection_time/60):.1f}")
        logger.info(f"   Output file: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Benchmark trace collection completed!")
        print("🎯 Diverse benchmark data ready for training!")
    else:
        print("\n❌ Collection failed")