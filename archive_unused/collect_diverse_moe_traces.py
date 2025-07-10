#!/usr/bin/env python3
"""
Collect Diverse MoE Routing Traces from REAL Switch Transformers
Use larger models with actual expert routing diversity
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

class DiverseMoETraceCollector:
    """Collect diverse traces from real Switch Transformer models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.routing_traces = []
        self.routing_hooks = {}
        
        logger.info(f"🚀 Diverse MoE Trace Collector")
        logger.info(f"Device: {self.device}")
        
    def try_load_switch_model(self, model_name):
        """Try to load a Switch Transformer model"""
        try:
            logger.info(f"Loading {model_name}...")
            
            # Try different parameter combinations
            for dtype in [torch.float16, torch.float32]:
                for device_map in [None, "auto"]:
                    try:
                        if "switch-base" in model_name:
                            model = SwitchTransformersForConditionalGeneration.from_pretrained(
                                model_name,
                                torch_dtype=dtype,
                                device_map=device_map,
                                output_router_logits=True
                            )
                        else:
                            model = SwitchTransformersForConditionalGeneration.from_pretrained(
                                model_name,
                                torch_dtype=dtype,
                                device_map=device_map
                            )
                        
                        if device_map is None:
                            model = model.to(self.device)
                        model.eval()
                        
                        logger.info(f"✅ Successfully loaded {model_name}")
                        logger.info(f"   Model device: {next(model.parameters()).device}")
                        logger.info(f"   Model dtype: {next(model.parameters()).dtype}")
                        
                        return model
                        
                    except Exception as e:
                        logger.debug(f"Failed with dtype={dtype}, device_map={device_map}: {e}")
                        continue
            
            logger.warning(f"Failed to load {model_name}")
            return None
            
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}")
            return None
    
    def load_best_available_model(self):
        """Load the best available MoE model"""
        
        # Try different Switch Transformer models in order of preference
        models_to_try = [
            "google/switch-base-8",      # 8 experts
            "google/switch-base-16",     # 16 experts  
            "google/switch-base-32",     # 32 experts
            "google/switch-large-128",   # 128 experts (if available)
            "google/switch-c-2048",      # Very large (if available)
        ]
        
        self.tokenizer = T5Tokenizer.from_pretrained("google/switch-base-8")
        
        for model_name in models_to_try:
            self.model = self.try_load_switch_model(model_name)
            if self.model is not None:
                self.model_name = model_name
                self.num_experts = self._detect_num_experts()
                logger.info(f"🎯 Using {model_name} with {self.num_experts} experts")
                self._setup_routing_hooks()
                return True
        
        # Fallback: Create synthetic diverse routing
        logger.warning("No Switch Transformer available, creating synthetic diverse MoE")
        return self._create_synthetic_diverse_model()
    
    def _detect_num_experts(self):
        """Detect number of experts in the loaded model"""
        try:
            # Look for expert layers in the model
            for name, module in self.model.named_modules():
                if 'expert' in name.lower() or 'moe' in name.lower():
                    logger.debug(f"Found MoE component: {name}")
            
            # Extract from model name
            if "switch-base-8" in self.model_name:
                return 8
            elif "switch-base-16" in self.model_name:
                return 16
            elif "switch-base-32" in self.model_name:
                return 32
            elif "switch-large-128" in self.model_name:
                return 128
            else:
                return 8  # Default
                
        except Exception as e:
            logger.warning(f"Could not detect num_experts: {e}")
            return 8
    
    def _create_synthetic_diverse_model(self):
        """Create synthetic model with diverse routing for testing"""
        from models.small_switch_transformer import SmallSwitchTransformer
        
        logger.info("Creating synthetic diverse MoE model")
        self.model = SmallSwitchTransformer(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            num_experts=16,  # More experts for diversity
            expert_capacity=4,
            top_k=2  # Route to top-2 experts
        ).to(self.device)
        
        self.model_name = "synthetic_diverse_moe"
        self.num_experts = 16
        self._setup_synthetic_hooks()
        return True
    
    def _setup_routing_hooks(self):
        """Set up hooks to capture Switch Transformer routing"""
        self.layer_routing_data = {}
        
        def create_routing_hook(layer_name, layer_idx):
            def hook(module, input, output):
                try:
                    # Handle different output formats
                    if hasattr(output, 'router_logits') and output.router_logits is not None:
                        router_logits = output.router_logits
                    elif isinstance(output, tuple) and len(output) > 1:
                        # Some models return (hidden_states, router_logits)
                        router_logits = output[1] if hasattr(output[1], 'shape') else None
                    else:
                        return
                    
                    if router_logits is not None and hasattr(router_logits, 'shape'):
                        # Ensure we have proper routing logits
                        if router_logits.numel() > 0:
                            gate_scores = torch.softmax(router_logits, dim=-1)
                            top_k_results = torch.topk(gate_scores, k=min(3, gate_scores.size(-1)), dim=-1)
                            
                            self.layer_routing_data[layer_idx] = {
                                'layer_name': layer_name,
                                'gate_scores': gate_scores.detach().cpu(),
                                'top_k_indices': top_k_results.indices.detach().cpu(),
                                'top_k_values': top_k_results.values.detach().cpu(),
                                'hidden_states': input[0].detach().cpu() if len(input) > 0 else None,
                                'num_experts': gate_scores.size(-1)
                            }
                except Exception as e:
                    logger.debug(f"Hook error in {layer_name}: {e}")
            
            return hook
        
        # Hook into MoE layers
        hook_count = 0
        for name, module in self.model.named_modules():
            # Look for MoE/expert layers
            if any(keyword in name.lower() for keyword in ['mlp', 'feed_forward', 'expert', 'moe']):
                if 'encoder' in name and 'block' in name:
                    # Extract layer index
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if 'block' in part and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                layer_name = f"encoder_layer_{layer_idx}"
                                handle = module.register_forward_hook(
                                    create_routing_hook(layer_name, layer_idx)
                                )
                                self.routing_hooks[layer_name] = handle
                                hook_count += 1
                                break
                            except ValueError:
                                continue
        
        logger.info(f"✅ Set up {hook_count} routing hooks")
    
    def _setup_synthetic_hooks(self):
        """Set up hooks for synthetic model"""
        self.layer_routing_data = {}
        
        def create_synthetic_hook(layer_idx):
            def hook(module, input, output):
                if len(output) > 1 and isinstance(output[1], dict):
                    routing_info = output[1]
                    if 'gate_scores' in routing_info:
                        self.layer_routing_data[layer_idx] = {
                            'layer_name': f"synthetic_layer_{layer_idx}",
                            'gate_scores': routing_info['gate_scores'].detach().cpu(),
                            'top_k_indices': routing_info['top_k_indices'].detach().cpu(),
                            'top_k_values': routing_info['top_k_values'].detach().cpu(),
                            'hidden_states': input[0].detach().cpu() if len(input) > 0 else None,
                            'num_experts': routing_info['gate_scores'].size(-1)
                        }
            return hook
        
        for layer_idx, layer in enumerate(self.model.layers):
            handle = layer.register_forward_hook(create_synthetic_hook(layer_idx))
            self.routing_hooks[f"layer_{layer_idx}"] = handle
        
        logger.info(f"✅ Set up {len(self.routing_hooks)} synthetic hooks")
    
    def collect_diverse_traces(self, num_samples=2000):
        """Collect diverse routing traces"""
        
        logger.info(f"🎯 Collecting {num_samples} diverse MoE traces")
        
        # Use diverse datasets to trigger different routing patterns
        diverse_tasks = [
            {
                'name': 'question_answering',
                'loader': self._load_qa_data,
                'samples': num_samples // 4
            },
            {
                'name': 'summarization', 
                'loader': self._load_summarization_data,
                'samples': num_samples // 4
            },
            {
                'name': 'translation',
                'loader': self._load_translation_data,
                'samples': num_samples // 4
            },
            {
                'name': 'diverse_text',
                'loader': self._load_diverse_text,
                'samples': num_samples // 4
            }
        ]
        
        all_traces = []
        
        for task in diverse_tasks:
            logger.info(f"\\n📋 Processing {task['name']} for diverse routing...")
            
            try:
                texts = task['loader'](task['samples'])
                traces = self._process_texts(texts, task['name'])
                all_traces.extend(traces)
                
                # Check routing diversity
                expert_usage = self._analyze_routing_diversity(traces)
                logger.info(f"   Expert usage in {task['name']}: {len(expert_usage)} experts used")
                
            except Exception as e:
                logger.warning(f"Failed to process {task['name']}: {e}")
                continue
        
        # Final diversity analysis
        total_expert_usage = self._analyze_routing_diversity(all_traces)
        logger.info(f"\\n🎉 Total traces collected: {len(all_traces)}")
        logger.info(f"📊 Expert diversity: {len(total_expert_usage)}/{self.num_experts} experts used")
        logger.info(f"   Expert distribution: {dict(list(total_expert_usage.most_common(8)))}")
        
        return all_traces
    
    def _load_qa_data(self, num_samples):
        """Load question answering data"""
        try:
            dataset = datasets.load_dataset("squad", split="train")
            texts = []
            for i, sample in enumerate(dataset[:num_samples]):
                if isinstance(sample, dict):
                    question = sample.get('question', '')
                    context = sample.get('context', '')[:300]
                    text = f"question: {question} context: {context}"
                    texts.append(text)
            return texts
        except Exception as e:
            logger.warning(f"QA data loading failed: {e}")
            return [f"question: What is example {i}? context: This is a test question." 
                   for i in range(num_samples)]
    
    def _load_summarization_data(self, num_samples):
        """Load summarization data"""
        try:
            dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")
            texts = []
            for i, sample in enumerate(dataset[:num_samples]):
                if isinstance(sample, dict):
                    article = sample.get('article', '')[:500]
                    text = f"summarize: {article}"
                    texts.append(text)
            return texts
        except Exception as e:
            logger.warning(f"Summarization data loading failed: {e}")
            return [f"summarize: This is a long article about topic {i} with many details." 
                   for i in range(num_samples)]
    
    def _load_translation_data(self, num_samples):
        """Load translation data"""
        translation_templates = [
            "translate English to French: Hello, how are you?",
            "translate English to German: Thank you very much.",
            "translate English to Spanish: Good morning everyone.",
            "translate English to Italian: Have a nice day.",
            "translate French to English: Bonjour mes amis.",
            "translate German to English: Guten Tag zusammen.",
            "translate Spanish to English: Muy buenos días.",
            "translate English to Portuguese: See you later."
        ]
        
        texts = []
        for i in range(num_samples):
            template = translation_templates[i % len(translation_templates)]
            text = template + f" (sample {i})"
            texts.append(text)
        
        return texts
    
    def _load_diverse_text(self, num_samples):
        """Load diverse text types to trigger different experts"""
        diverse_templates = [
            "analyze this data: The numbers show {data}",
            "code review: def function_{name}(): return {value}",
            "medical report: Patient shows symptoms of {condition}",
            "legal document: The contract states that {clause}",
            "scientific paper: Our research demonstrates {finding}",
            "news article: Breaking news about {event}",
            "recipe instructions: To make {dish}, first {step}",
            "financial analysis: The market trends indicate {trend}",
            "technical documentation: This API endpoint {description}",
            "creative writing: Once upon a time in {place}"
        ]
        
        texts = []
        for i in range(num_samples):
            template = diverse_templates[i % len(diverse_templates)]
            # Fill in placeholders with varied content
            text = template.format(
                data=f"pattern_{i%10}",
                name=f"func_{i%5}",
                value=f"result_{i%8}",
                condition=f"condition_{i%6}",
                clause=f"clause_{i%4}",
                finding=f"discovery_{i%7}",
                event=f"event_{i%9}",
                dish=f"recipe_{i%5}",
                step=f"step_{i%3}",
                trend=f"trend_{i%4}",
                description=f"feature_{i%6}",
                place=f"kingdom_{i%8}"
            )
            texts.append(text)
        
        return texts
    
    def _process_texts(self, texts, task_name):
        """Process texts and extract routing traces"""
        traces = []
        
        for i, text in enumerate(tqdm(texts, desc=f"Processing {task_name}")):
            try:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=256,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Clear routing data
                self.layer_routing_data.clear()
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Convert to training format
                if len(self.layer_routing_data) > 0:
                    trace_batch = self._convert_to_training_format(
                        inputs, self.layer_routing_data, task_name, f"{task_name}_{i}"
                    )
                    traces.extend(trace_batch)
                
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.debug(f"Failed to process text {i}: {e}")
                continue
        
        return traces
    
    def _convert_to_training_format(self, inputs, routing_data, dataset_name, sample_id):
        """Convert routing data to training format"""
        from training.gating_data_collector import GatingDataPoint
        
        traces = []
        layer_ids = sorted(routing_data.keys())
        
        for i, layer_id in enumerate(layer_ids):
            layer_data = routing_data[layer_id]
            
            # Get context from previous layers
            prev_layer_gates = []
            for j in range(max(0, i-3), i):
                if j < len(layer_ids):
                    prev_layer_id = layer_ids[j]
                    prev_data = routing_data[prev_layer_id]
                    prev_layer_gates.append(prev_data['gate_scores'])
            
            # Create training data point
            if layer_data['hidden_states'] is not None and layer_data['gate_scores'] is not None:
                hidden_states = layer_data['hidden_states'].squeeze(0)
                gate_scores = layer_data['gate_scores'].squeeze(0)
                top_k_indices = layer_data['top_k_indices'].squeeze(0)
                
                trace = GatingDataPoint(
                    layer_id=layer_id,
                    hidden_states=hidden_states,
                    input_embeddings=inputs['input_ids'].cpu(),
                    target_routing=gate_scores,
                    target_top_k=top_k_indices,
                    prev_layer_gates=prev_layer_gates,
                    sequence_length=hidden_states.size(0),
                    token_ids=inputs['input_ids'].cpu(),
                    dataset_name=dataset_name,
                    sample_id=sample_id
                )
                traces.append(trace)
        
        return traces
    
    def _analyze_routing_diversity(self, traces):
        """Analyze routing diversity in collected traces"""
        from collections import Counter
        
        all_experts = []
        for trace in traces:
            if hasattr(trace, 'target_routing'):
                routing = trace.target_routing
                if routing.numel() > 0:
                    experts = torch.argmax(routing, dim=-1)
                    all_experts.extend(experts.flatten().tolist())
        
        return Counter(all_experts)
    
    def save_diverse_traces(self, traces, output_file="routing_data/diverse_moe_traces.pkl"):
        """Save diverse traces"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving {len(traces)} diverse traces to {output_file}")
        
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
        expert_usage = self._analyze_routing_diversity(traces)
        metadata = {
            'total_traces': len(traces),
            'model_name': self.model_name,
            'num_experts': self.num_experts,
            'experts_used': len(expert_usage),
            'expert_distribution': dict(expert_usage),
            'collection_time': time.time(),
            'device': self.device
        }
        
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Diverse traces and metadata saved")
        return output_file
    
    def cleanup(self):
        """Cleanup hooks and memory"""
        for handle in self.routing_hooks.values():
            handle.remove()
        self.routing_hooks.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main diverse trace collection"""
    
    logger.info("🚀 DIVERSE MoE Trace Collection")
    logger.info("=" * 50)
    logger.info("Goal: Collect routing traces with expert diversity")
    
    collector = DiverseMoETraceCollector()
    
    try:
        # Load best available model
        if not collector.load_best_available_model():
            logger.error("Failed to load any MoE model")
            return False
        
        # Collect diverse traces
        start_time = time.time()
        traces = collector.collect_diverse_traces(num_samples=4000)
        collection_time = time.time() - start_time
        
        if len(traces) == 0:
            logger.error("❌ No traces collected!")
            return False
        
        # Save traces
        output_file = collector.save_diverse_traces(traces)
        
        # Final statistics
        logger.info(f"\\n🎉 DIVERSE COLLECTION COMPLETE!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Traces collected: {len(traces)}")
        logger.info(f"   Collection time: {collection_time:.1f}s")
        logger.info(f"   Model used: {collector.model_name}")
        logger.info(f"   Experts available: {collector.num_experts}")
        
        # Check diversity
        expert_usage = collector._analyze_routing_diversity(traces)
        diversity_ratio = len(expert_usage) / collector.num_experts
        logger.info(f"   Expert diversity: {len(expert_usage)}/{collector.num_experts} ({diversity_ratio:.1%})")
        
        if diversity_ratio > 0.5:
            logger.info("✅ Good routing diversity achieved!")
        else:
            logger.warning("⚠️  Limited routing diversity - may need larger/different model")
        
        logger.info(f"\\n🚀 Ready for diverse speculation training!")
        logger.info(f"   File: {output_file}")
        logger.info(f"   Next: python train_on_diverse_traces.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return False
    
    finally:
        collector.cleanup()

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n✅ Diverse MoE trace collection completed!")
        print("🎯 Ready for training on diverse routing patterns")
    else:
        print("\\n❌ Collection failed - check logs")