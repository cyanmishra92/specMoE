#!/usr/bin/env python3
"""
Switch Transformer Benchmarking Script

Evaluate fine-tuned Switch Transformer performance on research papers.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import math
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelBenchmark:
    """Benchmark Switch Transformer models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load test data
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> List[Dict]:
        """Load test dataset"""
        test_file = Path(self.config['data_dir']) / 'test' / 'finetuning_test.json'
        
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} test samples")
        return data
    
    def load_model_and_tokenizer(self, model_path: str) -> Tuple[object, object]:
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model from {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            
            # Ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.eval()
            
            # Model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded: {total_params:,} parameters")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None, None
    
    def calculate_perplexity(self, model, tokenizer, max_length: int = 512) -> float:
        """Calculate perplexity on test set"""
        logger.info("Calculating perplexity...")
        
        total_log_likelihood = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for item in tqdm(self.test_data, desc="Computing perplexity"):
                text = item['text']
                
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=False
                ).to(self.device)
                
                # Forward pass
                outputs = model(**inputs, labels=inputs['input_ids'])
                
                # Calculate log likelihood
                log_likelihood = -outputs.loss.item() * inputs['input_ids'].size(1)
                total_log_likelihood += log_likelihood
                total_tokens += inputs['input_ids'].size(1)
        
        # Calculate perplexity
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = math.exp(-avg_log_likelihood)
        
        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity
    
    def generate_samples(self, model, tokenizer, num_samples: int = 5, max_length: int = 200) -> List[Dict]:
        """Generate text samples from the model"""
        logger.info(f"Generating {num_samples} text samples...")
        
        # Use first few test samples as prompts
        prompts = [item['text'][:100] for item in self.test_data[:num_samples]]
        
        generated_samples = []
        
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                logger.info(f"Generating sample {i+1}/{num_samples}")
                
                # Tokenize prompt
                inputs = tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=100
                ).to(self.device)
                
                # Generate
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                generated_samples.append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'length': len(generated_text)
                })
        
        return generated_samples
    
    def evaluate_domain_specific_knowledge(self, model, tokenizer) -> Dict:
        """Evaluate domain-specific knowledge using test questions"""
        logger.info("Evaluating domain-specific knowledge...")
        
        # Create domain-specific prompts based on research areas
        domain_prompts = [
            "Machine learning models can be improved by",
            "In computer vision, the key challenge is",
            "Distributed systems require careful consideration of",
            "Deep learning architectures such as transformers",
            "The main advantage of mixture of experts is",
            "Edge computing enables applications to",
            "Neural networks can be optimized through",
            "Cloud computing security involves"
        ]
        
        domain_results = []
        
        with torch.no_grad():
            for prompt in domain_prompts:
                # Tokenize
                inputs = tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=50
                ).to(self.device)
                
                # Generate completion
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                domain_results.append({
                    'prompt': prompt,
                    'completion': completion
                })
        
        return domain_results
    
    def benchmark_model(self, model_path: str, model_name: str) -> Dict:
        """Comprehensive benchmark of a model"""
        logger.info(f"\\n=== Benchmarking {model_name} ===")
        
        model, tokenizer = self.load_model_and_tokenizer(model_path)
        
        if model is None:
            logger.error(f"Failed to load {model_name}")
            return {}
        
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 1. Perplexity evaluation
            perplexity = self.calculate_perplexity(model, tokenizer)
            results['perplexity'] = perplexity
            
            # 2. Text generation samples
            generated_samples = self.generate_samples(model, tokenizer)
            results['generated_samples'] = generated_samples
            
            # 3. Domain-specific knowledge
            domain_results = self.evaluate_domain_specific_knowledge(model, tokenizer)
            results['domain_knowledge'] = domain_results
            
            # 4. Model statistics
            results['model_stats'] = {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'vocabulary_size': len(tokenizer),
                'max_position_embeddings': getattr(model.config, 'max_position_embeddings', 'unknown')
            }
            
            logger.info(f"✅ {model_name} benchmark completed")
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            results['error'] = str(e)
        
        finally:
            # Clean up
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            torch.cuda.empty_cache()
        
        return results
    
    def compare_models(self, baseline_path: str, finetuned_path: str) -> Dict:
        """Compare baseline and fine-tuned models"""
        logger.info("\\n🔬 MODEL COMPARISON")
        logger.info("=" * 60)
        
        results = {
            'comparison_timestamp': datetime.now().isoformat(),
            'test_data_size': len(self.test_data)
        }
        
        # Benchmark baseline model
        baseline_results = self.benchmark_model(baseline_path, "Baseline Model")
        results['baseline'] = baseline_results
        
        # Benchmark fine-tuned model
        finetuned_results = self.benchmark_model(finetuned_path, "Fine-tuned Model")
        results['finetuned'] = finetuned_results
        
        # Calculate improvements
        if 'perplexity' in baseline_results and 'perplexity' in finetuned_results:
            baseline_ppl = baseline_results['perplexity']
            finetuned_ppl = finetuned_results['perplexity']
            
            improvement = ((baseline_ppl - finetuned_ppl) / baseline_ppl) * 100
            results['perplexity_improvement'] = improvement
            
            logger.info(f"📊 PERPLEXITY COMPARISON:")
            logger.info(f"   Baseline: {baseline_ppl:.2f}")
            logger.info(f"   Fine-tuned: {finetuned_ppl:.2f}")
            logger.info(f"   Improvement: {improvement:.1f}%")
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Save benchmark results"""
        output_path = Path(self.config['output_dir']) / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Results saved to {output_path}")
        
        # Also save human-readable summary
        summary_path = output_path.with_suffix('.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("SWITCH TRANSFORMER BENCHMARK RESULTS\\n")
            f.write("=" * 50 + "\\n\\n")
            
            if 'baseline' in results and 'finetuned' in results:
                f.write("MODEL COMPARISON\\n")
                f.write("-" * 20 + "\\n")
                
                baseline = results['baseline']
                finetuned = results['finetuned']
                
                if 'perplexity' in baseline and 'perplexity' in finetuned:
                    f.write(f"Baseline Perplexity: {baseline['perplexity']:.2f}\\n")
                    f.write(f"Fine-tuned Perplexity: {finetuned['perplexity']:.2f}\\n")
                    f.write(f"Improvement: {results.get('perplexity_improvement', 0):.1f}%\\n\\n")
                
                # Generated samples
                if 'generated_samples' in finetuned:
                    f.write("SAMPLE GENERATIONS\\n")
                    f.write("-" * 20 + "\\n")
                    for i, sample in enumerate(finetuned['generated_samples'][:3]):
                        f.write(f"Sample {i+1}:\\n")
                        f.write(f"Prompt: {sample['prompt'][:100]}...\\n")
                        f.write(f"Generated: {sample['generated']}\\n\\n")
        
        logger.info(f"📄 Summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Switch Transformer models")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../benchmarks")
    parser.add_argument("--baseline_model", type=str, help="Path to baseline model")
    parser.add_argument("--finetuned_model", type=str, help="Path to fine-tuned model")
    parser.add_argument("--model_path", type=str, help="Single model to benchmark")
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir
    }
    
    benchmark = ModelBenchmark(config)
    
    if args.model_path:
        # Benchmark single model
        results = benchmark.benchmark_model(args.model_path, "Custom Model")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark.save_results(results, f"single_model_benchmark_{timestamp}.json")
        
    elif args.baseline_model and args.finetuned_model:
        # Compare models
        results = benchmark.compare_models(args.baseline_model, args.finetuned_model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark.save_results(results, f"model_comparison_{timestamp}.json")
        
    else:
        logger.error("Please provide either --model_path or both --baseline_model and --finetuned_model")

if __name__ == "__main__":
    main()