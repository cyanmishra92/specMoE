#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluate trained models (Small MoE, Switch Transformer, etc.) with comprehensive metrics.
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
from transformers import AutoTokenizer, GPT2LMHeadModel, SwitchTransformersForConditionalGeneration
import numpy as np
from tqdm import tqdm

# Import our custom models
sys.path.append('.')
from train_small_moe import MoEGPT2, SmallMoEDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load test data
        self.test_data = self._load_test_data()
    
    def _load_test_data(self) -> List[str]:
        """Load test dataset"""
        test_file = Path(self.config['data_dir']) / 'test' / 'finetuning_test.json'
        
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract just the text
        texts = [item['text'] for item in data if item.get('text')]
        logger.info(f"Loaded {len(texts)} test samples")
        return texts
    
    def load_small_moe_model(self, model_path: str) -> Tuple[object, object]:
        """Load Small MoE model"""
        logger.info(f"Loading Small MoE model from {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create model structure
        model = MoEGPT2(self.config)
        
        # Load trained weights
        checkpoint_path = Path(model_path) / 'model.pth'
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"✅ Loaded model weights from {checkpoint_path}")
        else:
            logger.warning(f"No model weights found at {checkpoint_path}")
        
        model = model.to(self.device)
        model.eval()
        
        return model, tokenizer
    
    def load_switch_model(self, model_path: str) -> Tuple[object, object]:
        """Load Switch Transformer model"""
        logger.info(f"Loading Switch Transformer model from {model_path}")
        
        try:
            # Load tokenizer and model from the saved directory
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = SwitchTransformersForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # Use FP32 for evaluation stability
                device_map=None
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"✅ Switch Transformer loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load Switch Transformer: {e}")
            raise
    
    def calculate_perplexity(self, model, tokenizer, model_type: str = 'small_moe', max_length: int = 512) -> float:
        """Calculate perplexity on test set"""
        logger.info("Calculating perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for text in tqdm(self.test_data[:100], desc="Computing perplexity"):  # Use subset for speed
                try:
                    if model_type == 'switch':
                        # Switch Transformer seq2seq approach
                        words = text.split()
                        if len(words) < 6:
                            continue
                        
                        split_point = int(len(words) * 0.7)
                        input_text = f"continue: {' '.join(words[:split_point])}"
                        target_text = ' '.join(words[split_point:])
                        
                        # Tokenize input (encoder)
                        inputs = tokenizer(
                            input_text,
                            return_tensors='pt',
                            truncation=True,
                            max_length=int(max_length * 0.7),
                            padding=False
                        ).to(self.device)
                        
                        # Tokenize target (decoder)
                        with tokenizer.as_target_tokenizer():
                            targets = tokenizer(
                                target_text,
                                return_tensors='pt',
                                truncation=True,
                                max_length=int(max_length * 0.3),
                                padding=False
                            ).to(self.device)
                        
                        if inputs['input_ids'].size(1) < 2 or targets['input_ids'].size(1) < 2:
                            continue
                        
                        # Forward pass
                        outputs = model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=targets['input_ids']
                        )
                        
                    else:
                        # Small MoE causal LM approach
                        inputs = tokenizer(
                            f"<|startoftext|>{text}<|endoftext|>",
                            return_tensors='pt',
                            truncation=True,
                            max_length=max_length,
                            padding=False
                        ).to(self.device)
                        
                        if inputs['input_ids'].size(1) < 2:
                            continue
                        
                        # Forward pass
                        outputs = model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=inputs['input_ids']
                        )
                    
                    if outputs.loss is not None and not torch.isnan(outputs.loss):
                        total_loss += outputs.loss.item()
                        total_tokens += inputs['input_ids'].size(1)
                        num_batches += 1
                
                except Exception as e:
                    logger.warning(f"Error processing text: {e}")
                    continue
        
        if num_batches == 0:
            logger.error("No valid batches for perplexity calculation")
            return float('inf')
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        logger.info(f"Perplexity: {perplexity:.2f} (avg loss: {avg_loss:.4f})")
        return perplexity
    
    def generate_samples(self, model, tokenizer, model_type: str = 'small_moe', num_samples: int = 5) -> List[Dict]:
        """Generate text samples"""
        logger.info(f"Generating {num_samples} text samples...")
        
        # Use research paper prompts
        prompts = [
            "This research paper presents",
            "The main contribution of this work is",
            "Our experimental results show that",
            "The proposed method achieves",
            "In conclusion, we demonstrate"
        ]
        
        generated_samples = []
        
        with torch.no_grad():
            for i, prompt in enumerate(prompts[:num_samples]):
                try:
                    if model_type == 'switch':
                        # Switch Transformer seq2seq generation
                        input_text = f"continue: {prompt}"
                        inputs = tokenizer(
                            input_text,
                            return_tensors='pt',
                            truncation=True,
                            max_length=50
                        ).to(self.device)
                        
                        # Generate with Switch Transformer
                        outputs = model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=150,
                            num_return_sequences=1,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            no_repeat_ngram_size=2
                        )
                        
                        # Decode only the generated part (skip input)
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                    else:
                        # Small MoE causal LM generation
                        inputs = tokenizer(
                            f"<|startoftext|>{prompt}",
                            return_tensors='pt',
                            truncation=True,
                            max_length=50
                        ).to(self.device)
                        
                        # Generate
                        outputs = model.base_model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=200,
                            num_return_sequences=1,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            no_repeat_ngram_size=2
                        )
                        
                        # Decode
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    generated_samples.append({
                        'prompt': prompt,
                        'generated': generated_text,
                        'length': len(generated_text)
                    })
                    
                    logger.info(f"Sample {i+1}: {generated_text[:100]}...")
                
                except Exception as e:
                    logger.warning(f"Generation failed for prompt {i}: {e}")
                    generated_samples.append({
                        'prompt': prompt,
                        'generated': f"[Generation failed: {e}]",
                        'length': 0
                    })
        
        return generated_samples
    
    def calculate_token_accuracy(self, model, tokenizer, model_type: str = 'small_moe', max_length: int = 512) -> float:
        """Calculate token prediction accuracy with proper model handling"""
        logger.info("Calculating token prediction accuracy...")
        
        if model_type == 'switch':
            # For Switch Transformers, skip token accuracy due to seq2seq complexity
            logger.info("Skipping token accuracy for Switch Transformer (seq2seq model)")
            return -1.0  # Indicate N/A
        
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for text in tqdm(self.test_data[:50], desc="Computing accuracy"):  # Subset for speed
                try:
                    # Tokenize
                    inputs = tokenizer(
                        f"<|startoftext|>{text}<|endoftext|>",
                        return_tensors='pt',
                        truncation=True,
                        max_length=max_length,
                        padding=False
                    ).to(self.device)
                    
                    if inputs['input_ids'].size(1) < 2:
                        continue
                    
                    # Forward pass
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                    
                    logits = outputs['logits']
                    
                    # Calculate accuracy for next token prediction
                    predicted_tokens = torch.argmax(logits[0, :-1], dim=-1)
                    actual_tokens = inputs['input_ids'][0, 1:]
                    
                    # Only count non-padding tokens
                    mask = actual_tokens != tokenizer.pad_token_id
                    
                    correct = (predicted_tokens == actual_tokens) & mask
                    correct_predictions += correct.sum().item()
                    total_predictions += mask.sum().item()
                
                except Exception as e:
                    logger.warning(f"Error in accuracy calculation: {e}")
                    continue
        
        if total_predictions == 0:
            logger.error("No valid predictions for accuracy calculation")
            return 0.0
        
        accuracy = correct_predictions / total_predictions
        logger.info(f"Token accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        return accuracy
    
    def evaluate_research_understanding(self, model, tokenizer, model_type: str = 'small_moe') -> Dict:
        """Evaluate understanding of research concepts"""
        logger.info("Evaluating research paper understanding...")
        
        # Research-specific prompts
        research_prompts = [
            ("Machine learning models can be improved by", "optimization techniques"),
            ("The experimental setup includes", "datasets and metrics"),
            ("Our results demonstrate that", "the proposed method"),
            ("Compared to baseline methods", "our approach shows"),
            ("The main limitation of this work", "is computational cost"),
            ("Future work will focus on", "extending the approach"),
            ("The dataset consists of", "training and test samples"),
            ("We evaluate performance using", "standard benchmarks")
        ]
        
        results = []
        
        with torch.no_grad():
            for prompt, expected_concept in research_prompts:
                try:
                    if model_type == 'switch':
                        # Switch Transformer approach
                        input_text = f"continue: {prompt}"
                        inputs = tokenizer(
                            input_text,
                            return_tensors='pt',
                            max_length=50,
                            truncation=True
                        ).to(self.device)
                        
                        outputs = model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=100,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id
                        )
                        
                        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                    else:
                        # Small MoE approach
                        inputs = tokenizer(
                            f"<|startoftext|>{prompt}",
                            return_tensors='pt',
                            max_length=50,
                            truncation=True
                        ).to(self.device)
                        
                        outputs = model.base_model.generate(
                            inputs['input_ids'],
                            max_length=100,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id
                        )
                        
                        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Simple relevance check
                    completion_lower = completion.lower()
                    contains_concept = any(word in completion_lower for word in expected_concept.split())
                    
                    results.append({
                        'prompt': prompt,
                        'completion': completion,
                        'expected_concept': expected_concept,
                        'contains_concept': contains_concept
                    })
                
                except Exception as e:
                    logger.warning(f"Research understanding test failed: {e}")
                    results.append({
                        'prompt': prompt,
                        'completion': f"[Error: {e}]",
                        'expected_concept': expected_concept,
                        'contains_concept': False
                    })
        
        # Calculate concept coverage
        concept_hits = sum(1 for r in results if r['contains_concept'])
        concept_coverage = concept_hits / len(results) if results else 0
        
        logger.info(f"Research concept coverage: {concept_coverage:.2f} ({concept_hits}/{len(results)})")
        
        return {
            'concept_coverage': concept_coverage,
            'detailed_results': results
        }
    
    def comprehensive_evaluation(self, model_path: str, model_type: str = 'small_moe') -> Dict:
        """Run comprehensive evaluation"""
        logger.info(f"🔬 COMPREHENSIVE EVALUATION: {model_type}")
        logger.info("=" * 60)
        
        results = {
            'model_type': model_type,
            'model_path': model_path,
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_data_size': len(self.test_data)
        }
        
        try:
            # Load model
            if model_type == 'small_moe':
                model, tokenizer = self.load_small_moe_model(model_path)
            elif model_type == 'switch':
                model, tokenizer = self.load_switch_model(model_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Run evaluations
            logger.info("\n1. Calculating Perplexity...")
            results['perplexity'] = self.calculate_perplexity(model, tokenizer, model_type)
            
            logger.info("\n2. Calculating Token Accuracy...")
            results['token_accuracy'] = self.calculate_token_accuracy(model, tokenizer, model_type)
            
            logger.info("\n3. Generating Text Samples...")
            results['generated_samples'] = self.generate_samples(model, tokenizer, model_type)
            
            logger.info("\n4. Evaluating Research Understanding...")
            research_eval = self.evaluate_research_understanding(model, tokenizer, model_type)
            results['research_understanding'] = research_eval
            
            # Model statistics
            results['model_stats'] = {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'vocabulary_size': len(tokenizer)
            }
            
            logger.info("\n📊 EVALUATION COMPLETE")
            logger.info(f"Perplexity: {results['perplexity']:.2f}")
            if results['token_accuracy'] >= 0:
                logger.info(f"Token Accuracy: {results['token_accuracy']:.4f}")
            else:
                logger.info("Token Accuracy: N/A (seq2seq model)")
            logger.info(f"Research Understanding: {research_eval['concept_coverage']:.2f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results"""
        output_path = Path(self.config['output_dir']) / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Results saved to {output_path}")
        
        # Save human-readable summary
        summary_path = output_path.with_suffix('.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("MODEL EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model Type: {results.get('model_type', 'Unknown')}\n")
            f.write(f"Model Path: {results.get('model_path', 'Unknown')}\n")
            f.write(f"Evaluation Date: {results.get('evaluation_timestamp', 'Unknown')}\n")
            f.write(f"Test Samples: {results.get('test_data_size', 0)}\n\n")
            
            # Metrics
            f.write("METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Perplexity: {results.get('perplexity', 'N/A')}\n")
            f.write(f"Token Accuracy: {results.get('token_accuracy', 'N/A'):.4f}\n")
            
            if 'research_understanding' in results:
                f.write(f"Research Understanding: {results['research_understanding']['concept_coverage']:.2f}\n\n")
            
            # Sample generations
            if 'generated_samples' in results:
                f.write("SAMPLE GENERATIONS\n")
                f.write("-" * 20 + "\n")
                for i, sample in enumerate(results['generated_samples'][:3]):
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"Prompt: {sample['prompt']}\n")
                    f.write(f"Generated: {sample['generated']}\n\n")
        
        logger.info(f"📄 Summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_type", type=str, default="small_moe", choices=["small_moe", "switch", "mixtral"])
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../evaluations")
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir
    }
    
    evaluator = ModelEvaluator(config)
    
    # Run evaluation
    results = evaluator.comprehensive_evaluation(args.model_path, args.model_type)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.model_type}_evaluation_{timestamp}.json"
    evaluator.save_results(results, output_file)

if __name__ == "__main__":
    main()