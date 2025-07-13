#!/usr/bin/env python3
"""
Simple Switch Transformer Evaluation

Focus on basic metrics and text generation without complex seq2seq evaluation.
"""

import os
import sys
import json
import logging
from pathlib import Path
import math
from datetime import datetime

import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_evaluation(model_path: str):
    """Simple evaluation focusing on basic functionality"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info("Loading Switch Transformer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model loaded: {total_params:,} parameters")
    
    # Test basic generation
    logger.info("\n🔧 Testing Basic Generation...")
    test_prompts = [
        "continue: This research paper presents",
        "continue: The experimental results show",
        "continue: Our method achieves",
        "continue: The main contribution is",
        "continue: We propose a novel"
    ]
    
    results = {
        'model_path': model_path,
        'total_parameters': total_params,
        'evaluation_time': datetime.now().isoformat(),
        'generation_samples': []
    }
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            try:
                logger.info(f"Testing prompt {i+1}: {prompt}")
                
                # Tokenize
                inputs = tokenizer(
                    prompt,
                    return_tensors='pt',
                    max_length=50,
                    truncation=True
                ).to(device)
                
                # Generate with very conservative settings
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=100,
                    num_return_sequences=1,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                logger.info(f"Generated: {generated_text}")
                
                results['generation_samples'].append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Generation failed for prompt {i+1}: {e}")
                results['generation_samples'].append({
                    'prompt': prompt,
                    'generated': f"[Error: {e}]",
                    'success': False
                })
    
    # Test training loss on a few samples
    logger.info("\n📊 Testing Training Loss...")
    try:
        # Load some training data
        train_file = Path("../data/train/finetuning_train.json")
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        total_loss = 0.0
        valid_samples = 0
        
        for item in train_data[:10]:  # Just test 10 samples
            text = item.get('text', '').strip()
            if not text or len(text) < 20:
                continue
                
            words = text.split()
            if len(words) < 6:
                continue
                
            # Split into input/target like training
            split_point = int(len(words) * 0.7)
            input_text = f"continue: {' '.join(words[:split_point])}"
            target_text = ' '.join(words[split_point:])
            
            try:
                # Tokenize input
                inputs = tokenizer(
                    input_text,
                    return_tensors='pt',
                    max_length=180,
                    truncation=True,
                    padding=False
                ).to(device)
                
                # Tokenize target  
                with tokenizer.as_target_tokenizer():
                    targets = tokenizer(
                        target_text,
                        return_tensors='pt',
                        max_length=80,
                        truncation=True,
                        padding=False
                    ).to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=targets['input_ids']
                )
                
                if outputs.loss is not None and not torch.isnan(outputs.loss):
                    total_loss += outputs.loss.item()
                    valid_samples += 1
                
            except Exception as e:
                logger.warning(f"Loss calculation failed: {e}")
                continue
        
        if valid_samples > 0:
            avg_loss = total_loss / valid_samples
            perplexity = math.exp(avg_loss)
            results['avg_loss'] = avg_loss
            results['perplexity'] = perplexity
            results['valid_samples'] = valid_samples
            
            logger.info(f"Average Loss: {avg_loss:.4f}")
            logger.info(f"Perplexity: {perplexity:.2f}")
        else:
            logger.warning("No valid samples for loss calculation")
            results['avg_loss'] = None
            results['perplexity'] = None
        
    except Exception as e:
        logger.error(f"Loss evaluation failed: {e}")
        results['loss_eval_error'] = str(e)
    
    # Save results
    output_dir = Path("../evaluations")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f"simple_switch_eval_{timestamp}.json"
    txt_file = output_dir / f"simple_switch_eval_{timestamp}.txt"
    
    # Save JSON
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save readable summary
    with open(txt_file, 'w') as f:
        f.write("SIMPLE SWITCH TRANSFORMER EVALUATION\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {model_path}\n")
        f.write(f"Parameters: {results['total_parameters']:,}\n")
        f.write(f"Evaluation Time: {results['evaluation_time']}\n\n")
        
        if results.get('perplexity'):
            f.write(f"Average Loss: {results['avg_loss']:.4f}\n")
            f.write(f"Perplexity: {results['perplexity']:.2f}\n")
            f.write(f"Valid Samples: {results['valid_samples']}\n\n")
        
        f.write("GENERATION SAMPLES\n")
        f.write("-" * 30 + "\n")
        
        successful_gens = [s for s in results['generation_samples'] if s['success']]
        f.write(f"Successful generations: {len(successful_gens)}/5\n\n")
        
        for i, sample in enumerate(results['generation_samples']):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {sample['prompt']}\n")
            f.write(f"Generated: {sample['generated']}\n")
            f.write(f"Success: {sample['success']}\n\n")
    
    logger.info(f"✅ Results saved to {json_file}")
    logger.info(f"📄 Summary saved to {txt_file}")
    
    # Print summary
    logger.info("\n📊 EVALUATION SUMMARY")
    logger.info("=" * 40)
    if results.get('perplexity'):
        logger.info(f"Perplexity: {results['perplexity']:.2f}")
    else:
        logger.info("Perplexity: Unable to calculate")
    
    successful_gens = len([s for s in results['generation_samples'] if s['success']])
    logger.info(f"Successful Generations: {successful_gens}/5")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../models/switch_stabilized/final_model")
    args = parser.parse_args()
    
    simple_evaluation(args.model_path)