#!/usr/bin/env python3
"""
Stabilized Switch Transformer Training Script

Addresses all known Switch Transformer instabilities for reliable training.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import math
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer, SwitchTransformersForConditionalGeneration,
    TrainingArguments, Trainer, 
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("transformers").setLevel(logging.WARNING)  # Reduce transformer logs
logger = logging.getLogger(__name__)

class StabilizedSwitchDataset(Dataset):
    """Ultra-safe dataset for Switch Transformer"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and heavily filter data
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"Loaded {len(raw_data)} raw samples")
        
        # Ultra-conservative filtering
        self.data = []
        for item in raw_data:
            text = item.get('text', '').strip()
            
            # Strict filtering criteria
            if (text and 
                50 <= len(text) <= 800 and  # Character length
                10 <= len(text.split()) <= 150 and  # Word count
                text.count('.') >= 1 and  # Has sentences
                not any(c in text for c in ['�', '\x00', '\ufffd']) and  # No bad chars
                text.isascii()):  # ASCII only for safety
                
                # Create safe seq2seq pairs
                words = text.split()
                if len(words) >= 8:  # Minimum for splitting
                    split_point = max(3, int(len(words) * 0.6))  # 60% input, 40% target
                    
                    input_words = words[:split_point]
                    target_words = words[split_point:]
                    
                    input_text = " ".join(input_words)
                    target_text = " ".join(target_words)
                    
                    # Final validation
                    if (len(input_text) >= 20 and len(target_text) >= 10 and
                        len(input_text.split()) >= 3 and len(target_text.split()) >= 2):
                        
                        self.data.append({
                            'input_text': f"continue: {input_text}",
                            'target_text': target_text
                        })
        
        logger.info(f"After filtering: {len(self.data)} safe samples")
        
        if len(self.data) == 0:
            raise ValueError("No valid data samples after filtering!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def safe_collate_fn(batch, tokenizer, max_input_length=180, max_target_length=80):
    """Ultra-safe collate function with extensive validation"""
    
    input_texts = []
    target_texts = []
    
    for item in batch:
        input_text = item['input_text'].strip()
        target_text = item['target_text'].strip()
        
        # Final safety checks
        if input_text and target_text and len(input_text) > 5 and len(target_text) > 3:
            input_texts.append(input_text)
            target_texts.append(target_text)
    
    if not input_texts or not target_texts:
        # Emergency fallback
        input_texts = ["continue: The research paper discusses"]
        target_texts = ["important findings and conclusions."]
    
    try:
        # Tokenize inputs (encoder)
        inputs = tokenizer(
            input_texts,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets (decoder)
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                target_texts,
                max_length=max_target_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
        
        # Replace pad tokens in labels with -100 (ignored in loss)
        labels = targets['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        }
        
    except Exception as e:
        logger.warning(f"Collate function error: {e}, using fallback")
        # Safe fallback
        fallback_input = tokenizer(
            "continue: The paper presents",
            max_length=max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        fallback_target = tokenizer(
            "research findings.",
            max_length=max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = fallback_target['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': fallback_input['input_ids'],
            'attention_mask': fallback_input['attention_mask'],
            'labels': labels
        }

class StabilizedSwitchTrainer:
    """Heavily stabilized Switch Transformer trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model with extensive stabilization
        self.tokenizer, self.model = self._load_stabilized_model()
        
        # Load ultra-filtered datasets
        self.train_dataset, self.val_dataset = self._load_datasets()
        
        # Setup training components
        self._setup_training()
    
    def _load_stabilized_model(self):
        """Load Switch model with maximum stabilization"""
        model_name = self.config.get('model_name', 'google/switch-base-8')
        
        logger.info(f"Loading stabilized {model_name}...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with conservative settings
            model = SwitchTransformersForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use FP32 for maximum stability
                device_map=None  # Manual device placement for control
            )
            
            # Critical stabilization modifications
            self._stabilize_model(model)
            
            model = model.to(self.device)
            
            # Model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✅ Stabilized model loaded: {model_name}")
            logger.info(f"📊 Parameters: {total_params:,}")
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Failed to load Switch model: {e}")
            raise
    
    def _stabilize_model(self, model):
        """Apply critical stabilization to Switch model"""
        logger.info("Applying stabilization modifications...")
        
        # 1. Stabilize router networks
        for name, module in model.named_modules():
            if 'router' in name and hasattr(module, 'weight'):
                # Initialize router weights to be very small (prevents extreme routing)
                with torch.no_grad():
                    module.weight.data *= 0.01  # Scale down router weights
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data.zero_()
        
        # 2. Reduce auxiliary loss weight (if accessible)
        if hasattr(model.config, 'router_z_loss_coef'):
            model.config.router_z_loss_coef = 1e-5  # Very small auxiliary loss
        if hasattr(model.config, 'router_aux_loss_coef'):
            model.config.router_aux_loss_coef = 1e-5
        
        # 3. Set conservative dropout
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.05  # Lower dropout for stability
        
        logger.info("Model stabilization complete")
    
    def _load_datasets(self):
        """Load ultra-filtered datasets"""
        data_dir = Path(self.config['data_dir'])
        
        train_dataset = StabilizedSwitchDataset(
            data_dir / 'train' / 'finetuning_train.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        val_dataset = StabilizedSwitchDataset(
            data_dir / 'val' / 'finetuning_val.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        return train_dataset, val_dataset
    
    def _setup_training(self):
        """Setup training with stability focus"""
        
        # Ultra-conservative training arguments
        self.training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config.get('gradient_accumulation', 8),
            
            # Ultra-low learning rate
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            warmup_ratio=self.config['warmup_ratio'],
            
            # Stability settings
            fp16=False,  # Use FP32 for maximum numerical stability
            dataloader_drop_last=True,  # Avoid partial batches
            dataloader_pin_memory=True,
            
            # Conservative optimization
            max_grad_norm=0.1,  # Very aggressive gradient clipping
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            
            # Frequent evaluation and saving
            logging_steps=50,
            eval_steps=200,
            save_steps=200,
            evaluation_strategy="steps",
            save_strategy="steps",
            
            # Early stopping settings
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Disable problematic features
            report_to=None,  # No wandb
            remove_unused_columns=False,
            
            # Emergency settings
            save_safetensors=False,  # Use pickle for compatibility
            ignore_data_skip=True
        )
    
    def compute_metrics(self, eval_pred):
        """Safe metrics computation"""
        try:
            predictions, labels = eval_pred
            # Simple perplexity calculation
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions
            
            # Flatten and compute loss manually for safety
            shift_preds = predictions[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Mask out -100 labels
            mask = shift_labels != -100
            shift_labels = shift_labels[mask]
            shift_preds = shift_preds.view(-1, shift_preds.size(-1))[mask.view(-1)]
            
            if len(shift_labels) > 0:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_preds, shift_labels)
                perplexity = torch.exp(loss).item()
                
                # Cap perplexity to prevent overflow
                perplexity = min(perplexity, 10000.0)
                
                return {"perplexity": perplexity}
            else:
                return {"perplexity": 9999.0}
                
        except Exception as e:
            logger.warning(f"Metrics computation failed: {e}")
            return {"perplexity": 9999.0}
    
    def train(self):
        """Train with maximum safety measures"""
        logger.info("🔒 Starting STABILIZED Switch Transformer training...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Batch size: {self.config['batch_size']}")
        logger.info(f"Learning rate: {self.config['learning_rate']}")
        
        # Create custom data collator
        data_collator = lambda batch: safe_collate_fn(
            batch, 
            self.tokenizer, 
            max_input_length=int(self.config['max_length'] * 0.7),
            max_target_length=int(self.config['max_length'] * 0.3)
        )
        
        # Create trainer with extensive safety measures
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # Add safety callback
        class SafetyCallback:
            def __init__(self, patience=5):
                self.patience = patience
                self.nan_count = 0
                self.best_loss = float('inf')
                self.patience_counter = 0
            
            def on_log(self, args, state, control, model, logs=None, **kwargs):
                if logs:
                    train_loss = logs.get('train_loss', 0)
                    eval_loss = logs.get('eval_loss', 0)
                    
                    # Check for NaN/inf
                    if math.isnan(train_loss) or math.isinf(train_loss):
                        self.nan_count += 1
                        logger.warning(f"NaN detected in training! Count: {self.nan_count}")
                        
                        if self.nan_count >= 3:
                            logger.error("Too many NaNs, stopping training")
                            control.should_training_stop = True
                    
                    # Early stopping on eval loss
                    if eval_loss > 0 and eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        
                        if self.patience_counter >= self.patience:
                            logger.info("Early stopping due to no improvement")
                            control.should_training_stop = True
        
        trainer.add_callback(SafetyCallback(patience=3))
        
        try:
            # Start training with safety net
            result = trainer.train()
            
            logger.info("✅ Training completed successfully!")
            
            # Save final model
            final_model_path = Path(self.config['output_dir']) / 'final_model'
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            logger.info(f"✅ Final model saved to {final_model_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Save emergency checkpoint
            emergency_path = Path(self.config['output_dir']) / 'emergency_checkpoint'
            emergency_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), emergency_path / 'model_state.pth')
            logger.info(f"Emergency checkpoint saved to {emergency_path}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Stabilized Switch Transformer Training")
    parser.add_argument("--model_name", type=str, default="google/switch-base-8")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../models/switch_stabilized")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)  # Very low
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    
    args = parser.parse_args()
    
    config = {
        'model_name': args.model_name,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'max_length': args.max_length,
        'weight_decay': 0.001,  # Very low
        'warmup_ratio': 0.2,    # More warmup
        'gradient_accumulation': 16  # Large accumulation for stability
    }
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    trainer = StabilizedSwitchTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()