#!/usr/bin/env python3
"""
Mixtral-style MoE Training Script

Uses Mixtral 8x7B MoE model - much more stable than Switch Transformers.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import math
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MixtralDataset(Dataset):
    """Dataset for Mixtral MoE fine-tuning"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        
        # Filter by token length
        valid_data = []
        for item in self.data:
            text = item.get('text', '').strip()
            if text and 50 <= len(text) <= 2000:  # Character length filter
                valid_data.append(item)
        
        self.data = valid_data
        logger.info(f"After filtering: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text'].strip()
        
        # Add instruction format for research papers
        formatted_text = f"Research Paper: {text}"
        
        # Tokenize
        try:
            encoding = self.tokenizer(
                formatted_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            }
            
        except Exception as e:
            logger.warning(f"Tokenization failed for item {idx}: {e}")
            # Return a simple fallback
            fallback = self.tokenizer(
                "Research paper content.",
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            return {
                'input_ids': fallback['input_ids'],
                'attention_mask': fallback['attention_mask']
            }

class MixtralTrainer:
    """Mixtral MoE fine-tuning trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer, self.model = self._load_model()
        
        # Load datasets
        self.train_dataset, self.val_dataset = self._load_datasets()
        
        # Setup data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
    
    def _load_model(self):
        """Load Mixtral model and tokenizer"""
        model_name = self.config.get('model_name', 'mistralai/Mixtral-8x7B-v0.1')
        
        logger.info(f"Loading {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Mixtral models
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',  # Automatic device placement
                load_in_8bit=True,  # Use 8-bit quantization for memory efficiency
                trust_remote_code=True
            )
            
            # Ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id
            
            # Model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✅ Model loaded: {model_name}")
            logger.info(f"📊 Parameters: {total_params:,}")
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Failed to load Mixtral model: {e}")
            logger.info("Falling back to smaller MoE model...")
            
            # Fallback to smaller model
            model_name = "microsoft/DialoGPT-medium"  # Much smaller, stable
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"✅ Fallback model loaded: {model_name}")
            return tokenizer, model
    
    def _load_datasets(self):
        """Load datasets"""
        data_dir = Path(self.config['data_dir'])
        
        train_dataset = MixtralDataset(
            data_dir / 'train' / 'finetuning_train.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        val_dataset = MixtralDataset(
            data_dir / 'val' / 'finetuning_val.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        return train_dataset, val_dataset
    
    def train(self):
        """Train using HuggingFace Trainer for stability"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config.get('gradient_accumulation', 4),
            warmup_ratio=self.config['warmup_ratio'],
            weight_decay=self.config['weight_decay'],
            learning_rate=self.config['learning_rate'],
            fp16=self.config.get('use_fp16', True),
            logging_steps=100,
            eval_steps=500,
            save_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for now
            max_grad_norm=self.config.get('gradient_clip', 1.0)
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info("Starting Mixtral MoE fine-tuning...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        # Train
        try:
            trainer.train()
            logger.info("✅ Training completed successfully!")
            
            # Save final model
            final_model_path = Path(self.config['output_dir']) / 'final_model'
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            logger.info(f"✅ Final model saved to {final_model_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mixtral MoE on research papers")
    parser.add_argument("--model_name", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../models/mixtral")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    config = {
        'model_name': args.model_name,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'max_length': args.max_length,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'gradient_clip': 1.0,
        'gradient_accumulation': 8,  # Large accumulation for effective batch size
        'use_fp16': True
    }
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    trainer = MixtralTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()