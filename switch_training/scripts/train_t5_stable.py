#!/usr/bin/env python3
"""
T5 Fine-tuning Script (Stable Alternative to Switch Transformer)

Fine-tune T5 on research papers - more stable than Switch Transformers.
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
    T5Tokenizer, T5ForConditionalGeneration,
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
logger = logging.getLogger(__name__)

class T5ResearchDataset(Dataset):
    """T5 Dataset for research paper fine-tuning"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        
        # Filter by token length
        self.data = [item for item in self.data if 20 <= item['tokens'] <= max_length - 20]
        logger.info(f"After filtering: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Create seq2seq pairs
        words = text.split()
        if len(words) < 6:
            source_text = "summarize: " + text
            target_text = text[:100] + "..."
        else:
            # Split: first 70% as input, last 30% as target
            split_point = int(len(words) * 0.7)
            source_words = words[:split_point]
            target_words = words[split_point:]
            
            source_text = "continue: " + " ".join(source_words)
            target_text = " ".join(target_words)
        
        return {
            'input_text': source_text,
            'target_text': target_text
        }

def collate_fn(batch, tokenizer, max_length=512):
    """Custom collate function for T5"""
    input_texts = [item['input_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    
    # Tokenize inputs
    inputs = tokenizer(
        input_texts,
        max_length=int(max_length * 0.7),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # Tokenize targets
    targets = tokenizer(
        target_texts,
        max_length=int(max_length * 0.3),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids']
    }

class T5Trainer:
    """T5 fine-tuning trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer, self.model = self._load_model()
        
        # Load datasets
        self.train_dataset, self.val_dataset = self._load_datasets()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
    
    def _load_model(self):
        """Load T5 model and tokenizer"""
        model_name = self.config.get('model_name', 't5-base')
        
        logger.info(f"Loading {model_name}...")
        
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.get('use_fp16', True) else torch.float32
        )
        
        model = model.to(self.device)
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model loaded: {model_name}")
        logger.info(f"📊 Parameters: {total_params:,}")
        
        return tokenizer, model
    
    def _load_datasets(self):
        """Load datasets"""
        data_dir = Path(self.config['data_dir'])
        
        train_dataset = T5ResearchDataset(
            data_dir / 'train' / 'finetuning_train.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        val_dataset = T5ResearchDataset(
            data_dir / 'val' / 'finetuning_val.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        return train_dataset, val_dataset
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        from torch.optim import AdamW
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        logger.info(f"Optimizer: lr={self.config['learning_rate']}")
        return optimizer
    
    def _setup_scheduler(self):
        """Setup scheduler"""
        total_steps = len(self.train_dataset) // self.config['batch_size'] * self.config['num_epochs']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Scheduler: {total_steps} total steps, {warmup_steps} warmup")
        return scheduler
    
    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer, self.config['max_length'])
        )
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Replace pad tokens in labels with -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Check for issues
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at batch {batch_idx}, skipping")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Tracking
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1} - Training loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self, epoch: int):
        """Evaluate on validation set"""
        self.model.eval()
        
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer, self.config['max_length'])
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Replace pad tokens in labels with -100
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['output_dir']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        model_dir = checkpoint_dir / f"epoch_{epoch+1}"
        model_dir.mkdir(exist_ok=True)
        
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        if is_best:
            best_dir = checkpoint_dir / 'best_model'
            best_dir.mkdir(exist_ok=True)
            
            self.model.save_pretrained(best_dir)
            self.tokenizer.save_pretrained(best_dir)
            
            logger.info(f"✅ Best model saved to {best_dir}")
        
        logger.info(f"Checkpoint saved: {model_dir}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting T5 fine-tuning...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"\n=== Epoch {epoch+1}/{self.config['num_epochs']} ===")
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_perplexity = self.evaluate(epoch)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("✅ Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5 on research papers")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../models")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
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
        'patience': 2,
        'use_fp16': True
    }
    
    trainer = T5Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()