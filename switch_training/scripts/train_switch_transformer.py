#!/usr/bin/env python3
"""
Switch Transformer Fine-tuning Script

Fine-tune a pre-trained Switch Transformer on custom research papers.
Optimized for RTX 3090 (24GB VRAM).
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Transformers and datasets
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, SwitchTransformersForConditionalGeneration,
    TrainingArguments, Trainer, 
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset, load_dataset

# Monitoring
import wandb
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchPaperDataset(Dataset):
    """Dataset for Switch Transformer seq2seq fine-tuning"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        
        # Filter by token length (ensure reasonable range)
        self.data = [item for item in self.data if 20 <= item['tokens'] <= max_length - 20]
        logger.info(f"After filtering: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # For seq2seq: create input-target pairs by splitting text
        words = text.split()
        if len(words) < 4:
            # Fallback for very short texts
            source_text = "Summarize: " + text
            target_text = text[:50] + "..."
        else:
            # Split text into input (first 70%) and target (last 30%)
            split_point = int(len(words) * 0.7)
            source_words = words[:split_point]
            target_words = words[split_point:]
            
            source_text = "Continue: " + " ".join(source_words)
            target_text = " ".join(target_words)
        
        # Tokenize source (encoder input)
        source_encoding = self.tokenizer(
            source_text,
            truncation=True,
            max_length=int(self.max_length * 0.7),
            padding=False,
            return_tensors=None
        )
        
        # Tokenize target (decoder output)
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            max_length=int(self.max_length * 0.3),
            padding=False,
            return_tensors=None
        )
        
        return {
            'input_ids': source_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
            'labels': target_encoding['input_ids']
        }

class SwitchTrainer:
    """Switch Transformer fine-tuning trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = self._setup_tokenizer()
        
        # Initialize model
        self.model = self._setup_model()
        
        # Load datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self._load_datasets()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup tracking
        self._setup_logging()
    
    def _setup_tokenizer(self):
        """Initialize tokenizer"""
        try:
            # Try Switch Transformer tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        except:
            # Fallback to T5 tokenizer (Switch is based on T5)
            logger.warning("Switch tokenizer not available, using T5 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained('t5-base')
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
        return tokenizer
    
    def _setup_model(self):
        """Initialize Switch Transformer model"""
        # List of Switch models to try (in order of preference for RTX 3090)
        switch_models = [
            self.config['model_name'],
            "google/switch-base-16",
            "google/switch-base-8", 
            "google/switch-base-32"
        ]
        
        model = None
        model_used = None
        
        # Try Switch Transformer models
        for model_name in switch_models:
            try:
                logger.info(f"Attempting to load {model_name}...")
                if "switch" in model_name.lower():
                    # Use Switch Transformer specific class
                    model = SwitchTransformersForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.config['use_fp16'] else torch.float32,
                        device_map='auto' if self.config['use_device_map'] else None
                    )
                else:
                    # Use general seq2seq class
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.config['use_fp16'] else torch.float32,
                        device_map='auto' if self.config['use_device_map'] else None
                    )
                model_used = model_name
                logger.info(f"✅ Successfully loaded {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        # Fallback to T5 if no Switch model works
        if model is None:
            logger.warning("No Switch Transformer models available, using T5 fallback")
            try:
                from transformers import T5ForConditionalGeneration
                model = T5ForConditionalGeneration.from_pretrained('t5-small')
                model_used = "t5-small (fallback)"
            except Exception as e:
                logger.error(f"Failed to load fallback model: {e}")
                raise
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model loaded: {model_used}")
        logger.info(f"📊 Total parameters: {total_params:,}")
        logger.info(f"🏋️ Trainable parameters: {trainable_params:,}")
        logger.info(f"💾 Model size: ~{total_params * 4 / 1024**3:.2f} GB (fp32)")
        
        # Store model info for later use
        self.model_info = {
            'model_name': model_used,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
        
        return model.to(self.device)
    
    def _load_datasets(self):
        """Load training datasets"""
        data_dir = Path(self.config['data_dir'])
        
        train_dataset = ResearchPaperDataset(
            data_dir / 'train' / 'finetuning_train.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        val_dataset = ResearchPaperDataset(
            data_dir / 'val' / 'finetuning_val.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        test_dataset = ResearchPaperDataset(
            data_dir / 'test' / 'finetuning_test.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def _setup_optimizer(self):
        """Setup optimizer with weight decay"""
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_params = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_params,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        logger.info(f"Optimizer setup: lr={self.config['learning_rate']}, weight_decay={self.config['weight_decay']}")
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = len(self.train_dataset) // self.config['batch_size'] * self.config['num_epochs']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Scheduler setup: {total_steps} total steps, {warmup_steps} warmup steps")
        return scheduler
    
    def _setup_logging(self):
        """Setup experiment tracking"""
        if self.config['use_wandb']:
            wandb.init(
                project="switch-transformer-finetuning",
                config=self.config,
                name=f"switch-finetune-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        # Use DataCollatorForSeq2Seq for proper padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=data_collator
        )
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with gradient accumulation
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / self.config.get('gradient_accumulation', 1)
            
            # Check for NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation - only step every N batches
            if (batch_idx + 1) % self.config.get('gradient_accumulation', 1) == 0:
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
            
            # Log to wandb
            if self.config['use_wandb'] and batch_idx % 50 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': epoch * len(dataloader) + batch_idx
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1} - Average training loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, epoch: int):
        """Evaluate on validation set"""
        self.model.eval()
        
        # Use DataCollatorForSeq2Seq for validation too
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=data_collator
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
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
        
        # Log to wandb
        if self.config['use_wandb']:
            wandb.log({
                'val_loss': avg_loss,
                'val_perplexity': perplexity,
                'epoch': epoch
            })
        
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
        
        # Save training state
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, model_dir / 'training_state.pth')
        
        # Save best model
        if is_best:
            best_dir = checkpoint_dir / 'best_model'
            best_dir.mkdir(exist_ok=True)
            
            self.model.save_pretrained(best_dir)
            self.tokenizer.save_pretrained(best_dir)
            
            logger.info(f"✅ Best model saved to {best_dir}")
        
        logger.info(f"Checkpoint saved: {model_dir}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting Switch Transformer fine-tuning...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"\\n=== Epoch {epoch+1}/{self.config['num_epochs']} ===")
            
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
        
        if self.config['use_wandb']:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Switch Transformer")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--model_name", type=str, default="google/switch-base-8")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../models")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Configuration
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
        'patience': 3,
        'use_fp16': True,
        'use_device_map': True,
        'use_wandb': args.use_wandb
    }
    
    # Create trainer and start training
    trainer = SwitchTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()