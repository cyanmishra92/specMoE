#!/usr/bin/env python3
"""
Large Switch Transformer Training Script

Train the largest Switch Transformer model (google/switch-base-256) on the processed dataset.
Optimized for A6000 hardware with extensive data.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    SwitchTransformersForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from transformers.trainer_callback import TrainerCallback
import numpy as np
from datetime import datetime
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SwitchDataset(Dataset):
    """Dataset for Switch Transformer training"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Create input-output pairs for seq2seq
        # Split text into input (70%) and target (30%)
        words = text.split()
        if len(words) < 6:
            # Fallback for short texts
            input_text = f"continue: {text[:len(text)//2]}"
            target_text = text[len(text)//2:]
        else:
            split_point = int(len(words) * 0.7)
            input_text = f"continue: {' '.join(words[:split_point])}"
            target_text = ' '.join(words[split_point:])
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=int(self.max_length * 0.7),
            padding=False,
            return_tensors='pt'
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(
                target_text,
                truncation=True,
                max_length=int(self.max_length * 0.3),
                padding=False,
                return_tensors='pt'
            )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten(),
            'source': item.get('source', 'unknown')
        }

class SafetyCallback(TrainerCallback):
    """Safety callback to detect and handle training issues"""
    
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.nan_count = 0
        self.last_loss = None
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            return
            
        current_loss = logs.get('train_loss', None)
        if current_loss is not None:
            if np.isnan(current_loss) or np.isinf(current_loss):
                self.nan_count += 1
                logger.warning(f"NaN/Inf loss detected! Count: {self.nan_count}")
                
                if self.nan_count >= self.patience:
                    logger.error("Too many NaN/Inf losses. Stopping training.")
                    control.should_training_stop = True
                    return
                
                # Apply gradient clipping and reduce learning rate
                if hasattr(args, 'learning_rate'):
                    args.learning_rate *= 0.5
                    logger.info(f"Reduced learning rate to {args.learning_rate}")
            else:
                self.nan_count = 0  # Reset counter on valid loss
                
        # Log router stats if available
        if hasattr(model, 'router') and hasattr(model.router, 'aux_loss'):
            aux_loss = model.router.aux_loss
            if aux_loss is not None:
                logger.info(f"Router aux loss: {aux_loss}")

def apply_switch_stabilization(model):
    """Apply stabilization techniques to Switch Transformer"""
    logger.info("Applying Switch Transformer stabilization...")
    
    # Scale down router weights
    for name, module in model.named_modules():
        if 'router' in name.lower() and hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.data *= 0.01
                logger.info(f"Scaled router weights in {name}")
    
    # Enable gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    logger.info("✅ Switch stabilization applied")

def main():
    parser = argparse.ArgumentParser(description="Train Large Switch Transformer")
    parser.add_argument("--model_name", type=str, default="google/switch-base-256", help="Switch model to use")
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="../models/switch_large", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if not args.disable_wandb:
        try:
            wandb.init(
                project="switch-transformer-large",
                config=vars(args),
                name=f"switch-{args.model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # Use FP32 for stability
        device_map=None,  # Handle device placement manually
        router_z_loss_coef=0.001,  # Add router loss coefficient
        router_aux_loss_coef=0.001  # Add auxiliary loss coefficient
    )
    
    # Apply stabilization
    apply_switch_stabilization(model)
    
    # Move to device
    model = model.to(device)
    
    # Create datasets
    data_dir = Path(args.data_dir)
    train_dataset = SwitchDataset(
        data_dir / 'train' / 'train_data.json',
        tokenizer,
        args.max_length
    )
    
    val_dataset = SwitchDataset(
        data_dir / 'val' / 'val_data.json',
        tokenizer,
        args.max_length
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=args.max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=False,  # Use FP32 for stability
        bf16=False,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb" if not args.disable_wandb else None,
        run_name=f"switch-large-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        dataloader_pin_memory=True,
        max_grad_norm=1.0,  # Gradient clipping
        seed=42,
        logging_first_step=True,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
    )
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.patience),
        SafetyCallback(patience=3)
    ]
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    # Save training info
    training_info = {
        'model_name': args.model_name,
        'total_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'start_time': datetime.now().isoformat(),
        'args': vars(args),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"🚀 Starting training with {training_info['total_samples']} samples")
    logger.info(f"📊 Model has {training_info['model_parameters']:,} parameters")
    logger.info(f"🎯 Training for {args.num_epochs} epochs")
    
    # Start training
    try:
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Update training info
        training_info['end_time'] = datetime.now().isoformat()
        training_info['status'] = 'completed'
        
        with open(output_dir / 'training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        
        # Update training info
        training_info['end_time'] = datetime.now().isoformat()
        training_info['status'] = 'failed'
        training_info['error'] = str(e)
        
        with open(output_dir / 'training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        raise
    
    finally:
        if not args.disable_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()