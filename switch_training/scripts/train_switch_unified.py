#!/usr/bin/env python3
"""
Unified Switch Transformer Training Script

Train any Switch Transformer base model (8, 16, 32, 64, 128, 256 experts) on the processed dataset.
Automatically optimizes settings based on the number of experts.
Designed for A6000 hardware.
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

# Available Switch models
SWITCH_MODELS = {
    8: "google/switch-base-8",
    16: "google/switch-base-16", 
    32: "google/switch-base-32",
    64: "google/switch-base-64",
    128: "google/switch-base-128",
    256: "google/switch-base-256"
}

# Expert-specific optimizations
EXPERT_CONFIGS = {
    8: {
        'batch_size': 6,
        'learning_rate': 5e-6,
        'gradient_accumulation_steps': 2,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'router_weight_scale': 0.01
    },
    16: {
        'batch_size': 5,
        'learning_rate': 4e-6,
        'gradient_accumulation_steps': 3,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'router_weight_scale': 0.01
    },
    32: {
        'batch_size': 4,
        'learning_rate': 3e-6,
        'gradient_accumulation_steps': 4,
        'warmup_ratio': 0.12,
        'max_grad_norm': 0.8,
        'router_weight_scale': 0.01
    },
    64: {
        'batch_size': 4,
        'learning_rate': 3e-6,
        'gradient_accumulation_steps': 4,
        'warmup_ratio': 0.12,
        'max_grad_norm': 0.8,
        'router_weight_scale': 0.01
    },
    128: {
        'batch_size': 3,
        'learning_rate': 2e-6,
        'gradient_accumulation_steps': 5,
        'warmup_ratio': 0.15,
        'max_grad_norm': 0.6,
        'router_weight_scale': 0.01
    },
    256: {
        'batch_size': 3,
        'learning_rate': 2e-6,
        'gradient_accumulation_steps': 6,
        'warmup_ratio': 0.15,
        'max_grad_norm': 0.5,
        'router_weight_scale': 0.01
    }
}

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
        words = text.split()
        if len(words) < 6:
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
        self.loss_history = []
    
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
            else:
                self.nan_count = 0
                self.loss_history.append(current_loss)
        
        # Log router statistics if available
        if hasattr(model, 'router') and len(self.loss_history) > 0:
            recent_loss = self.loss_history[-1]
            logger.info(f"Recent loss: {recent_loss:.4f}")

def apply_switch_stabilization(model, num_experts: int):
    """Apply stabilization techniques based on number of experts"""
    logger.info(f"Applying Switch Transformer stabilization for {num_experts} experts...")
    
    config = EXPERT_CONFIGS[num_experts]
    
    # Scale down router weights
    for name, module in model.named_modules():
        if 'router' in name.lower() and hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.data *= config['router_weight_scale']
                logger.info(f"Scaled router weights in {name} by {config['router_weight_scale']}")
    
    logger.info("✅ Switch stabilization applied")

def get_optimized_config(num_experts: int, args):
    """Get optimized configuration for the number of experts"""
    if num_experts not in EXPERT_CONFIGS:
        raise ValueError(f"Unsupported number of experts: {num_experts}")
    
    config = EXPERT_CONFIGS[num_experts].copy()
    
    # Override with command line arguments if provided
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.gradient_accumulation_steps is not None:
        config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.warmup_ratio is not None:
        config['warmup_ratio'] = args.warmup_ratio
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Train Switch Transformer (Unified)")
    parser.add_argument("--experts", type=int, required=True, choices=[8, 16, 32, 64, 128, 256], 
                        help="Number of experts (8, 16, 32, 64, 128, 256)")
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
    parser.add_argument("--output_dir", type=str, help="Output directory (auto-generated if not provided)")
    parser.add_argument("--batch_size", type=int, help="Batch size (auto-optimized if not provided)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (auto-optimized if not provided)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio (auto-optimized if not provided)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation (auto-optimized if not provided)")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Get model name and optimized config
    model_name = SWITCH_MODELS[args.experts]
    config = get_optimized_config(args.experts, args)
    
    # Set output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"../models/switch_{args.experts}_experts"
    
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
                project="switch-transformer-unified",
                config={**vars(args), **config},
                name=f"switch-{args.experts}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    # Load tokenizer and model
    logger.info(f"Loading Switch Transformer with {args.experts} experts...")
    logger.info(f"Model: {model_name}")
    logger.info(f"Optimized config: {config}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        router_z_loss_coef=0.001,
        router_aux_loss_coef=0.001
    )
    
    # Apply stabilization
    apply_switch_stabilization(model, args.experts)
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
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=0.01,
        warmup_ratio=config['warmup_ratio'],
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb" if not args.disable_wandb else None,
        run_name=f"switch-{args.experts}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        max_grad_norm=config['max_grad_norm'],
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
        'model_name': model_name,
        'experts': args.experts,
        'optimized_config': config,
        'total_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'start_time': datetime.now().isoformat(),
        'args': vars(args),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"🚀 Starting Switch-{args.experts} training")
    logger.info(f"📊 Model: {model_name}")
    logger.info(f"💾 Samples: {training_info['total_samples']:,} train, {training_info['val_samples']:,} val")
    logger.info(f"🧠 Parameters: {training_info['model_parameters']:,}")
    logger.info(f"⚡ Batch size: {config['batch_size']}, LR: {config['learning_rate']}")
    logger.info(f"🔄 Grad accum: {config['gradient_accumulation_steps']}, Warmup: {config['warmup_ratio']}")
    
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
        
        logger.info(f"✅ Switch-{args.experts} training completed successfully!")
        logger.info(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        
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