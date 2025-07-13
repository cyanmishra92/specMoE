#!/usr/bin/env python3
"""
Small MoE Training Script

Uses a smaller, RTX 3090-friendly MoE model or creates a simple MoE layer.
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    GPT2LMHeadModel, GPT2Config
)
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleMoELayer(nn.Module):
    """Simple MoE layer that can be added to existing models"""
    
    def __init__(self, d_model: int, num_experts: int = 8, expert_capacity: int = None):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity or d_model * 2
        
        # Router/gating network
        self.router = nn.Linear(d_model, num_experts)
        
        # Expert networks (simple FFNs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])
        
        # Load balancing
        self.load_balance_loss_weight = 0.01
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Flatten for routing
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        
        # Router logits
        router_logits = self.router(x_flat)  # (batch_size * seq_len, num_experts)
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # Select top-1 expert per token (simpler than top-k)
        expert_indices = torch.argmax(router_probs, dim=-1)  # (batch_size * seq_len)
        
        # Process through experts
        outputs = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Mask for tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx)
            
            if expert_mask.sum() > 0:  # If any tokens assigned to this expert
                expert_tokens = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_tokens)
                outputs[expert_mask] = expert_output
        
        # Reshape back
        outputs = outputs.view(batch_size, seq_len, d_model)
        
        # Load balancing loss (simplified)
        expert_usage = torch.bincount(expert_indices, minlength=self.num_experts).float()
        expert_usage = expert_usage / expert_usage.sum()
        uniform_dist = torch.ones_like(expert_usage) / self.num_experts
        load_balance_loss = torch.mean((expert_usage - uniform_dist) ** 2)
        
        return outputs, load_balance_loss

class MoEGPT2(nn.Module):
    """GPT-2 model with MoE layers"""
    
    def __init__(self, config):
        super().__init__()
        
        # Base GPT-2 configuration (smaller for RTX 3090)
        gpt_config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,  # Smaller embedding
            n_layer=12,  # Fewer layers
            n_head=12,
            n_inner=3072,
            activation_function='gelu_new',
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True
        )
        
        # Load base GPT-2 model
        self.base_model = GPT2LMHeadModel(gpt_config)
        
        # Add MoE layers to some transformer blocks
        self.moe_layers = nn.ModuleList()
        moe_layer_indices = [3, 6, 9]  # Add MoE to layers 3, 6, 9
        
        for layer_idx in moe_layer_indices:
            if layer_idx < len(self.base_model.transformer.h):
                moe_layer = SimpleMoELayer(
                    d_model=gpt_config.n_embd,
                    num_experts=8,  # 8 experts per MoE layer
                    expert_capacity=gpt_config.n_embd * 2
                )
                self.moe_layers.append(moe_layer)
        
        self.moe_layer_indices = moe_layer_indices[:len(self.moe_layers)]
        self.config = gpt_config
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get base model hidden states
        base_outputs = self.base_model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = base_outputs.last_hidden_state
        all_hidden_states = base_outputs.hidden_states
        
        # Apply MoE layers
        total_load_balance_loss = 0
        
        for i, layer_idx in enumerate(self.moe_layer_indices):
            if i < len(self.moe_layers):
                # Get hidden states from the specified layer
                layer_hidden_states = all_hidden_states[layer_idx]
                
                # Apply MoE
                moe_output, load_balance_loss = self.moe_layers[i](layer_hidden_states)
                
                # Residual connection
                hidden_states = hidden_states + moe_output
                total_load_balance_loss += load_balance_loss
        
        # Final language modeling head
        lm_logits = self.base_model.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add load balancing loss
            aux_loss = total_load_balance_loss * 0.01  # Small weight
            loss = lm_loss + aux_loss
        
        return {'loss': loss, 'logits': lm_logits}

class SmallMoEDataset(Dataset):
    """Dataset for small MoE fine-tuning"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        
        # Filter and prepare data
        valid_data = []
        for item in self.data:
            text = item.get('text', '').strip()
            if text and 20 <= len(text) <= 1500:
                valid_data.append(text)
        
        self.data = valid_data
        logger.info(f"After filtering: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Add special formatting
        formatted_text = f"<|startoftext|>{text}<|endoftext|>"
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For causal LM
        }

class SmallMoETrainer:
    """Small MoE trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create MoE model
        self.model = MoEGPT2(config).to(self.device)
        
        # Load datasets
        self.train_dataset, self.val_dataset = self._load_datasets()
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup scheduler
        total_steps = len(self.train_dataset) // config['batch_size'] * config['num_epochs']
        warmup_steps = int(total_steps * config['warmup_ratio'])
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_datasets(self):
        """Load datasets"""
        data_dir = Path(self.config['data_dir'])
        
        train_dataset = SmallMoEDataset(
            data_dir / 'train' / 'finetuning_train.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        val_dataset = SmallMoEDataset(
            data_dir / 'val' / 'finetuning_val.json',
            self.tokenizer,
            self.config['max_length']
        )
        
        return train_dataset, val_dataset
    
    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Check for valid loss
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
    
    def train(self):
        """Main training loop"""
        logger.info("Starting Small MoE training...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"\n=== Epoch {epoch+1}/{self.config['num_epochs']} ===")
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:  # Save every epoch
                checkpoint_dir = Path(self.config['output_dir']) / f'epoch_{epoch+1}'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save(self.model.state_dict(), checkpoint_dir / 'model.pth')
                self.tokenizer.save_pretrained(checkpoint_dir)
                
                logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
        logger.info("✅ Training completed!")

def main():
    parser = argparse.ArgumentParser(description="Train Small MoE on research papers")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../models/small_moe")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'max_length': args.max_length,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'gradient_clip': 1.0
    }
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    trainer = SmallMoETrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()