#!/usr/bin/env python3
"""
Distributed Switch Transformer Training Script

Multi-GPU training for Switch Transformers with automatic GPU detection and selection.
Supports any number of GPUs with automatic fallback and optimization.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
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

# Multi-GPU optimized configurations
DISTRIBUTED_CONFIGS = {
    8: {
        'base_batch_size': 8,
        'learning_rate': 6e-6,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'router_weight_scale': 0.01
    },
    16: {
        'base_batch_size': 6,
        'learning_rate': 5e-6,
        'gradient_accumulation_steps': 1,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'router_weight_scale': 0.01
    },
    32: {
        'base_batch_size': 4,
        'learning_rate': 4e-6,
        'gradient_accumulation_steps': 2,
        'warmup_ratio': 0.12,
        'max_grad_norm': 0.8,
        'router_weight_scale': 0.01
    },
    64: {
        'base_batch_size': 4,
        'learning_rate': 4e-6,
        'gradient_accumulation_steps': 2,
        'warmup_ratio': 0.12,
        'max_grad_norm': 0.8,
        'router_weight_scale': 0.01
    },
    128: {
        'base_batch_size': 1,
        'learning_rate': 2e-6,
        'gradient_accumulation_steps': 4,
        'warmup_ratio': 0.15,
        'max_grad_norm': 0.6,
        'router_weight_scale': 0.01
    },
    256: {
        'base_batch_size': 1,
        'learning_rate': 1e-6,
        'gradient_accumulation_steps': 6,
        'warmup_ratio': 0.15,
        'max_grad_norm': 0.5,
        'router_weight_scale': 0.01
    }
}

def get_gpu_info():
    """Get detailed GPU information"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_total': int(parts[2]),
                        'memory_free': int(parts[3]),
                        'utilization': int(parts[4])
                    })
        return gpus
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return []

def select_best_gpus(requested_gpus: int, min_memory_gb: int = 16) -> List[int]:
    """Select the best available GPUs"""
    gpus = get_gpu_info()
    
    if not gpus:
        logger.error("No GPUs found!")
        return []
    
    # Filter GPUs by memory and utilization
    suitable_gpus = []
    for gpu in gpus:
        memory_gb = gpu['memory_free'] / 1024
        if memory_gb >= min_memory_gb and gpu['utilization'] < 90:
            suitable_gpus.append(gpu)
    
    if len(suitable_gpus) < requested_gpus:
        logger.warning(f"Only {len(suitable_gpus)} suitable GPUs available, requested {requested_gpus}")
        print(f"\nAvailable GPUs:")
        for gpu in gpus:
            memory_gb = gpu['memory_total'] / 1024
            free_gb = gpu['memory_free'] / 1024
            utilization = gpu['utilization']
            status = "✅ Available" if gpu in suitable_gpus else "❌ Busy/Low Memory"
            print(f"  GPU {gpu['index']}: {gpu['name']} - {memory_gb:.1f}GB total, {free_gb:.1f}GB free, {utilization}% util - {status}")
        
        # Ask user to select fewer GPUs
        while True:
            try:
                new_count = int(input(f"\nEnter number of GPUs to use (max {len(suitable_gpus)}): "))
                if 1 <= new_count <= len(suitable_gpus):
                    requested_gpus = new_count
                    break
                else:
                    print(f"Please enter a number between 1 and {len(suitable_gpus)}")
            except ValueError:
                print("Please enter a valid number")
    
    # Sort by free memory (descending) and select top N
    suitable_gpus.sort(key=lambda x: x['memory_free'], reverse=True)
    selected_gpus = [gpu['index'] for gpu in suitable_gpus[:requested_gpus]]
    
    logger.info(f"Selected GPUs: {selected_gpus}")
    for i, gpu_idx in enumerate(selected_gpus):
        gpu = next(g for g in gpus if g['index'] == gpu_idx)
        logger.info(f"  GPU {gpu_idx}: {gpu['name']} - {gpu['memory_free']/1024:.1f}GB free")
    
    return selected_gpus

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
            'labels': target_encoding['input_ids'].flatten()
        }

class SafetyCallback(TrainerCallback):
    """Safety callback for distributed training"""
    
    def __init__(self, patience: int = 3, rank: int = 0):
        self.patience = patience
        self.nan_count = 0
        self.rank = rank
        self.loss_history = []
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None or self.rank != 0:  # Only log from rank 0
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

def apply_switch_stabilization(model, num_experts: int):
    """Apply stabilization techniques based on number of experts"""
    config = DISTRIBUTED_CONFIGS[num_experts]
    
    # Scale down router weights
    for name, module in model.named_modules():
        if 'router' in name.lower() and hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.data *= config['router_weight_scale']
    
    logger.info(f"✅ Applied stabilization for {num_experts} experts")

def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    logger.info(f"Rank {rank}/{world_size} initialized")

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def get_distributed_config(num_experts: int, world_size: int, args):
    """Get distributed configuration"""
    base_config = DISTRIBUTED_CONFIGS[num_experts].copy()
    
    # Scale batch size with number of GPUs
    base_config['per_device_batch_size'] = base_config['base_batch_size']
    base_config['effective_batch_size'] = base_config['base_batch_size'] * world_size
    
    # Adjust learning rate for multi-GPU (linear scaling)
    base_config['learning_rate'] *= world_size
    
    # Override with command line arguments if provided
    if args.batch_size is not None:
        base_config['per_device_batch_size'] = args.batch_size
    if args.learning_rate is not None:
        base_config['learning_rate'] = args.learning_rate
    if args.gradient_accumulation_steps is not None:
        base_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.warmup_ratio is not None:
        base_config['warmup_ratio'] = args.warmup_ratio
    
    return base_config

def train_distributed(rank: int, world_size: int, gpu_ids: List[int], args):
    """Main distributed training function"""
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Set GPU device - use local rank for device mapping
    local_device_id = rank  # Use rank as local device ID
    device = torch.device(f'cuda:{local_device_id}')
    torch.cuda.set_device(local_device_id)
    
    # Get model configuration
    model_name = SWITCH_MODELS[args.experts]
    config = get_distributed_config(args.experts, world_size, args)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"../models/switch_{args.experts}_experts_distributed"
    
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test wandb early and disable completely if it fails
    if not args.disable_wandb:
        try:
            # Only test on rank 0
            if rank == 0:
                import wandb
                # Test if wandb is configured
                if not wandb.api.api_key:
                    raise Exception("No wandb API key found")
        except Exception as e:
            if rank == 0:
                logger.warning(f"Wandb not available: {e}")
            args.disable_wandb = True
            # Set environment variable to completely disable wandb
            os.environ['WANDB_DISABLED'] = 'true'
    
    # Sync wandb status across all ranks early
    disable_wandb_tensor = torch.tensor([args.disable_wandb], dtype=torch.bool, device=device)
    dist.broadcast(disable_wandb_tensor, 0)
    args.disable_wandb = disable_wandb_tensor.item()
    
    # Set environment variable on all ranks if disabled
    if args.disable_wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Now initialize wandb properly if not disabled
    if rank == 0 and not args.disable_wandb:
        try:
            wandb.init(
                project="switch-transformer-distributed",
                config={**vars(args), **config, 'world_size': world_size},
                name=f"switch-{args.experts}-{world_size}gpu-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            args.disable_wandb = True
            os.environ['WANDB_DISABLED'] = 'true'
    
    # Load tokenizer and model
    if rank == 0:
        logger.info(f"Loading Switch Transformer with {args.experts} experts on {world_size} GPUs...")
        logger.info(f"Model: {model_name}")
        logger.info(f"Distributed config: {config}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        router_z_loss_coef=0.001,
        router_aux_loss_coef=0.001,
        low_cpu_mem_usage=True
    )
    
    # Apply stabilization
    apply_switch_stabilization(model, args.experts)
    
    # Clear cache before moving model to device
    torch.cuda.empty_cache()
    
    # Move model to device
    model = model.to(device)
    
    # Clear cache again after model move
    torch.cuda.empty_cache()
    
    # Wrap model with DDP - use local device ID
    model = DDP(model, device_ids=[local_device_id], find_unused_parameters=False)
    
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
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Data collator - pass the underlying model for DDP
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.module if hasattr(model, 'module') else model,
        padding=True,
        max_length=args.max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=config['per_device_batch_size'],
        per_device_eval_batch_size=config['per_device_batch_size'],
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
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="wandb" if (rank == 0 and not args.disable_wandb) else None,
        run_name=f"switch-{args.experts}-{world_size}gpu-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        gradient_checkpointing=False,
        dataloader_pin_memory=False,  # Disable pin memory for large models
        max_grad_norm=config['max_grad_norm'],
        seed=42,
        logging_first_step=True,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
        # Distributed training specific
        local_rank=rank,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        # Memory optimizations
        dataloader_persistent_workers=False,
        skip_memory_metrics=True,
    )
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.patience),
        SafetyCallback(patience=3, rank=rank)
    ]
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # Save training info (only from rank 0)
    if rank == 0:
        training_info = {
            'model_name': model_name,
            'experts': args.experts,
            'world_size': world_size,
            'gpu_ids': gpu_ids,
            'distributed_config': config,
            'total_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'effective_batch_size': config['effective_batch_size'],
            'start_time': datetime.now().isoformat(),
            'args': vars(args),
            'model_parameters': sum(p.numel() for p in model.module.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        }
        
        with open(output_dir / 'training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"🚀 Starting distributed Switch-{args.experts} training")
        logger.info(f"💾 Samples: {training_info['total_samples']:,} train, {training_info['val_samples']:,} val")
        logger.info(f"🧠 Parameters: {training_info['model_parameters']:,}")
        logger.info(f"⚡ Per-device batch: {config['per_device_batch_size']}, Effective batch: {config['effective_batch_size']}")
        logger.info(f"📊 GPUs: {world_size} x {gpu_ids}")
    
    # Start training
    try:
        trainer.train()
        
        # Save final model (only from rank 0)
        if rank == 0:
            logger.info("💾 Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # Update training info
            training_info['end_time'] = datetime.now().isoformat()
            training_info['status'] = 'completed'
            
            with open(output_dir / 'training_info.json', 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info(f"✅ Distributed Switch-{args.experts} training completed!")
            logger.info(f"📁 Model saved to: {output_dir}")
    
    except Exception as e:
        if rank == 0:
            logger.error(f"❌ Training failed: {e}")
            
            training_info['end_time'] = datetime.now().isoformat()
            training_info['status'] = 'failed'
            training_info['error'] = str(e)
            
            with open(output_dir / 'training_info.json', 'w') as f:
                json.dump(training_info, f, indent=2)
        
        raise
    
    finally:
        if rank == 0 and not args.disable_wandb:
            try:
                wandb.finish()
            except:
                pass
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Distributed Switch Transformer Training")
    parser.add_argument("--experts", type=int, required=True, choices=[8, 16, 32, 64, 128, 256], 
                        help="Number of experts (8, 16, 32, 64, 128, 256)")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
    parser.add_argument("--output_dir", type=str, help="Output directory (auto-generated if not provided)")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size (auto-optimized if not provided)")
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
    parser.add_argument("--single_gpu", action="store_true", help="Force single GPU training")
    
    args = parser.parse_args()
    
    # Check if single GPU training is requested
    if args.single_gpu:
        logger.info("Single GPU training requested, falling back to unified script")
        # Import and run the unified script
        from train_switch_unified import main as unified_main
        
        # Modify sys.argv to match unified script expectations
        sys.argv = [sys.argv[0], '--experts', str(args.experts)]
        if args.data_dir != "../data":
            sys.argv.extend(['--data_dir', args.data_dir])
        if args.output_dir:
            sys.argv.extend(['--output_dir', args.output_dir])
        if args.batch_size:
            sys.argv.extend(['--batch_size', str(args.batch_size)])
        if args.learning_rate:
            sys.argv.extend(['--learning_rate', str(args.learning_rate)])
        if args.num_epochs != 3:
            sys.argv.extend(['--num_epochs', str(args.num_epochs)])
        if args.disable_wandb:
            sys.argv.append('--disable_wandb')
        
        unified_main()
        return
    
    # Select GPUs
    logger.info(f"Requesting {args.gpus} GPUs for distributed training...")
    gpu_ids = select_best_gpus(args.gpus, min_memory_gb=16)
    
    if not gpu_ids:
        logger.error("No suitable GPUs found. Exiting.")
        return
    
    world_size = len(gpu_ids)
    
    if world_size == 1:
        logger.info("Only 1 GPU available, using single GPU training")
        # Import and run the unified script
        from train_switch_unified import main as unified_main
        
        # Modify sys.argv to match unified script expectations
        sys.argv = [sys.argv[0], '--experts', str(args.experts)]
        if args.data_dir != "../data":
            sys.argv.extend(['--data_dir', args.data_dir])
        if args.output_dir:
            sys.argv.extend(['--output_dir', args.output_dir])
        if args.batch_size:
            sys.argv.extend(['--batch_size', str(args.batch_size)])
        if args.learning_rate:
            sys.argv.extend(['--learning_rate', str(args.learning_rate)])
        if args.num_epochs != 3:
            sys.argv.extend(['--num_epochs', str(args.num_epochs)])
        if args.disable_wandb:
            sys.argv.append('--disable_wandb')
        
        unified_main()
        return
    
    logger.info(f"Starting distributed training on {world_size} GPUs: {gpu_ids}")
    
    # Set environment variables for distributed training
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    # Launch distributed training - pass reindexed gpu_ids
    # After setting CUDA_VISIBLE_DEVICES, GPU indices start from 0
    reindexed_gpu_ids = list(range(world_size))
    mp.spawn(
        train_distributed,
        args=(world_size, reindexed_gpu_ids, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()