"""
Gating Model Trainer
Training framework for learnable gating models using masked training approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
# import wandb  # Optional for logging
try:
    import wandb
except ImportError:
    wandb = None
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict

from .learnable_gating_models import (
    GatingModelConfig, 
    create_gating_model,
    GatingDataset,
    collate_fn
)
from .gating_data_collector import GatingDataPoint

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training gating models"""
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_steps: int = 1000
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "onecycle", "linear"
    
    # Loss configuration
    routing_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.1
    consistency_loss_weight: float = 0.05
    diversity_loss_weight: float = 0.02
    
    # Masking strategy
    mask_ratio: float = 0.15           # Fraction of tokens to mask
    mask_strategy: str = "random"      # "random", "block", "layer_aware"
    
    # Regularization
    gradient_clip_norm: float = 1.0
    dropout_rate: float = 0.1
    
    # Validation
    validation_split: float = 0.2
    eval_steps: int = 500
    
    # Logging
    log_steps: int = 100
    save_steps: int = 1000
    
    # Paths
    output_dir: str = "trained_models"
    wandb_project: str = "specmoe_gating"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

class GatingLoss(nn.Module):
    """Custom loss function for gating prediction"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Loss functions
        self.routing_loss = nn.CrossEntropyLoss(reduction='none')
        self.confidence_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(
        self,
        pred_logits: torch.Tensor,      # [batch_size, seq_len, num_experts]
        pred_confidence: torch.Tensor,   # [batch_size, seq_len, 1]
        target_routing: torch.Tensor,    # [batch_size, seq_len, num_experts]
        target_top_k: torch.Tensor,      # [batch_size, seq_len, top_k]
        mask: torch.Tensor,              # [batch_size, seq_len] - which tokens to compute loss for
        attention_mask: torch.Tensor     # [batch_size, seq_len] - padding mask
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size, seq_len, num_experts = pred_logits.shape
        
        # Create valid token mask (not padded and not masked)
        valid_mask = (~attention_mask) & mask
        
        # 1. Routing prediction loss
        # Convert target routing to class labels (top-1 expert)
        target_classes = torch.argmax(target_routing, dim=-1)  # [batch_size, seq_len]
        
        # Compute cross-entropy loss
        routing_loss = self.routing_loss(
            pred_logits.view(-1, num_experts),
            target_classes.view(-1)
        ).view(batch_size, seq_len)
        
        # Apply mask
        routing_loss = routing_loss * valid_mask.float()
        routing_loss = routing_loss.sum() / valid_mask.sum().clamp(min=1)
        
        # 2. Confidence calibration loss
        # Target confidence based on how well the prediction matches
        pred_probs = F.softmax(pred_logits, dim=-1)
        target_probs = F.softmax(target_routing, dim=-1)
        
        # Compute KL divergence as confidence target
        kl_div = F.kl_div(pred_probs.log(), target_probs, reduction='none').sum(dim=-1)
        target_confidence = torch.exp(-kl_div).unsqueeze(-1)  # High confidence for low KL
        
        confidence_loss = self.confidence_loss(
            pred_confidence.view(-1, 1),
            target_confidence.view(-1, 1)
        ).view(batch_size, seq_len, 1)
        
        confidence_loss = confidence_loss.squeeze(-1) * valid_mask.float()
        confidence_loss = confidence_loss.sum() / valid_mask.sum().clamp(min=1)
        
        # 3. Consistency loss (predictions should be consistent across similar tokens)
        consistency_loss = torch.tensor(0.0, device=pred_logits.device)
        if seq_len > 1:
            # Compute similarity between adjacent tokens
            pred_sim = F.cosine_similarity(pred_logits[:, :-1], pred_logits[:, 1:], dim=-1)
            target_sim = F.cosine_similarity(target_routing[:, :-1], target_routing[:, 1:], dim=-1)
            
            consistency_loss = self.mse_loss(pred_sim, target_sim)
            adj_mask = valid_mask[:, :-1] & valid_mask[:, 1:]
            consistency_loss = consistency_loss * adj_mask.float()
            consistency_loss = consistency_loss.sum() / adj_mask.sum().clamp(min=1)
        
        # 4. Diversity loss (encourage diverse expert usage)
        diversity_loss = torch.tensor(0.0, device=pred_logits.device)
        if valid_mask.sum() > 1:
            # Compute expert usage distribution
            pred_probs_masked = pred_probs * valid_mask.unsqueeze(-1).float()
            expert_usage = pred_probs_masked.sum(dim=(0, 1)) / valid_mask.sum()
            
            # Encourage uniform distribution
            uniform_dist = torch.ones_like(expert_usage) / num_experts
            diversity_loss = F.kl_div(expert_usage.log(), uniform_dist, reduction='sum')
        
        # Combine losses
        total_loss = (
            self.config.routing_loss_weight * routing_loss +
            self.config.confidence_loss_weight * confidence_loss +
            self.config.consistency_loss_weight * consistency_loss +
            self.config.diversity_loss_weight * diversity_loss
        )
        
        # Return loss components for logging
        loss_dict = {
            'total_loss': total_loss,
            'routing_loss': routing_loss,
            'confidence_loss': confidence_loss,
            'consistency_loss': consistency_loss,
            'diversity_loss': diversity_loss
        }
        
        return total_loss, loss_dict

class MaskedGatingTrainer:
    """Trainer for gating models using masked training approach"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        gating_config: GatingModelConfig,
        train_dataset: GatingDataset,
        val_dataset: Optional[GatingDataset] = None
    ):
        self.model = model
        self.config = config
        self.gating_config = gating_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup data loaders first
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.loss_fn = GatingLoss(config)
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,  # Disable multiprocessing
                pin_memory=True
            )
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Trainer initialized with {len(train_dataset)} training samples")
        if val_dataset:
            logger.info(f"Validation set: {len(val_dataset)} samples")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config.num_epochs
        
        if self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=total_steps)
        elif self.config.scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1
            )
        else:
            return None
    
    def create_mask(self, batch_size: int, seq_len: int, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create masking pattern for training"""
        
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        if self.config.mask_strategy == "random":
            # Random masking
            for i in range(batch_size):
                valid_positions = (~attention_mask[i]).nonzero().squeeze(-1)
                if len(valid_positions) > 0:
                    num_mask = int(len(valid_positions) * self.config.mask_ratio)
                    mask_indices = torch.randperm(len(valid_positions))[:num_mask]
                    mask[i, valid_positions[mask_indices]] = False
        
        elif self.config.mask_strategy == "block":
            # Block masking
            for i in range(batch_size):
                valid_positions = (~attention_mask[i]).nonzero().squeeze(-1)
                if len(valid_positions) > 0:
                    block_size = max(1, int(len(valid_positions) * self.config.mask_ratio))
                    start_idx = torch.randint(0, max(1, len(valid_positions) - block_size + 1), (1,)).item()
                    mask[i, valid_positions[start_idx:start_idx + block_size]] = False
        
        return mask
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device with special handling for prev_layer_gates
            device_batch = {}
            for k, v in batch.items():
                if k == 'prev_layer_gates':
                    # Handle list of tensors
                    device_batch[k] = [tensor.to(self.device) for tensor in v]
                elif isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(self.device)
                else:
                    device_batch[k] = v
            batch = device_batch
            
            # Create training mask
            batch_size, seq_len = batch['hidden_states'].shape[:2]
            training_mask = self.create_mask(batch_size, seq_len, batch['attention_mask'])
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_step(batch, training_mask)
                    total_loss, loss_dict = outputs
            else:
                outputs = self._forward_step(batch, training_mask)
                total_loss, loss_dict = outputs
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            epoch_losses.append(total_loss.item())
            for key, value in loss_dict.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value.item())
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Logging
            if self.global_step % self.config.log_steps == 0:
                self._log_training_step(loss_dict, epoch)
            
            # Validation
            if self.config.eval_steps > 0 and self.global_step % self.config.eval_steps == 0:
                val_metrics = self.validate()
                self._log_validation_step(val_metrics, epoch)
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(epoch, batch_idx)
        
        # Compute epoch metrics
        epoch_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        return epoch_metrics
    
    def _forward_step(self, batch: Dict, training_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for a single step"""
        
        # Get model predictions
        pred_logits, pred_confidence, attention_weights = self.model(
            hidden_states=batch['hidden_states'],
            prev_layer_gates=batch['prev_layer_gates'],
            layer_id=batch['layer_id'],
            attention_mask=batch['attention_mask']
        )
        
        # Compute loss
        total_loss, loss_dict = self.loss_fn(
            pred_logits=pred_logits,
            pred_confidence=pred_confidence,
            target_routing=batch['target_routing'],
            target_top_k=batch.get('target_top_k'),
            mask=training_mask,
            attention_mask=batch['attention_mask']
        )
        
        return total_loss, loss_dict
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        
        if not self.val_loader:
            return {}
        
        self.model.eval()
        val_losses = []
        val_metrics = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # No masking for validation
                batch_size, seq_len = batch['hidden_states'].shape[:2]
                val_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
                
                # Forward pass
                total_loss, loss_dict = self._forward_step(batch, val_mask)
                
                val_losses.append(total_loss.item())
                for key, value in loss_dict.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value.item())
        
        # Compute average metrics
        val_metrics = {f"val_{key}": np.mean(values) for key, values in val_metrics.items()}
        
        self.model.train()
        return val_metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop"""
        
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update training stats
            self.training_stats['train_losses'].append(train_metrics['total_loss'])
            self.training_stats['learning_rates'].append(train_metrics['learning_rate'])
            
            if val_metrics:
                self.training_stats['val_losses'].append(val_metrics['val_total_loss'])
                
                # Save best model
                if val_metrics['val_total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_total_loss']
                    self.save_checkpoint(epoch, -1, is_best=True)
            
            epoch_time = time.time() - epoch_start
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"  Train loss: {train_metrics['total_loss']:.4f}")
            if val_metrics:
                logger.info(f"  Val loss: {val_metrics['val_total_loss']:.4f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            logger.info(f"  LR: {train_metrics['learning_rate']:.6f}")
        
        logger.info("Training completed!")
        return self.training_stats
    
    def save_checkpoint(self, epoch: int, batch_idx: int, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'training_stats': self.training_stats,
            'config': asdict(self.config),
            'gating_config': asdict(self.gating_config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def _log_training_step(self, loss_dict: Dict, epoch: int):
        """Log training step metrics"""
        
        metrics = {f"train/{key}": value.item() for key, value in loss_dict.items()}
        metrics['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
        metrics['train/epoch'] = epoch
        
        if wandb and wandb.run:
            wandb.log(metrics, step=self.global_step)
    
    def _log_validation_step(self, val_metrics: Dict, epoch: int):
        """Log validation step metrics"""
        
        if wandb and wandb.run:
            wandb.log(val_metrics, step=self.global_step)

def train_gating_model(
    data_points: List[GatingDataPoint],
    model_type: str = "contextual",
    training_config: Optional[TrainingConfig] = None,
    gating_config: Optional[GatingModelConfig] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a gating model on collected data
    
    Args:
        data_points: Training data
        model_type: Type of model to train
        training_config: Training configuration
        gating_config: Model configuration
    
    Returns:
        Trained model and training statistics
    """
    
    # Default configs
    if training_config is None:
        training_config = TrainingConfig()
    if gating_config is None:
        gating_config = GatingModelConfig()
    
    # Create datasets
    full_dataset = GatingDataset(data_points, gating_config.max_seq_len)
    
    # Train/val split
    train_size = int(len(full_dataset) * (1 - training_config.validation_split))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create model
    model = create_gating_model(model_type, gating_config)
    
    # Create trainer
    trainer = MaskedGatingTrainer(
        model=model,
        config=training_config,
        gating_config=gating_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # Train
    training_stats = trainer.train()
    
    return model, training_stats

if __name__ == "__main__":
    # Example usage
    logger.info("Testing gating trainer...")
    
    # Create dummy data
    from .gating_data_collector import GatingDataPoint
    
    dummy_data = []
    for i in range(100):
        dp = GatingDataPoint(
            layer_id=i % 6,
            hidden_states=torch.randn(64, 512),
            input_embeddings=torch.randint(0, 1000, (64,)),
            target_routing=torch.randn(64, 8),
            target_top_k=torch.randint(0, 8, (64, 1)),
            sequence_length=64,
            dataset_name="test",
            sample_id=f"sample_{i}"
        )
        dummy_data.append(dp)
    
    # Train model
    model, stats = train_gating_model(
        dummy_data,
        model_type="contextual",
        training_config=TrainingConfig(num_epochs=2, batch_size=4),
        gating_config=GatingModelConfig(hidden_size=512, num_experts=8)
    )
    
    logger.info("✅ Training test completed successfully!")
    logger.info(f"Final train loss: {stats['train_losses'][-1]:.4f}")
    logger.info(f"Final val loss: {stats['val_losses'][-1]:.4f}")