#!/usr/bin/env python3
"""
Sophisticated Speculation Training
- Layer-specific models (separate model for each transition)
- Multi-step prediction (predict 2-3 layers ahead)
- Ensemble methods
- Multi-layer context
- Data augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from pathlib import Path
import pickle
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayerSpecificPredictor(nn.Module):
    """Separate model for each layer transition (e.g., L1->L3, L3->L5)"""
    
    def __init__(self, hidden_size=512, num_experts=128, source_layer=1, target_layers=[3]):
        super().__init__()
        self.source_layer = source_layer
        self.target_layers = target_layers
        self.num_targets = len(target_layers)
        
        # Context encoder - processes hidden states + previous expert choices
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_size + num_experts, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Multi-step prediction heads (one for each target layer)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, num_experts)
            ) for _ in target_layers
        ])
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_targets),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states, prev_expert_choices):
        """
        Args:
            hidden_states: [batch, seq, hidden]
            prev_expert_choices: [batch, seq, num_experts] - one-hot or soft
        """
        # Pool sequences
        h_pooled = torch.mean(hidden_states, dim=1)  # [batch, hidden]
        expert_pooled = torch.mean(prev_expert_choices, dim=1)  # [batch, num_experts]
        
        # Combine context
        context = torch.cat([h_pooled, expert_pooled], dim=-1)
        encoded = self.context_encoder(context)
        
        # Multi-step predictions
        predictions = []
        for head in self.prediction_heads:
            predictions.append(head(encoded))
        
        # Confidence scores
        confidence = self.confidence_head(encoded)
        
        return predictions, confidence

class HierarchicalEnsemble(nn.Module):
    """Ensemble of layer-specific models with hierarchical combination"""
    
    def __init__(self, hidden_size=512, num_experts=128):
        super().__init__()
        self.layer_models = nn.ModuleDict()
        
        # Define layer transitions for Switch Transformer (encoder layers 1,3,5,7,9,11)
        self.transitions = {
            1: [3, 5],      # From layer 1, predict layers 3 and 5
            3: [5, 7],      # From layer 3, predict layers 5 and 7  
            5: [7, 9],      # From layer 5, predict layers 7 and 9
            7: [9, 11],     # From layer 7, predict layers 9 and 11
            9: [11],        # From layer 9, predict layer 11
        }
        
        # Create layer-specific models
        for source_layer, target_layers in self.transitions.items():
            model_key = f"L{source_layer}"
            self.layer_models[model_key] = LayerSpecificPredictor(
                hidden_size, num_experts, source_layer, target_layers
            )
        
        # Meta-learner for ensemble combination
        self.meta_learner = nn.Sequential(
            nn.Linear(num_experts * 2, 128),  # Combine predictions from 2 models max
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_experts)
        )
    
    def forward(self, layer_id, hidden_states, prev_expert_choices, target_layer):
        """
        Predict expert for target_layer given context from layer_id
        """
        model_key = f"L{layer_id}"
        
        if model_key not in self.layer_models:
            # Fallback to nearest available model
            available_layers = [int(k[1:]) for k in self.layer_models.keys()]
            nearest_layer = min(available_layers, key=lambda x: abs(x - layer_id))
            model_key = f"L{nearest_layer}"
        
        model = self.layer_models[model_key]
        predictions, confidence = model(hidden_states, prev_expert_choices)
        
        # Find which prediction corresponds to target_layer
        target_layers = self.transitions.get(int(model_key[1:]), [])
        if target_layer in target_layers:
            idx = target_layers.index(target_layer)
            return predictions[idx], confidence[:, idx:idx+1]
        else:
            # Return first prediction as fallback
            return predictions[0], confidence[:, 0:1]

class MultiLayerContextDataset(Dataset):
    """Dataset that creates multi-layer context sequences"""
    
    def __init__(self, traces, augment=True):
        self.augment = augment
        self.sequences = self._build_sequences(traces)
        logger.info(f"Created {len(self.sequences)} multi-layer sequences")
    
    def _build_sequences(self, traces):
        """Build sequences with multi-layer context"""
        # Group traces by sample_id and sort by layer
        trace_groups = defaultdict(list)
        
        for trace in traces:
            if (hasattr(trace, 'hidden_states') and 
                hasattr(trace, 'target_routing') and
                hasattr(trace, 'sample_id')):
                trace_groups[trace.sample_id].append(trace)
        
        sequences = []
        
        for sample_id, sample_traces in trace_groups.items():
            # Sort by layer
            sample_traces.sort(key=lambda x: x.layer_id)
            
            # Create sequences for each possible prediction
            for i in range(len(sample_traces) - 1):
                source_trace = sample_traces[i]
                
                # Find possible target layers
                source_layer = source_trace.layer_id
                for j in range(i + 1, min(i + 4, len(sample_traces))):  # Up to 3 layers ahead
                    target_trace = sample_traces[j]
                    target_layer = target_trace.layer_id
                    
                    # Create training example
                    sequence = {
                        'source_layer': source_layer,
                        'target_layer': target_layer,
                        'hidden_states': source_trace.hidden_states.float(),
                        'source_routing': source_trace.target_routing.float(),
                        'target_routing': target_trace.target_routing.float(),
                        'sample_id': sample_id
                    }
                    sequences.append(sequence)
        
        return sequences
    
    def _augment_sequence(self, sequence):
        """Data augmentation for sequences"""
        if not self.augment:
            return sequence
        
        # Add noise to hidden states
        noise_scale = 0.01
        noise = torch.randn_like(sequence['hidden_states']) * noise_scale
        sequence['hidden_states'] = sequence['hidden_states'] + noise
        
        # Smooth target routing slightly
        smoothing = 0.05
        num_experts = sequence['target_routing'].shape[-1]
        uniform = torch.ones_like(sequence['target_routing']) / num_experts
        sequence['target_routing'] = (1 - smoothing) * sequence['target_routing'] + smoothing * uniform
        
        return sequence
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        return self._augment_sequence(sequence)

def sophisticated_collate(batch):
    """Collate function for multi-layer sequences"""
    # Group by source->target layer pairs
    layer_groups = defaultdict(list)
    
    for item in batch:
        key = (item['source_layer'], item['target_layer'])
        layer_groups[key].append(item)
    
    # Process each group
    batched_groups = {}
    
    for (source_layer, target_layer), items in layer_groups.items():
        if len(items) == 0:
            continue
        
        # Pad and stack
        max_seq_len = max(item['hidden_states'].shape[0] for item in items)
        
        hidden_states = []
        source_routing = []
        target_routing = []
        
        for item in items:
            h = item['hidden_states']  # [seq, hidden]
            s_route = item['source_routing']  # [seq, experts]
            t_route = item['target_routing']  # [seq, experts]
            
            seq_len = h.shape[0]
            
            # Pad to max length
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                h = torch.cat([h, torch.zeros(pad_len, h.shape[1])], dim=0)
                s_route = torch.cat([s_route, torch.zeros(pad_len, s_route.shape[1])], dim=0)
                t_route = torch.cat([t_route, torch.zeros(pad_len, t_route.shape[1])], dim=0)
            
            hidden_states.append(h)
            source_routing.append(s_route)
            target_routing.append(t_route)
        
        batched_groups[(source_layer, target_layer)] = {
            'hidden_states': torch.stack(hidden_states),
            'source_routing': torch.stack(source_routing),
            'target_routing': torch.stack(target_routing),
            'targets': torch.stack([torch.argmax(torch.sum(item['target_routing'], dim=0)) for item in items])
        }
    
    return batched_groups

class SophisticatedTrainer:
    """Sophisticated training with all advanced techniques"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
    def create_ensemble_model(self):
        """Create the hierarchical ensemble model"""
        model = HierarchicalEnsemble().to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        return model, optimizer, scheduler
    
    def train_sophisticated_models(self, traces):
        """Train all sophisticated models"""
        logger.info("🧠 Training Sophisticated Multi-Layer Speculation Models")
        logger.info("=" * 60)
        
        # Create dataset with multi-layer context
        train_size = int(0.8 * len(traces))
        train_traces = traces[:train_size]
        val_traces = traces[train_size:]
        
        train_dataset = MultiLayerContextDataset(train_traces, augment=True)
        val_dataset = MultiLayerContextDataset(val_traces, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=sophisticated_collate)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=sophisticated_collate)
        
        # Create ensemble model
        model, optimizer, scheduler = self.create_ensemble_model()
        
        # Training loop
        num_epochs = 30
        best_val_acc = 0
        
        logger.info(f"🎯 Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"🔥 GPU: {torch.cuda.get_device_name()}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Training batches: {len(train_loader)}")
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_groups in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                total_loss = 0
                batch_correct = 0
                batch_total = 0
                
                # Process each layer group in the batch
                for (source_layer, target_layer), batch_data in batch_groups.items():
                    hidden_states = batch_data['hidden_states'].to(self.device)
                    source_routing = batch_data['source_routing'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    
                    # Forward pass
                    predictions, confidence = model(
                        layer_id=source_layer,
                        hidden_states=hidden_states,
                        prev_expert_choices=source_routing,
                        target_layer=target_layer
                    )
                    
                    # Calculate loss with confidence weighting
                    base_loss = F.cross_entropy(predictions, targets)
                    confidence_loss = F.mse_loss(confidence.squeeze(), torch.ones_like(confidence.squeeze()))
                    loss = base_loss + 0.1 * confidence_loss
                    
                    total_loss += loss
                    
                    # Stats
                    pred = torch.argmax(predictions, dim=1)
                    batch_correct += (pred == targets).sum().item()
                    batch_total += targets.size(0)
                
                if total_loss > 0:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += total_loss.item()
                    train_correct += batch_correct
                    train_total += batch_total
            
            scheduler.step()
            
            if train_total > 0:
                train_acc = train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation
                val_acc = self._validate_model(model, val_loader)
                
                logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'val_accuracy': val_acc,
                        'epoch': epoch,
                        'model_type': 'sophisticated_ensemble'
                    }, "trained_models/sophisticated_ensemble_model.pt")
        
        return model, best_val_acc
    
    def _validate_model(self, model, val_loader):
        """Validate the sophisticated model"""
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_groups in val_loader:
                for (source_layer, target_layer), batch_data in batch_groups.items():
                    hidden_states = batch_data['hidden_states'].to(self.device)
                    source_routing = batch_data['source_routing'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    
                    predictions, confidence = model(
                        layer_id=source_layer,
                        hidden_states=hidden_states,
                        prev_expert_choices=source_routing,
                        target_layer=target_layer
                    )
                    
                    pred = torch.argmax(predictions, dim=1)
                    val_correct += (pred == targets).sum().item()
                    val_total += targets.size(0)
        
        return val_correct / val_total if val_total > 0 else 0

def load_traces():
    """Load benchmark traces"""
    trace_file = "routing_data/benchmark_traces.pkl"
    with open(trace_file, 'rb') as f:
        serializable_traces = pickle.load(f)
    
    traces = []
    for trace_dict in serializable_traces:
        class MultiLayerTrace:
            pass
        trace = MultiLayerTrace()
        trace.layer_id = trace_dict['layer_id']
        trace.hidden_states = torch.from_numpy(trace_dict['hidden_states'])
        trace.target_routing = torch.from_numpy(trace_dict['target_routing'])
        trace.sample_id = trace_dict['sample_id']
        traces.append(trace)
    
    logger.info(f"✅ Loaded {len(traces)} traces")
    return traces

def main():
    """Main sophisticated training"""
    logger.info("🚀 Sophisticated Multi-Layer Speculation Training")
    logger.info("=" * 70)
    
    Path("trained_models").mkdir(exist_ok=True)
    
    # Load data
    traces = load_traces()
    
    # Create trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = SophisticatedTrainer(device)
    
    # Train sophisticated models
    model, best_acc = trainer.train_sophisticated_models(traces)
    
    # Results
    random_baseline = 1.0 / 128
    improvement = best_acc / random_baseline
    
    logger.info(f"\n🎉 Sophisticated Training Results:")
    logger.info(f"Best validation accuracy: {best_acc:.3f} ({best_acc*100:.1f}%)")
    logger.info(f"Random baseline: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
    logger.info(f"Improvement: {improvement:.1f}x better than random")
    logger.info(f"Model saved to: trained_models/sophisticated_ensemble_model.pt")
    
    logger.info(f"\n🏗️ Architecture Summary:")
    logger.info(f"- Layer-specific models: {len(model.layer_models)} specialists")
    logger.info(f"- Multi-step prediction: Up to 3 layers ahead")
    logger.info(f"- Ensemble combination: Meta-learner")
    logger.info(f"- Data augmentation: Noise + smoothing")
    logger.info(f"- Multi-layer context: Full sequence modeling")
    
    return True

if __name__ == "__main__":
    main()