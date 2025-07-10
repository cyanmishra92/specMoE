#!/usr/bin/env python3
"""
Fixed Training Script for 128-Expert Benchmark Traces
Quick fix for tensor dimension mismatches
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from pathlib import Path
import json
import time
import pickle
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Simple128ExpertPredictor(nn.Module):
    """Simple model for 128-expert prediction"""
    
    def __init__(self, hidden_size=512, num_experts=128, context_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.context_layers = context_layers
        
        # Input processing
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)
        
        # Context from previous layers (if available)
        self.context_proj = nn.Linear(num_experts * context_layers, hidden_size // 2) if context_layers > 0 else None
        
        # Prediction head
        input_dim = hidden_size + (hidden_size // 2 if context_layers > 0 else 0)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states, prev_layer_gates=None, layer_id=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Process hidden states
        h = self.hidden_proj(hidden_states)  # [batch, seq, hidden]
        h = torch.mean(h, dim=1)  # Pool sequence dimension: [batch, hidden]
        
        features = [h]
        
        # Add context from previous layers if available
        if prev_layer_gates and self.context_proj:
            try:
                # Flatten and concatenate previous gates
                prev_context = []
                for gate in prev_layer_gates[:self.context_layers]:
                    if gate.dim() == 3:  # [batch, seq, experts]
                        gate_pooled = torch.mean(gate, dim=1)  # [batch, experts]
                    elif gate.dim() == 2:  # [seq, experts]
                        gate_pooled = torch.mean(gate, dim=0).unsqueeze(0)  # [1, experts]
                    else:
                        continue
                    prev_context.append(gate_pooled)
                
                if prev_context:
                    # Pad to context_layers if needed
                    while len(prev_context) < self.context_layers:
                        prev_context.append(torch.zeros_like(prev_context[0]))
                    
                    context = torch.cat(prev_context, dim=-1)  # [batch, experts * context_layers]
                    context_features = self.context_proj(context)
                    features.append(context_features)
            except Exception as e:
                logger.debug(f"Context processing failed: {e}")
        
        # Combine features
        combined = torch.cat(features, dim=-1)
        
        # Predict gating logits
        gating_logits = self.predictor(combined)
        
        # Predict confidence
        confidence = self.confidence_head(combined)
        
        return gating_logits, confidence, combined

class BenchmarkDataset(Dataset):
    """Dataset for benchmark traces"""
    
    def __init__(self, traces, split='train'):
        self.traces = traces
        self.split = split
        
        # Filter valid traces
        self.valid_traces = []
        for trace in traces:
            if (hasattr(trace, 'hidden_states') and 
                hasattr(trace, 'target_routing') and
                trace.hidden_states is not None and 
                trace.target_routing is not None):
                self.valid_traces.append(trace)
        
        logger.info(f"{split} dataset: {len(self.valid_traces)} valid traces")
    
    def __len__(self):
        return len(self.valid_traces)
    
    def __getitem__(self, idx):
        trace = self.valid_traces[idx]
        
        # Get hidden states
        hidden_states = trace.hidden_states.float()
        if hidden_states.dim() == 2:  # [seq, hidden]
            hidden_states = hidden_states.unsqueeze(0)  # [1, seq, hidden]
        
        # Get target routing (convert to top-1 experts)
        target_routing = trace.target_routing.float()
        if target_routing.dim() == 2:  # [seq, experts]
            target_experts = torch.argmax(target_routing, dim=-1)  # [seq]
        else:
            target_experts = target_routing.long()
        
        # Get previous layer gates (limit to last 3)
        prev_gates = []
        if hasattr(trace, 'prev_layer_gates') and trace.prev_layer_gates:
            for gate in trace.prev_layer_gates[-3:]:  # Last 3 layers
                if gate is not None:
                    gate_tensor = gate.float()
                    if gate_tensor.dim() == 2:  # [seq, experts]
                        gate_tensor = gate_tensor.unsqueeze(0)  # [1, seq, experts]
                    prev_gates.append(gate_tensor)
        
        return {
            'hidden_states': hidden_states,
            'target_experts': target_experts,
            'target_routing': target_routing,
            'prev_gates': prev_gates,
            'layer_id': trace.layer_id,
            'sample_id': trace.sample_id
        }

def load_benchmark_traces():
    """Load benchmark traces"""
    trace_file = "routing_data/benchmark_traces.pkl"
    
    logger.info(f"Loading benchmark traces from {trace_file}")
    
    if not Path(trace_file).exists():
        logger.error(f"Trace file not found: {trace_file}")
        return None
    
    with open(trace_file, 'rb') as f:
        serializable_traces = pickle.load(f)
    
    # Convert back to objects
    traces = []
    for trace_dict in serializable_traces:
        # Create a simple object
        class TraceObj:
            pass
        
        trace = TraceObj()
        trace.layer_id = trace_dict['layer_id']
        trace.hidden_states = torch.from_numpy(trace_dict['hidden_states'])
        trace.target_routing = torch.from_numpy(trace_dict['target_routing'])
        trace.prev_layer_gates = [torch.from_numpy(g) for g in trace_dict['prev_layer_gates']]
        trace.sample_id = trace_dict['sample_id']
        trace.dataset_name = trace_dict['dataset_name']
        
        traces.append(trace)
    
    logger.info(f"✅ Loaded {len(traces)} benchmark traces")
    
    # Print statistics
    num_experts = traces[0].target_routing.shape[-1]
    layers = set(trace.layer_id for trace in traces)
    
    logger.info(f"Data configuration:")
    logger.info(f"  Num experts: {num_experts}")
    logger.info(f"  Layers: {sorted(layers)}")
    logger.info(f"  Hidden size: {traces[0].hidden_states.shape[-1]}")
    
    return traces

def train_model(traces, model_name="simple"):
    """Train a simple model on benchmark traces"""
    
    logger.info(f"🧠 Training {model_name} model on benchmark traces")
    
    # Split data
    train_size = int(0.8 * len(traces))
    train_traces = traces[:train_size]
    val_traces = traces[train_size:]
    
    # Create datasets
    train_dataset = BenchmarkDataset(train_traces, 'train')
    val_dataset = BenchmarkDataset(val_traces, 'val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple128ExpertPredictor(
        hidden_size=512,
        num_experts=128,
        context_layers=3
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training on device: {device}")
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0
    training_stats = {'train_losses': [], 'val_losses': [], 'val_accuracies': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            hidden_states = batch['hidden_states'].to(device)
            target_experts = batch['target_experts'].to(device)
            prev_gates = [g.to(device) for g in batch['prev_gates']] if batch['prev_gates'] else []
            
            # Forward pass
            gating_logits, confidence, _ = model(
                hidden_states=hidden_states,
                prev_layer_gates=prev_gates
            )
            
            # Calculate loss (use first token for simplicity)
            if target_experts.dim() > 1:
                target_first_token = target_experts[:, 0]  # [batch]
            else:
                target_first_token = target_experts
            
            loss = criterion(gating_logits, target_first_token)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            pred = torch.argmax(gating_logits, dim=-1)
            train_correct += (pred == target_first_token).sum().item()
            train_total += target_first_token.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.3f}"
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                hidden_states = batch['hidden_states'].to(device)
                target_experts = batch['target_experts'].to(device)
                prev_gates = [g.to(device) for g in batch['prev_gates']] if batch['prev_gates'] else []
                
                gating_logits, confidence, _ = model(
                    hidden_states=hidden_states,
                    prev_layer_gates=prev_gates
                )
                
                if target_experts.dim() > 1:
                    target_first_token = target_experts[:, 0]
                else:
                    target_first_token = target_experts
                
                loss = criterion(gating_logits, target_first_token)
                val_loss += loss.item()
                
                pred = torch.argmax(gating_logits, dim=-1)
                val_correct += (pred == target_first_token).sum().item()
                val_total += target_first_token.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Save stats
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(avg_val_loss)
        training_stats['val_accuracies'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'training_stats': training_stats,
                'val_accuracy': val_acc,
                'epoch': epoch
            }, f"trained_models/benchmark_128expert_model.pt")
    
    logger.info(f"✅ Training completed! Best validation accuracy: {best_val_acc:.3f}")
    
    return model, training_stats, best_val_acc

def collate_fn(batch):
    """Custom collate function for variable sequence lengths"""
    # Handle variable sequence lengths by padding
    max_seq_len = max(b['hidden_states'].squeeze(0).size(0) for b in batch)
    
    # Pad hidden states
    hidden_states = []
    target_experts = []
    target_routing = []
    
    for b in batch:
        h = b['hidden_states'].squeeze(0)  # [seq, hidden]
        t_exp = b['target_experts']        # [seq]
        t_route = b['target_routing']      # [seq, experts]
        
        seq_len = h.size(0)
        
        # Pad sequences to max length
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            h = torch.cat([h, torch.zeros(pad_len, h.size(1))], dim=0)
            t_exp = torch.cat([t_exp, torch.zeros(pad_len, dtype=t_exp.dtype)], dim=0)
            t_route = torch.cat([t_route, torch.zeros(pad_len, t_route.size(1))], dim=0)
        
        hidden_states.append(h)
        target_experts.append(t_exp)
        target_routing.append(t_route)
    
    return {
        'hidden_states': torch.stack(hidden_states),
        'target_experts': torch.stack(target_experts),
        'target_routing': torch.stack(target_routing),
        'prev_gates': batch[0]['prev_gates'] if batch[0]['prev_gates'] else [],
        'layer_id': batch[0]['layer_id'],
        'sample_id': batch[0]['sample_id']
    }

def main():
    """Main training function"""
    
    logger.info("🚀 Training 128-Expert Speculation Model")
    logger.info("=" * 50)
    
    # Load benchmark traces
    traces = load_benchmark_traces()
    if not traces:
        logger.error("Failed to load traces")
        return False
    
    # Train model
    model, stats, best_acc = train_model(traces)
    
    # Print results
    logger.info(f"\n🎉 Training Results:")
    logger.info(f"Best validation accuracy: {best_acc:.3f} ({best_acc*100:.1f}%)")
    logger.info(f"Improvement over random (0.78%): {best_acc/0.0078:.1f}x")
    logger.info(f"Model saved to: trained_models/benchmark_128expert_model.pt")
    
    return True

if __name__ == "__main__":
    Path("trained_models").mkdir(exist_ok=True)
    
    success = main()
    if success:
        print("\n✅ 128-expert model training completed!")
        print("Ready for speculation testing!")
    else:
        print("\n❌ Training failed")