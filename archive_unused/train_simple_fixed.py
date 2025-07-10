#!/usr/bin/env python3
"""
Ultra-Simple Training Script for 128-Expert Benchmark Traces
Minimal architecture to avoid dimension issues
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraSimple128Predictor(nn.Module):
    """Ultra-simple model that just works"""
    
    def __init__(self, hidden_size=512, num_experts=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Simple: just hidden states -> experts
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_experts)
        )
    
    def forward(self, hidden_states):
        # Input: [batch, seq, hidden] -> Pool to [batch, hidden]
        if hidden_states.dim() == 3:
            pooled = torch.mean(hidden_states, dim=1)  # Average over sequence
        else:
            pooled = hidden_states
        
        # Predict expert logits
        logits = self.predictor(pooled)
        return logits

class SimpleDataset(Dataset):
    """Ultra-simple dataset"""
    
    def __init__(self, traces):
        self.valid_traces = []
        
        for trace in traces:
            if (hasattr(trace, 'hidden_states') and 
                hasattr(trace, 'target_routing') and
                trace.hidden_states is not None and 
                trace.target_routing is not None):
                self.valid_traces.append(trace)
        
        logger.info(f"Dataset: {len(self.valid_traces)} valid traces")
    
    def __len__(self):
        return len(self.valid_traces)
    
    def __getitem__(self, idx):
        trace = self.valid_traces[idx]
        
        # Get hidden states [seq, hidden]
        hidden_states = trace.hidden_states.float()
        
        # Get target expert (just use the most common expert for this sequence)
        target_routing = trace.target_routing.float()  # [seq, experts]
        target_experts = torch.argmax(target_routing, dim=-1)  # [seq]
        
        # Use the most frequent expert as the target
        most_common_expert = torch.mode(target_experts).values.item()
        
        return {
            'hidden_states': hidden_states,
            'target_expert': most_common_expert
        }

def simple_collate(batch):
    """Simple collate function"""
    # Just take the mean of hidden states for each sample
    hidden_states = []
    targets = []
    
    for item in batch:
        # Pool sequence dimension
        h_mean = torch.mean(item['hidden_states'], dim=0)  # [hidden]
        hidden_states.append(h_mean)
        targets.append(item['target_expert'])
    
    return {
        'hidden_states': torch.stack(hidden_states),  # [batch, hidden]
        'targets': torch.tensor(targets, dtype=torch.long)  # [batch]
    }

def load_traces():
    """Load benchmark traces"""
    trace_file = "routing_data/benchmark_traces.pkl"
    
    if not Path(trace_file).exists():
        logger.error(f"Trace file not found: {trace_file}")
        return None
    
    with open(trace_file, 'rb') as f:
        serializable_traces = pickle.load(f)
    
    # Convert to simple objects
    traces = []
    for trace_dict in serializable_traces:
        class SimpleTrace:
            pass
        
        trace = SimpleTrace()
        trace.hidden_states = torch.from_numpy(trace_dict['hidden_states'])
        trace.target_routing = torch.from_numpy(trace_dict['target_routing'])
        traces.append(trace)
    
    logger.info(f"✅ Loaded {len(traces)} traces")
    return traces

def train_simple_model():
    """Train the ultra-simple model"""
    
    logger.info("🚀 Ultra-Simple 128-Expert Training")
    logger.info("=" * 40)
    
    # Load data
    traces = load_traces()
    if not traces:
        return False
    
    # Split data
    split_idx = int(0.8 * len(traces))
    train_traces = traces[:split_idx]
    val_traces = traces[split_idx:]
    
    # Create datasets
    train_dataset = SimpleDataset(train_traces)
    val_dataset = SimpleDataset(val_traces)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=simple_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=simple_collate)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UltraSimple128Predictor().to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"🎯 Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU: {torch.cuda.get_device_name()}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Training loop
    num_epochs = 15
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            hidden_states = batch['hidden_states'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(hidden_states)
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Stats
            train_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            train_correct += (pred == targets).sum().item()
            train_total += targets.size(0)
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                hidden_states = batch['hidden_states'].to(device)
                targets = batch['targets'].to(device)
                
                logits = model(hidden_states)
                loss = criterion(logits, targets)
                
                val_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                val_correct += (pred == targets).sum().item()
                val_total += targets.size(0)
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'epoch': epoch
            }, "trained_models/simple_128expert_model.pt")
    
    # Final results
    random_baseline = 1.0 / 128  # 0.78%
    improvement = best_val_acc / random_baseline
    
    logger.info(f"\n🎉 Training Results:")
    logger.info(f"Best validation accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    logger.info(f"Random baseline: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
    logger.info(f"Improvement: {improvement:.1f}x better than random")
    logger.info(f"Model saved to: trained_models/simple_128expert_model.pt")
    
    return True

if __name__ == "__main__":
    Path("trained_models").mkdir(exist_ok=True)
    
    success = train_simple_model()
    if success:
        print("\n✅ Simple 128-expert training completed!")
    else:
        print("\n❌ Training failed")