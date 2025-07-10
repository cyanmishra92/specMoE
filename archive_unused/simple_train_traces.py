#!/usr/bin/env python3
"""
Simple Speculation Model Training on Real Traces
Fixed working version with proper GPU training and accuracy reporting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSpeculationDataset(Dataset):
    """Simple dataset for speculation training"""
    
    def __init__(self, traces, max_seq_len=32):
        self.traces = []
        self.max_seq_len = max_seq_len
        
        # Process traces into simple format
        for trace in traces:
            # Skip if no previous layer context
            if not trace.prev_layer_gates or len(trace.prev_layer_gates) == 0:
                continue
                
            # Get dimensions
            hidden_states = trace.hidden_states
            target_routing = trace.target_routing
            
            # Ensure 2D tensors
            if hidden_states.ndim > 2:
                hidden_states = hidden_states.squeeze(0)
            if target_routing.ndim > 2:
                target_routing = target_routing.squeeze(0)
            
            # Truncate to max sequence length
            seq_len = min(hidden_states.size(0), self.max_seq_len)
            hidden_states = hidden_states[:seq_len]
            target_routing = target_routing[:seq_len]
            
            # Get previous layer context (use last one if available)
            prev_gate = trace.prev_layer_gates[-1]
            if prev_gate.ndim > 2:
                prev_gate = prev_gate.squeeze(0)
            prev_gate = prev_gate[:seq_len]
            
            self.traces.append({
                'hidden_states': hidden_states,
                'prev_gate': prev_gate,
                'target_routing': target_routing,
                'layer_id': trace.layer_id,
                'seq_len': seq_len
            })
        
        logger.info(f"Created dataset with {len(self.traces)} valid samples")
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        
        # Pad to max_seq_len
        hidden_states = torch.zeros(self.max_seq_len, trace['hidden_states'].size(-1))
        prev_gate = torch.zeros(self.max_seq_len, trace['prev_gate'].size(-1))
        target_routing = torch.zeros(self.max_seq_len, trace['target_routing'].size(-1))
        
        seq_len = trace['seq_len']
        hidden_states[:seq_len] = trace['hidden_states']
        prev_gate[:seq_len] = trace['prev_gate']
        target_routing[:seq_len] = trace['target_routing']
        
        # Create mask
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:seq_len] = True
        
        return {
            'hidden_states': hidden_states,
            'prev_gate': prev_gate,
            'target_routing': target_routing,
            'mask': mask,
            'layer_id': trace['layer_id']
        }

class SimpleSpeculationModel(nn.Module):
    """Simple speculation model that works reliably"""
    
    def __init__(self, hidden_size, num_experts, prev_gate_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Simple feedforward network
        self.hidden_proj = nn.Linear(hidden_size, 256)
        self.prev_gate_proj = nn.Linear(prev_gate_size, 256)
        self.combined_proj = nn.Linear(512, 256)
        self.output_proj = nn.Linear(256, num_experts)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states, prev_gate, mask=None):
        # Project inputs
        hidden_features = torch.relu(self.hidden_proj(hidden_states))
        gate_features = torch.relu(self.prev_gate_proj(prev_gate))
        
        # Combine features
        combined = torch.cat([hidden_features, gate_features], dim=-1)
        combined = torch.relu(self.combined_proj(combined))
        combined = self.dropout(combined)
        
        # Output logits
        logits = self.output_proj(combined)
        
        return logits

def train_simple_model(traces, device):
    """Train a simple working speculation model"""
    
    logger.info("🧠 Training Simple Speculation Model")
    
    # Create dataset
    dataset = SimpleSpeculationDataset(traces)
    if len(dataset) == 0:
        logger.error("No valid training data!")
        return None
    
    # Get data dimensions
    sample = dataset[0]
    hidden_size = sample['hidden_states'].size(-1)
    num_experts = sample['target_routing'].size(-1)
    prev_gate_size = sample['prev_gate'].size(-1)
    
    logger.info(f"Model config: hidden_size={hidden_size}, num_experts={num_experts}, prev_gate_size={prev_gate_size}")
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model
    model = SimpleSpeculationModel(hidden_size, num_experts, prev_gate_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    logger.info(f"Model on device: {next(model.parameters()).device}")
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    num_epochs = 15
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move to device
            hidden_states = batch['hidden_states'].to(device)
            prev_gate = batch['prev_gate'].to(device)
            target_routing = batch['target_routing'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            logits = model(hidden_states, prev_gate, mask)
            
            # Get target indices (top-1 expert for each position)
            target_indices = torch.argmax(target_routing, dim=-1)
            
            # Apply mask to ignore padded positions
            masked_logits = logits[mask]
            masked_targets = target_indices[mask]
            
            if len(masked_logits) > 0:
                loss = criterion(masked_logits, masked_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                pred_indices = torch.argmax(masked_logits, dim=-1)
                train_correct += (pred_indices == masked_targets).sum().item()
                train_total += len(masked_targets)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                hidden_states = batch['hidden_states'].to(device)
                prev_gate = batch['prev_gate'].to(device)
                target_routing = batch['target_routing'].to(device)
                mask = batch['mask'].to(device)
                
                logits = model(hidden_states, prev_gate, mask)
                target_indices = torch.argmax(target_routing, dim=-1)
                
                masked_logits = logits[mask]
                masked_targets = target_indices[mask]
                
                if len(masked_logits) > 0:
                    loss = criterion(masked_logits, masked_targets)
                    val_loss += loss.item()
                    
                    pred_indices = torch.argmax(masked_logits, dim=-1)
                    val_correct += (pred_indices == masked_targets).sum().item()
                    val_total += len(masked_targets)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        logger.info(f"          Train Acc={train_accuracy:.3f} ({train_accuracy*100:.1f}%), Val Acc={val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': {
                    'hidden_size': hidden_size,
                    'num_experts': num_experts,
                    'prev_gate_size': prev_gate_size
                }
            }, 'trained_models/simple_speculation_model.pt')
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_accuracy': train_accuracy,
        'final_val_accuracy': val_accuracy
    }

def load_traces(trace_file="routing_data/proper_traces.pkl"):
    """Load traces from file"""
    
    logger.info(f"Loading traces from {trace_file}")
    
    if not Path(trace_file).exists():
        logger.error(f"Trace file not found: {trace_file}")
        return None
    
    with open(trace_file, 'rb') as f:
        serializable_traces = pickle.load(f)
    
    # Convert back to objects
    from training.gating_data_collector import GatingDataPoint
    
    traces = []
    for trace_dict in serializable_traces:
        trace = GatingDataPoint(
            layer_id=trace_dict['layer_id'],
            hidden_states=torch.from_numpy(trace_dict['hidden_states']),
            input_embeddings=torch.from_numpy(trace_dict['input_embeddings']),
            target_routing=torch.from_numpy(trace_dict['target_routing']),
            target_top_k=torch.from_numpy(trace_dict['target_top_k']),
            prev_layer_gates=[torch.from_numpy(g) for g in trace_dict['prev_layer_gates']],
            sequence_length=trace_dict['sequence_length'],
            token_ids=torch.from_numpy(trace_dict['token_ids']) if trace_dict['token_ids'] is not None else None,
            dataset_name=trace_dict['dataset_name'],
            sample_id=trace_dict['sample_id']
        )
        traces.append(trace)
    
    logger.info(f"✅ Loaded {len(traces)} traces")
    return traces

def main():
    """Main training function"""
    
    logger.info("🚀 Simple Speculation Model Training")
    logger.info("=" * 50)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load traces
    traces = load_traces()
    if not traces:
        logger.error("Failed to load traces")
        return False
    
    # Train model
    start_time = time.time()
    results = train_simple_model(traces, device)
    training_time = time.time() - start_time
    
    if results:
        logger.info(f"\n✅ Training completed successfully!")
        logger.info(f"Training time: {training_time:.1f} seconds")
        logger.info(f"Final train accuracy: {results['final_train_accuracy']:.3f} ({results['final_train_accuracy']*100:.1f}%)")
        logger.info(f"Final validation accuracy: {results['final_val_accuracy']:.3f} ({results['final_val_accuracy']*100:.1f}%)")
        
        # Calculate improvement over random baseline
        baseline = 1.0 / 8  # Random guess for 8 experts = 12.5%
        improvement = (results['final_val_accuracy'] - baseline) / baseline * 100
        logger.info(f"📈 Improvement over random baseline: {improvement:.1f}% relative gain")
        
        # Loss reduction
        if len(results['train_losses']) > 1:
            initial_loss = results['train_losses'][0]
            final_loss = results['train_losses'][-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100
            logger.info(f"📉 Training loss reduction: {loss_reduction:.1f}%")
        
        logger.info(f"💾 Model saved to: trained_models/simple_speculation_model.pt")
        return True
    else:
        logger.error("Training failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Speculation model training completed!")
        print("✅ GPU training successful with accuracy reporting")
    else:
        print("\n❌ Training failed")