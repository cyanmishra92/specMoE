#!/usr/bin/env python3
"""
Proper Train/Test Split for Speculation Model
Detect overfitting and measure true generalization performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustSpeculationDataset(Dataset):
    """Dataset with proper data splits and no data leakage"""
    
    def __init__(self, traces, max_seq_len=32, split='train'):
        self.traces = []
        self.max_seq_len = max_seq_len
        self.split = split
        
        # Group traces by sample_id to avoid data leakage
        sample_groups = {}
        for trace in traces:
            if not trace.prev_layer_gates or len(trace.prev_layer_gates) == 0:
                continue
                
            sample_id = trace.sample_id
            if sample_id not in sample_groups:
                sample_groups[sample_id] = []
            sample_groups[sample_id].append(trace)
        
        # Split by sample groups, not individual traces
        sample_ids = list(sample_groups.keys())
        train_ids, test_ids = train_test_split(
            sample_ids, 
            test_size=0.3, 
            random_state=42,
            shuffle=True
        )
        
        # Further split train into train/val
        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=0.25,  # 0.25 * 0.7 = 0.175 for val, 0.525 for train
            random_state=42,
            shuffle=True
        )
        
        # Select traces based on split
        if split == 'train':
            selected_ids = train_ids
        elif split == 'val':
            selected_ids = val_ids
        elif split == 'test':
            selected_ids = test_ids
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Process selected traces
        for sample_id in selected_ids:
            for trace in sample_groups[sample_id]:
                self._process_trace(trace)
        
        logger.info(f"Created {split} dataset with {len(self.traces)} samples from {len(selected_ids)} unique sequences")
    
    def _process_trace(self, trace):
        """Process individual trace"""
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
        
        # Get previous layer context
        prev_gate = trace.prev_layer_gates[-1]
        if prev_gate.ndim > 2:
            prev_gate = prev_gate.squeeze(0)
        prev_gate = prev_gate[:seq_len]
        
        self.traces.append({
            'hidden_states': hidden_states,
            'prev_gate': prev_gate,
            'target_routing': target_routing,
            'layer_id': trace.layer_id,
            'seq_len': seq_len,
            'sample_id': trace.sample_id,
            'dataset_name': trace.dataset_name
        })
    
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

class RegularizedSpeculationModel(nn.Module):
    """Speculation model with regularization to prevent overfitting"""
    
    def __init__(self, hidden_size, num_experts, prev_gate_size, dropout_rate=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Smaller network with more regularization
        self.hidden_proj = nn.Linear(hidden_size, 128)
        self.prev_gate_proj = nn.Linear(prev_gate_size, 128)
        
        self.combined_layer1 = nn.Linear(256, 128)
        self.combined_layer2 = nn.Linear(128, 64)
        self.output_proj = nn.Linear(64, num_experts)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)
        
    def forward(self, hidden_states, prev_gate, mask=None):
        # Project inputs
        hidden_features = torch.relu(self.hidden_proj(hidden_states))
        hidden_features = self.dropout(hidden_features)
        
        gate_features = torch.relu(self.prev_gate_proj(prev_gate))
        gate_features = self.dropout(gate_features)
        
        # Combine features
        combined = torch.cat([hidden_features, gate_features], dim=-1)
        
        # First combined layer
        combined = torch.relu(self.combined_layer1(combined))
        combined = self.layer_norm1(combined)
        combined = self.dropout(combined)
        
        # Second combined layer
        combined = torch.relu(self.combined_layer2(combined))
        combined = self.layer_norm2(combined)
        combined = self.dropout(combined)
        
        # Output logits
        logits = self.output_proj(combined)
        
        return logits

def evaluate_model(model, data_loader, device, criterion):
    """Evaluate model and return detailed metrics"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    top3_correct = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            hidden_states = batch['hidden_states'].to(device)
            prev_gate = batch['prev_gate'].to(device)
            target_routing = batch['target_routing'].to(device)
            mask = batch['mask'].to(device)
            
            logits = model(hidden_states, prev_gate, mask)
            target_indices = torch.argmax(target_routing, dim=-1)
            
            # Apply mask
            masked_logits = logits[mask]
            masked_targets = target_indices[mask]
            
            if len(masked_logits) > 0:
                loss = criterion(masked_logits, masked_targets)
                total_loss += loss.item()
                
                # Top-1 accuracy
                pred_indices = torch.argmax(masked_logits, dim=-1)
                correct += (pred_indices == masked_targets).sum().item()
                total += len(masked_targets)
                
                # Top-3 accuracy
                _, top3_preds = torch.topk(masked_logits, k=3, dim=-1)
                top3_correct += (masked_targets.unsqueeze(1) == top3_preds).any(dim=1).sum().item()
                
                # Store for confusion analysis
                predictions.extend(pred_indices.cpu().numpy())
                targets.extend(masked_targets.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    top3_accuracy = top3_correct / total if total > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'predictions': predictions,
        'targets': targets
    }

def analyze_predictions(predictions, targets, num_experts=8):
    """Analyze prediction patterns"""
    from collections import Counter
    
    pred_dist = Counter(predictions)
    target_dist = Counter(targets)
    
    logger.info("Prediction Distribution:")
    for expert in range(num_experts):
        pred_count = pred_dist.get(expert, 0)
        target_count = target_dist.get(expert, 0)
        logger.info(f"  Expert {expert}: Predicted {pred_count}, Actual {target_count}")
    
    # Check if model is just predicting most common expert
    most_common_pred = max(pred_dist, key=pred_dist.get) if pred_dist else 0
    most_common_target = max(target_dist, key=target_dist.get) if target_dist else 0
    
    pred_entropy = -sum((count/len(predictions)) * np.log2(count/len(predictions)) 
                       for count in pred_dist.values() if count > 0)
    target_entropy = -sum((count/len(targets)) * np.log2(count/len(targets)) 
                         for count in target_dist.values() if count > 0)
    
    logger.info(f"Most common prediction: Expert {most_common_pred}")
    logger.info(f"Most common target: Expert {most_common_target}")
    logger.info(f"Prediction entropy: {pred_entropy:.3f}")
    logger.info(f"Target entropy: {target_entropy:.3f}")
    
    return {
        'pred_dist': pred_dist,
        'target_dist': target_dist,
        'pred_entropy': pred_entropy,
        'target_entropy': target_entropy
    }

def train_with_proper_validation(traces, device):
    """Train with proper train/val/test splits"""
    
    logger.info("🧠 Training with Proper Validation")
    
    # Create datasets with proper splits
    train_dataset = RobustSpeculationDataset(traces, split='train')
    val_dataset = RobustSpeculationDataset(traces, split='val')
    test_dataset = RobustSpeculationDataset(traces, split='test')
    
    if len(train_dataset) == 0:
        logger.error("No training data!")
        return None
    
    # Get data dimensions
    sample = train_dataset[0]
    hidden_size = sample['hidden_states'].size(-1)
    num_experts = sample['target_routing'].size(-1)
    prev_gate_size = sample['prev_gate'].size(-1)
    
    logger.info(f"Model config: hidden_size={hidden_size}, num_experts={num_experts}, prev_gate_size={prev_gate_size}")
    logger.info(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model with regularization
    model = RegularizedSpeculationModel(hidden_size, num_experts, prev_gate_size, dropout_rate=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)  # Lower LR, weight decay
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    logger.info(f"Model on device: {next(model.parameters()).device}")
    
    # Training loop with early stopping
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    num_epochs = 30
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_metrics = {'loss': 0.0, 'correct': 0, 'total': 0}
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            hidden_states = batch['hidden_states'].to(device)
            prev_gate = batch['prev_gate'].to(device)
            target_routing = batch['target_routing'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            logits = model(hidden_states, prev_gate, mask)
            target_indices = torch.argmax(target_routing, dim=-1)
            
            # Apply mask
            masked_logits = logits[mask]
            masked_targets = target_indices[mask]
            
            if len(masked_logits) > 0:
                loss = criterion(masked_logits, masked_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_metrics['loss'] += loss.item()
                pred_indices = torch.argmax(masked_logits, dim=-1)
                train_metrics['correct'] += (pred_indices == masked_targets).sum().item()
                train_metrics['total'] += len(masked_targets)
        
        # Validation
        val_results = evaluate_model(model, val_loader, device, criterion)
        
        # Calculate metrics
        train_loss = train_metrics['loss'] / len(train_loader)
        train_acc = train_metrics['correct'] / train_metrics['total'] if train_metrics['total'] > 0 else 0
        
        train_losses.append(train_loss)
        val_losses.append(val_results['loss'])
        train_accuracies.append(train_acc)
        val_accuracies.append(val_results['accuracy'])
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_results['loss']:.4f}")
        logger.info(f"          Train Acc={train_acc:.3f} ({train_acc*100:.1f}%), Val Acc={val_results['accuracy']:.3f} ({val_results['accuracy']*100:.1f}%)")
        logger.info(f"          Val Top-3 Acc={val_results['top3_accuracy']:.3f} ({val_results['top3_accuracy']*100:.1f}%)")
        
        # Learning rate scheduling
        scheduler.step(val_results['loss'])
        
        # Early stopping
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'config': {
                    'hidden_size': hidden_size,
                    'num_experts': num_experts,
                    'prev_gate_size': prev_gate_size
                }
            }, 'trained_models/robust_speculation_model.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final test evaluation
    logger.info("\n🔍 Final Test Set Evaluation:")
    test_results = evaluate_model(model, test_loader, device, criterion)
    
    logger.info(f"Test Loss: {test_results['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_results['accuracy']:.3f} ({test_results['accuracy']*100:.1f}%)")
    logger.info(f"Test Top-3 Accuracy: {test_results['top3_accuracy']:.3f} ({test_results['top3_accuracy']*100:.1f}%)")
    
    # Analyze predictions
    logger.info("\n📊 Test Set Prediction Analysis:")
    pred_analysis = analyze_predictions(test_results['predictions'], test_results['targets'])
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_results': test_results,
        'pred_analysis': pred_analysis
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
    """Main training function with proper validation"""
    
    logger.info("🚀 Robust Speculation Model Training")
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
    
    # Train model with proper validation
    start_time = time.time()
    results = train_with_proper_validation(traces, device)
    training_time = time.time() - start_time
    
    if results:
        logger.info(f"\n✅ Training completed successfully!")
        logger.info(f"Training time: {training_time:.1f} seconds")
        
        test_acc = results['test_results']['accuracy']
        test_top3 = results['test_results']['top3_accuracy']
        
        logger.info(f"🎯 FINAL TEST RESULTS:")
        logger.info(f"   Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        logger.info(f"   Test Top-3 Accuracy: {test_top3:.3f} ({test_top3*100:.1f}%)")
        
        # Calculate improvement over random baseline
        baseline = 1.0 / 8  # Random guess for 8 experts = 12.5%
        improvement = (test_acc - baseline) / baseline * 100
        logger.info(f"📈 Improvement over random baseline: {improvement:.1f}% relative gain")
        
        # Check for overfitting
        final_train_acc = results['train_accuracies'][-1]
        final_val_acc = results['val_accuracies'][-1]
        overfitting_gap = final_train_acc - final_val_acc
        
        logger.info(f"\n🔍 OVERFITTING ANALYSIS:")
        logger.info(f"   Final Train Accuracy: {final_train_acc:.3f} ({final_train_acc*100:.1f}%)")
        logger.info(f"   Final Val Accuracy: {final_val_acc:.3f} ({final_val_acc*100:.1f}%)")
        logger.info(f"   Train-Val Gap: {overfitting_gap:.3f} ({overfitting_gap*100:.1f}%)")
        
        if overfitting_gap > 0.1:
            logger.warning("⚠️  Model shows signs of overfitting!")
        else:
            logger.info("✅ Model generalizes well!")
        
        logger.info(f"💾 Best model saved to: trained_models/robust_speculation_model.pt")
        return True
    else:
        logger.error("Training failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Robust speculation model training completed!")
        print("✅ Proper train/test splits with overfitting detection")
    else:
        print("\n❌ Training failed")