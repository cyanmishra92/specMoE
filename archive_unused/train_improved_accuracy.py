#!/usr/bin/env python3
"""
Improved Training for Higher Accuracy
Multiple strategies to boost speculation accuracy
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedPredictor(nn.Module):
    """Improved model with multiple enhancements"""
    
    def __init__(self, hidden_size=512, num_experts=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Multi-scale feature extraction
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU()
            )
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Expert prediction head
        self.expert_head = nn.Linear(256, num_experts)
        
        # Top-k prediction head (predict multiple experts)
        self.topk_head = nn.Linear(256, num_experts)
        
    def forward(self, hidden_states):
        if hidden_states.dim() == 3:
            # Use both mean and max pooling
            mean_pooled = torch.mean(hidden_states, dim=1)
            max_pooled = torch.max(hidden_states, dim=1)[0]
            pooled = (mean_pooled + max_pooled) / 2
        else:
            pooled = hidden_states
        
        # Multi-scale features
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(pooled))
        
        # Fuse features
        fused = self.fusion(torch.cat(features, dim=-1))
        
        # Predictions
        expert_logits = self.expert_head(fused)
        topk_logits = self.topk_head(fused)
        
        return expert_logits, topk_logits

class ImprovedDataset(Dataset):
    """Improved dataset with better target extraction"""
    
    def __init__(self, traces, strategy='weighted'):
        self.traces = traces
        self.strategy = strategy
        self.valid_samples = []
        
        for trace in traces:
            if (hasattr(trace, 'hidden_states') and 
                hasattr(trace, 'target_routing') and
                trace.hidden_states is not None and 
                trace.target_routing is not None):
                
                # Extract better targets
                targets = self._extract_targets(trace)
                if targets is not None:
                    self.valid_samples.append({
                        'hidden_states': trace.hidden_states.float(),
                        'targets': targets
                    })
        
        logger.info(f"Dataset ({strategy}): {len(self.valid_samples)} valid samples")
    
    def _extract_targets(self, trace):
        """Extract targets using different strategies"""
        routing = trace.target_routing.float()  # [seq, experts]
        
        if self.strategy == 'weighted':
            # Weighted average of expert probabilities
            weights = torch.sum(routing, dim=0)  # [experts]
            top_expert = torch.argmax(weights).item()
            
            # Also get top-3 experts
            top3_experts = torch.topk(weights, k=min(3, len(weights))).indices
            
            return {
                'primary': top_expert,
                'top3': top3_experts,
                'weights': weights / weights.sum()
            }
            
        elif self.strategy == 'sequence_aware':
            # Consider sequence patterns
            expert_per_token = torch.argmax(routing, dim=-1)  # [seq]
            
            # Find most stable expert (appears in consecutive positions)
            seq_len = len(expert_per_token)
            stability_scores = torch.zeros(128)
            
            for i in range(seq_len - 1):
                current_expert = expert_per_token[i]
                next_expert = expert_per_token[i + 1]
                if current_expert == next_expert:
                    stability_scores[current_expert] += 1
            
            if stability_scores.sum() > 0:
                stable_expert = torch.argmax(stability_scores).item()
            else:
                stable_expert = torch.mode(expert_per_token).values.item()
            
            return {
                'primary': stable_expert,
                'stability': stability_scores
            }
            
        else:  # 'frequent'
            expert_per_token = torch.argmax(routing, dim=-1)
            most_frequent = torch.mode(expert_per_token).values.item()
            return {'primary': most_frequent}
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        return self.valid_samples[idx]

def improved_collate(batch):
    """Improved collate with better target handling"""
    hidden_states = []
    primary_targets = []
    top3_targets = []
    weights = []
    
    for item in batch:
        # Pool sequence dimension
        h = item['hidden_states']
        if h.dim() == 2:  # [seq, hidden]
            h_mean = torch.mean(h, dim=0)
            h_max = torch.max(h, dim=0)[0]
            h_pooled = (h_mean + h_max) / 2
        else:
            h_pooled = h
        
        hidden_states.append(h_pooled)
        primary_targets.append(item['targets']['primary'])
        
        # Handle top3 if available
        if 'top3' in item['targets']:
            top3 = item['targets']['top3']
            # Pad to length 3
            if len(top3) < 3:
                top3 = torch.cat([top3, torch.zeros(3 - len(top3), dtype=torch.long)])
            top3_targets.append(top3[:3])
        
        # Handle weights if available
        if 'weights' in item['targets']:
            weights.append(item['targets']['weights'])
    
    result = {
        'hidden_states': torch.stack(hidden_states),
        'primary_targets': torch.tensor(primary_targets, dtype=torch.long)
    }
    
    if top3_targets:
        result['top3_targets'] = torch.stack(top3_targets)
    if weights:
        result['weights'] = torch.stack(weights)
    
    return result

def train_improved_model(strategy='weighted'):
    """Train improved model with better accuracy"""
    
    logger.info(f"🧠 Training Improved Model (strategy: {strategy})")
    logger.info("=" * 50)
    
    # Load data
    trace_file = "routing_data/benchmark_traces.pkl"
    with open(trace_file, 'rb') as f:
        serializable_traces = pickle.load(f)
    
    # Convert to objects
    traces = []
    for trace_dict in serializable_traces:
        class SimpleTrace:
            pass
        trace = SimpleTrace()
        trace.hidden_states = torch.from_numpy(trace_dict['hidden_states'])
        trace.target_routing = torch.from_numpy(trace_dict['target_routing'])
        traces.append(trace)
    
    # Split data
    split_idx = int(0.8 * len(traces))
    train_traces = traces[:split_idx]
    val_traces = traces[split_idx:]
    
    # Create datasets
    train_dataset = ImprovedDataset(train_traces, strategy)
    val_dataset = ImprovedDataset(val_traces, strategy)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=improved_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=improved_collate)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedPredictor().to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Loss function with label smoothing
    def improved_loss(logits, targets, weights=None):
        # Primary loss with label smoothing
        primary_loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
        
        # Add weight-based loss if available
        if weights is not None:
            weight_loss = F.kl_div(F.log_softmax(logits, dim=1), weights, reduction='batchmean')
            return primary_loss + 0.1 * weight_loss
        
        return primary_loss
    
    logger.info(f"🎯 Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU: {torch.cuda.get_device_name()}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    num_epochs = 25  # More epochs
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            hidden_states = batch['hidden_states'].to(device)
            targets = batch['primary_targets'].to(device)
            weights = batch.get('weights', None)
            if weights is not None:
                weights = weights.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            expert_logits, topk_logits = model(hidden_states)
            
            # Calculate loss
            loss = improved_loss(expert_logits, targets, weights)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Stats
            train_loss += loss.item()
            pred = torch.argmax(expert_logits, dim=1)
            train_correct += (pred == targets).sum().item()
            train_total += targets.size(0)
        
        scheduler.step()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_top3_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                hidden_states = batch['hidden_states'].to(device)
                targets = batch['primary_targets'].to(device)
                weights = batch.get('weights', None)
                if weights is not None:
                    weights = weights.to(device)
                
                expert_logits, topk_logits = model(hidden_states)
                loss = improved_loss(expert_logits, targets, weights)
                
                val_loss += loss.item()
                
                # Top-1 accuracy
                pred = torch.argmax(expert_logits, dim=1)
                val_correct += (pred == targets).sum().item()
                
                # Top-3 accuracy
                top3_pred = torch.topk(expert_logits, k=3, dim=1).indices
                val_top3_correct += (targets.unsqueeze(1) == top3_pred).any(dim=1).sum().item()
                
                val_total += targets.size(0)
        
        val_acc = val_correct / val_total
        val_top3_acc = val_top3_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.3f}")
        logger.info(f"           Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.3f}, Val Top-3: {val_top3_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'val_top3_accuracy': val_top3_acc,
                'strategy': strategy,
                'epoch': epoch
            }, f"trained_models/improved_128expert_{strategy}.pt")
    
    # Final results
    random_baseline = 1.0 / 128
    improvement = best_val_acc / random_baseline
    
    logger.info(f"\n🎉 Improved Training Results ({strategy}):")
    logger.info(f"Best validation accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    logger.info(f"Random baseline: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
    logger.info(f"Improvement: {improvement:.1f}x better than random")
    logger.info(f"Model saved to: trained_models/improved_128expert_{strategy}.pt")
    
    return best_val_acc

def main():
    """Test multiple strategies"""
    Path("trained_models").mkdir(exist_ok=True)
    
    strategies = ['weighted', 'sequence_aware', 'frequent']
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING STRATEGY: {strategy.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            accuracy = train_improved_model(strategy)
            results[strategy] = accuracy
        except Exception as e:
            logger.error(f"Strategy {strategy} failed: {e}")
            results[strategy] = 0
    
    # Summary
    logger.info(f"\n🏆 STRATEGY COMPARISON:")
    logger.info("-" * 30)
    
    best_strategy = max(results.keys(), key=lambda x: results[x])
    for strategy, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🥇" if strategy == best_strategy else "  "
        logger.info(f"{marker} {strategy:15}: {acc:.3f} ({acc*100:.1f}%)")
    
    return True

if __name__ == "__main__":
    main()