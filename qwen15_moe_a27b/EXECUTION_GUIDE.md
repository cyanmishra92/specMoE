# 🚀 Execution Guide: Step-by-Step Commands

This guide shows you **exactly what to run** in sequence for each training approach.

## 📋 Prerequisites

```bash
# Ensure you're in the right directory
cd /data/research/specMoE/specMoE/qwen15_moe_a27b/scripts

# Check if you have data (should see 10 shard files)
ls ../routing_data/shards/
```

---

## 🎯 RECOMMENDED: Hybrid Approach (Best Results)

### Step 1: Clean Environment (Optional)
```bash
# Clean previous results if needed
rm -rf ../models/hybrid_checkpoints/*
rm -rf ../results/*
```

### Step 2: Train Hybrid Model
```bash
# Train with hybrid classification + temporal prediction
python train_hybrid_predictor.py \
  --shards-per-group 2 \
  --batch-size 12 \
  --lr 3e-5 \
  --epochs 30 \
  --lookahead-steps 4
```

**Expected output:**
```
INFO - Using 2 shards per group (~8GB memory)
INFO - Hybrid Model parameters: 1,234,567 total, 1,234,567 trainable
INFO - Train - Total: 2.1234, Classification: 1.8765, Temporal: 0.2469
INFO - Current Expert Accuracy: Top-1: 0.0892 | Top-3: 0.2341 | Top-5: 0.3892
INFO - Temporal Expert Accuracy: Top-1: 0.0567 | Top-3: 0.1678 | Top-5: 0.2845
```

---

## 🔧 Alternative: Simple Approach (Stable Baseline)

### Step 1: Clean Environment (Optional)
```bash
rm -rf ../models/simple_checkpoints/*
```

### Step 2: Train Simple Model
```bash
# Train stable baseline model
python train_simple_predictor.py \
  --shards-per-group 4 \
  --batch-size 8 \
  --lr 5e-5 \
  --epochs 20
```

**Expected output:**
```
INFO - Simple Model parameters: 567,890 total, 567,890 trainable
INFO - Train loss: 2.3456
INFO - Val loss: 2.1987, Top-1: 0.0234, Top-5: 0.1456
```

---

## ⚡ Advanced: Complex Approach (Experimental)

### Step 1: Clean Environment (Optional)
```bash
rm -rf ../models/checkpoints/*
```

### Step 2: Train Complex Model
```bash
# Train full transformer model
python train_qwen_multi_expert_predictor.py \
  --shards-per-group 4 \
  --batch-size 16 \
  --lr 1e-4 \
  --epochs 50
```

---

## 📊 Model Evaluation (Any Approach)

### Evaluate Trained Model
```bash
# For Hybrid model
python evaluate_multi_expert_predictor.py \
  --checkpoint ../models/hybrid_checkpoints/best_checkpoint.pth \
  --shard-dir ../routing_data/shards

# For Simple model  
python evaluate_multi_expert_predictor.py \
  --checkpoint ../models/simple_checkpoints/best_checkpoint.pth \
  --shard-dir ../routing_data/shards

# For Complex model
python evaluate_multi_expert_predictor.py \
  --checkpoint ../models/checkpoints/best_checkpoint.pth \
  --shard-dir ../routing_data/shards
```

---

## 🎛️ Memory Configuration Guide

### RTX 3090 (24GB) - Recommended Settings

| Approach | Shards/Group | Batch Size | Memory Usage | Status |
|----------|-------------|------------|--------------|---------|
| **Simple** | 4 | 8 | ~17GB | ✅ Safe |
| **Hybrid** | 2 | 12 | ~11GB | ✅ Optimal |
| **Complex** | 4 | 16 | ~18GB | ⚠️ Tight |

### A6000 (48GB) - Aggressive Settings
```bash
# Can use larger shard groups
--shards-per-group 6    # ~24GB data
--batch-size 16         # Larger batches
```

### Low VRAM (16GB) - Conservative Settings
```bash
# Use minimal settings
--shards-per-group 1    # ~4GB data only
--batch-size 4          # Small batches
```

---

## 🚨 Troubleshooting

### Out of Memory Error
```bash
# Reduce memory usage
--shards-per-group 1
--batch-size 4

# Or use simple model
python train_simple_predictor.py --shards-per-group 1 --batch-size 4
```

### NaN Loss Issues
```bash
# Use more conservative learning rate
--lr 1e-5

# Or try simple model first
python train_simple_predictor.py --lr 5e-5
```

### Slow Training
```bash
# Increase shard groups for better GPU utilization
--shards-per-group 4    # If you have memory

# Or reduce sequence length (edit in code)
# Change max_seq_len from 256 to 128
```

---

## 📈 Expected Training Times

| Model | RTX 3090 | A6000 | Epochs | Total Time |
|-------|----------|-------|--------|------------|
| Simple | ~2 min/epoch | ~1 min/epoch | 20 | ~30-40 min |
| Hybrid | ~4 min/epoch | ~2 min/epoch | 30 | ~90-120 min |
| Complex | ~8 min/epoch | ~4 min/epoch | 50 | ~6-8 hours |

---

## 🎯 Quick Decision Matrix

**Want stable baseline?** → Use **Simple**  
**Want best results?** → Use **Hybrid** (recommended)  
**Want cutting-edge?** → Use **Complex** (experimental)  

**Limited memory?** → Use `--shards-per-group 1`  
**Have 24GB VRAM?** → Use `--shards-per-group 2-4`  
**Have 48GB VRAM?** → Use `--shards-per-group 6-8`

---

## ✅ Success Indicators

### Good Training (Hybrid):
```
Current Expert Accuracy: Top-1: 0.15+ | Top-5: 0.40+ | Top-10: 0.65+
Temporal Expert Accuracy: Top-1: 0.08+ | Top-5: 0.25+ | Top-10: 0.45+
```

### Good Training (Simple):
```
Top-1: 0.05+ (above random baseline of ~1.67%)
Top-5: 0.15+ 
No NaN losses
```

Run the **Hybrid approach** first - it's our recommended method!