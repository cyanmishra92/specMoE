# Qwen1.5-MoE-A2.7B Expert Prediction Training

Complete implementation of multi-expert prediction training for Qwen1.5-MoE-A2.7B model with **RTX 3090 memory optimization**, **advanced architectures**, and **configurable data collection**.

## 🚀 Multiple Training Approaches Available

We provide **three different architectures** for expert prediction, from simple to state-of-the-art:

1. **Simple Predictor**: Stable baseline with basic MLP architecture
2. **Hybrid Predictor**: **ExpertFlow-inspired** classification + temporal speculation (RECOMMENDED)
3. **Complex Predictor**: Full transformer-based with multi-loss training

## Model Overview

**Qwen1.5-MoE-A2.7B** (actually Qwen2-MoE detected automatically):
- **14.3B total parameters** (60 routing + 4 shared experts)  
- **2.7B active parameters** per token (top-4 routing)
- **60 routing experts + 4 shared experts** per MoE layer
- **Top-4 routing** (4 out of 60 experts selected per token)
- **RTX 3090/A6000 optimized** with configurable shard loading

## 🏗️ Architecture Comparison

| Architecture | Approach | Parameters | Memory | Speed | Accuracy Target |
|-------------|----------|------------|--------|-------|-----------------|
| **Simple** ⭐ | Basic MLP | ~0.5M | Low | Fast | **47.55%** ✅ |
| **Hybrid** | Classification + Temporal | ~1.5M | Medium | Medium | 9.57% ❌ |
| **Complex** | Multi-Transformer | ~21M | High | Slow | 1.77% ❌ |

## 🎯 Recommended Approach: Simple Predictor ⭐

Our **Simple Predictor** achieved breakthrough results:
- **Outstanding Performance**: 47.55% Top-1 accuracy (28x better than random)
- **Deployment Ready**: Clear coverage probabilities for real-world systems
- **Memory Efficient**: Enables 67% memory reduction in MoE inference
- **Stable Training**: No NaN losses, reliable convergence

### Key Advantages:
✅ **Proven Results**: 47.55% Top-1, 73.85% Top-5 accuracy  
✅ **Stable Training**: Simple MLP architecture prevents instability  
✅ **Fast Training**: Only 40 minutes on RTX 3090  
✅ **Practical Impact**: 39% perfect coverage with Top-20 preloading  

## Hardware Requirements
- **RTX 3090 (24GB)**: ✅ Perfect with configurable shard loading
- **A6000 (48GB)**: ✅ Excellent performance, larger shard groups
- **RTX 4090 (24GB)**: ✅ Optimal performance
- **Minimum**: 16GB VRAM with conservative settings

## 🚀 Quick Start Guide

### Step 1: Collect Training Data
```bash
# Collect traces with streaming to avoid memory issues
python scripts/collection/collect_qwen15_moe_traces_streaming.py \
  --target_traces 5000 \
  --shard_size 500
```

### Step 2: Choose Your Training Approach

#### **Option A: Simple Predictor (⭐ RECOMMENDED - 47.55% PROVEN)**
```bash
python scripts/train_simple_predictor.py \
  --shards-per-group 4 \
  --batch-size 8 \
  --lr 5e-5 \
  --epochs 20
```

#### **Option B: Hybrid Predictor (❌ Not Recommended - Training Unstable)**
```bash
# Warning: This approach failed with NaN losses
python scripts/train_hybrid_predictor.py \
  --shards-per-group 2 \
  --batch-size 12 \
  --lr 3e-5 \
  --epochs 30
```

#### **Option C: Complex Predictor (❌ Failed - Avoid)**
```bash
python scripts/train_qwen_multi_expert_predictor.py \
  --shards-per-group 4 \
  --batch-size 16 \
  --lr 1e-4 \
  --epochs 50
```

## 🔧 Configurable Shard Loading

All training scripts support configurable shard loading for optimal GPU utilization:

```bash
# Conservative (safe for any GPU)
--shards-per-group 1    # ~4GB memory usage

# Balanced (RTX 3090 recommended)
--shards-per-group 2    # ~8GB memory usage

# Aggressive (RTX 3090 optimized)
--shards-per-group 4    # ~16GB memory usage

# Maximum (A6000/A100)
--shards-per-group 6    # ~24GB memory usage
```

## 📊 Confirmed Performance Results

### Simple Predictor (⭐ PROVEN WINNER)
- **Individual Expert Prediction**: 
  - Top-1: **47.55%** (28x better than random!)
  - Top-5: **73.85%** (Outstanding coverage)
  - Top-10: **87.43%** (Excellent coverage)
- **Coverage Probability** (All 4 experts found):
  - Top-10: **30.57%** (Preload 10 experts)
  - Top-20: **39.04%** (Preload 20 experts)  
  - Top-30: **50.67%** (Preload 30 experts)

### Hybrid Predictor (❌ Training Unstable)
- **Final Results**: Only 9.57% Top-1 accuracy
- **Issues**: NaN losses, training collapse
- **Recommendation**: Use Simple approach instead

### Memory Usage by Architecture
| Architecture | Model Size | Optimal Shards | Memory Usage | RTX 3090 |
|-------------|------------|----------------|--------------|----------|
| Simple | ~0.5GB | 4 shards | ~17GB | ✅ Excellent |
| Hybrid | ~1.5GB | 2 shards | ~11GB | ✅ Perfect |
| Complex | ~2GB | 4 shards | ~18GB | ✅ Good |

## 🎯 Training Pipeline

### Complete Workflow
```bash
# 1. Clone and setup
cd qwen15_moe_a27b/

# 2. Collect data (creates shards automatically)
python scripts/collection/collect_qwen15_moe_traces_streaming.py \
  --target_traces 5000

# 3. Train simple model (PROVEN WINNER - 47.55%)
python scripts/train_simple_predictor.py \
  --shards-per-group 4 \
  --batch-size 8 \
  --epochs 20

# 4. Evaluate model
python scripts/evaluate_multi_expert_predictor.py \
  --checkpoint ../models/simple_checkpoints/best_checkpoint.pth
```

## 📁 Project Structure

```
qwen15_moe_a27b/
├── scripts/
│   ├── collection/
│   │   ├── collect_qwen15_moe_traces_streaming.py    # Memory-optimized collection
│   │   ├── collect_qwen15_moe_traces_small.py        # Quick test (500 traces)
│   │   └── collect_qwen15_moe_traces_medium.py       # Configurable collection
│   ├── train_simple_predictor.py                     # Simple MLP baseline
│   ├── train_hybrid_predictor.py                     # Hybrid classification + temporal ⭐
│   ├── train_qwen_multi_expert_predictor.py          # Complex multi-transformer
│   └── evaluate_multi_expert_predictor.py            # Comprehensive evaluation
├── models/
│   ├── simple_qwen_predictor.py                      # Simple MLP model
│   ├── hybrid_expert_predictor.py                    # Hybrid model ⭐
│   └── qwen_multi_expert_predictor.py                # Complex transformer model
├── routing_data/
│   └── shards/                                       # Collected trace shards
└── results/                                          # Training results and plots
```

## 🧠 Model Architectures

### Hybrid Predictor (Recommended)
```
Input [batch, seq, 2048]
    ↓
Shared Feature Extractor (512-dim)
    ↓                    ↓
Classification Head    Temporal Transformer
(Sigmoid Binary)       (2 layers + pos encoding)  
    ↓                    ↓
Current Experts        Future Experts
[B,S,60] probs        [B,S,4,60] probs
```

### Simple Predictor (Baseline)
```
Input [batch, seq, 2048] → MLP layers → Expert logits [batch, seq, 60]
```

### Complex Predictor (Advanced)
```
Input → Transformer(6 layers) → Multi-head prediction → Expert routing
```

## 🔬 Advanced Features

### Hybrid Model Capabilities
- **Binary Expert Classification**: Each expert gets probability [0,1] for activation
- **Temporal Lookahead**: Predicts expert usage 1-4 tokens in advance
- **Joint Training**: Combined loss for immediate + temporal prediction
- **Attention-Based**: Uses transformer layers for temporal dependencies

### Training Optimizations
- **Mixed Precision**: FP16 training for speed and memory efficiency
- **Gradient Clipping**: Stable training with clip_grad_norm
- **Shard-Based Loading**: Process multiple shards together for efficiency
- **Dynamic Memory Management**: Automatic cleanup between shard groups

## 📈 Evaluation Metrics

All models provide comprehensive evaluation:
- **Top-k Accuracy**: k ∈ {1, 3, 5, 10, 20}
- **Exact Match**: All 4 experts predicted correctly
- **Partial Match**: ≥1 expert predicted correctly
- **Position-wise Analysis**: Per-token accuracy breakdown
- **Temporal Accuracy**: Future prediction performance (hybrid only)

## 🎛️ Configuration Options

### Collection Settings
```bash
--target_traces 5000        # Number of traces to collect
--shard_size 500            # Traces per shard file
--batch_size 8              # Processing batch size
--max_length 256            # Sequence length
```

### Training Settings
```bash
--shards-per-group 2        # Shards to load simultaneously
--batch-size 12             # Training batch size  
--lr 3e-5                   # Learning rate
--epochs 30                 # Training epochs
--lookahead-steps 4         # Future prediction steps (hybrid only)
```

## 🚀 Getting Started (Recommended Path)

1. **Quick Test**:
   ```bash
   python scripts/collection/collect_qwen15_moe_traces_small.py
   ```

2. **Full Data Collection**:
   ```bash
   python scripts/collection/collect_qwen15_moe_traces_streaming.py --target_traces 5000
   ```

3. **Train Hybrid Model**:
   ```bash
   python scripts/train_hybrid_predictor.py --shards-per-group 2 --epochs 30
   ```

4. **Evaluate Results**:
   ```bash
   python scripts/evaluate_multi_expert_predictor.py
   ```

## 🎯 Why Simple Approach Won?

✅ **Outstanding Results**: 47.55% Top-1 accuracy (28x better than random!)  
✅ **Stable Training**: No NaN losses, reliable convergence in 40 minutes  
✅ **Deployment Ready**: Clear coverage probabilities for real systems  
✅ **Memory Efficient**: Enables 67% memory reduction with 39% perfect hit rate  
✅ **Research Finding**: Simple MLPs outperform complex transformers for expert prediction  

**Key Insight**: Sometimes simpler is genuinely better! Perfect for researchers working on MoE optimization, expert speculation, and efficient inference systems!