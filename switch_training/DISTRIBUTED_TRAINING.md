# Distributed Switch Transformer Training

This document covers distributed training setup for Switch Transformers using multiple GPUs.

## 🚀 Quick Start

### Simple Launcher (Recommended)
```bash
# Check GPU status
python launch_training.py --status

# Train with 2 GPUs (auto-selects best GPUs)
python launch_training.py --experts 128 --gpus 2

# Train with 4 GPUs
python launch_training.py --experts 256 --gpus 4 --num_epochs 5
```

### Direct Distributed Training
```bash
# Train with 4 GPUs
python train_switch_distributed.py --experts 128 --gpus 4

# Train with 2 GPUs, custom settings
python train_switch_distributed.py --experts 256 --gpus 2 \
  --batch_size 4 --learning_rate 4e-6 --num_epochs 5
```

## 🎯 Training Options

### 1. Launcher Script (Easiest)
**File**: `launch_training.py`

**Features**:
- Automatic GPU detection and selection
- Data availability checking
- GPU status display
- Confirmation prompts
- Fallback to single GPU if needed

**Usage**:
```bash
# Basic usage
python launch_training.py --experts 128 --gpus 2

# Advanced usage
python launch_training.py --experts 256 --gpus 4 \
  --batch_size 3 --learning_rate 3e-6 --num_epochs 5 \
  --disable_wandb --yes
```

### 2. Direct Distributed Training
**File**: `train_switch_distributed.py`

**Features**:
- Intelligent GPU selection based on memory and utilization
- Automatic fallback to fewer GPUs if requested count unavailable
- Linear learning rate scaling for multi-GPU
- Distributed data loading with DistributedSampler
- Proper synchronization and cleanup

**Usage**:
```bash
# Auto-select 4 best GPUs
python train_switch_distributed.py --experts 128 --gpus 4

# Force single GPU training
python train_switch_distributed.py --experts 128 --single_gpu
```

### 3. Single GPU Training (Fallback)
**File**: `train_switch_unified.py`

Automatically used when:
- Only 1 GPU requested
- Only 1 GPU available
- `--single_gpu` flag is used
- `--force-single` flag is used in launcher

## 📊 Multi-GPU Optimizations

### Automatic Scaling
The distributed training automatically scales:

| Parameter | Single GPU | 2 GPUs | 4 GPUs |
|-----------|------------|--------|--------|
| Learning Rate | 3e-6 | 6e-6 | 12e-6 |
| Effective Batch Size | 3 | 6 | 12 |
| Per-device Batch Size | 3 | 3 | 3 |

### Expert-Specific Configurations
Each expert count has optimized distributed settings:

| Experts | Per-Device Batch | Base LR | Grad Accum | Warmup |
|---------|------------------|---------|------------|--------|
| 8       | 8                | 6e-6    | 1          | 0.1    |
| 16      | 6                | 5e-6    | 1          | 0.1    |
| 32      | 4                | 4e-6    | 2          | 0.12   |
| 64      | 4                | 4e-6    | 2          | 0.12   |
| 128     | 3                | 3e-6    | 3          | 0.15   |
| 256     | 2                | 2e-6    | 4          | 0.15   |

## 🔧 GPU Selection Logic

### Automatic Selection
1. **Memory Check**: Requires ≥16GB free memory
2. **Utilization Check**: Selects GPUs with <90% utilization
3. **Memory Ranking**: Prioritizes GPUs with most free memory
4. **Fallback**: Prompts user if fewer suitable GPUs available

### Manual Override
```bash
# Force specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_switch_distributed.py --experts 128 --gpus 4
```

## 📈 Performance Expectations

### Training Speed (30K samples)
| Model | 1 GPU (A6000) | 2 GPUs | 4 GPUs |
|-------|---------------|--------|--------|
| Switch-32 | 90 min | 50 min | 30 min |
| Switch-128 | 180 min | 95 min | 55 min |
| Switch-256 | 240 min | 130 min | 75 min |

### Memory Usage
| Model | 1 GPU | 2 GPUs | 4 GPUs |
|-------|-------|--------|--------|
| Switch-32 | 14GB | 14GB/GPU | 14GB/GPU |
| Switch-128 | 32GB | 32GB/GPU | 32GB/GPU |
| Switch-256 | 48GB | 48GB/GPU | 48GB/GPU |

## 🛠️ Advanced Usage

### Custom GPU Configuration
```bash
# Check available GPUs
python launch_training.py --status

# Train with specific settings
python train_switch_distributed.py --experts 128 --gpus 2 \
  --batch_size 4 \
  --learning_rate 4e-6 \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.1 \
  --num_epochs 5 \
  --output_dir ../models/custom_switch_128
```

### Environment Variables
```bash
# Set specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Disable wandb
export WANDB_DISABLED=true

# Set distributed backend
export NCCL_DEBUG=INFO
```

### Multi-Node Training (Advanced)
```bash
# Node 0 (master)
python train_switch_distributed.py --experts 128 --gpus 4 \
  --master_addr 192.168.1.100 --master_port 12355

# Node 1 (worker)
python train_switch_distributed.py --experts 128 --gpus 4 \
  --master_addr 192.168.1.100 --master_port 12355 \
  --node_rank 1
```

## 🔍 Monitoring and Debugging

### Training Logs
```bash
# Check training progress
tail -f ../models/switch_128_experts_distributed/logs/train.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Distributed Training Debugging
```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO

# Check distributed process group
export TORCH_DISTRIBUTED_DEBUG=INFO
```

### Common Issues and Solutions

#### 1. GPU Selection Issues
```
Error: Only 2 suitable GPUs available, requested 4
```
**Solution**: The script will automatically prompt for a lower count, or use `--gpus 2`

#### 2. Memory Issues
```
CUDA out of memory
```
**Solution**: Reduce batch size or use fewer experts
```bash
python train_switch_distributed.py --experts 64 --gpus 2 --batch_size 2
```

#### 3. NCCL Initialization Failed
```
ProcessGroupNCCL initialization failed
```
**Solution**: Check GPU connectivity and reduce GPU count
```bash
# Test with single GPU first
python train_switch_distributed.py --experts 128 --single_gpu
```

#### 4. Distributed Synchronization Issues
```
Distributed package doesn't have NCCL built in
```
**Solution**: Reinstall PyTorch with NCCL support
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

## 📊 Benchmarking

### Compare Training Methods
```bash
# Single GPU baseline
time python train_switch_unified.py --experts 128

# 2 GPU distributed
time python train_switch_distributed.py --experts 128 --gpus 2

# 4 GPU distributed
time python train_switch_distributed.py --experts 128 --gpus 4
```

### Measure Throughput
```bash
# Samples per second
python -c "
import json
with open('../models/switch_128_experts_distributed/training_info.json') as f:
    info = json.load(f)
    print(f'Samples/sec: {info[\"total_samples\"] / training_time_seconds:.2f}')
"
```

## 🎯 Best Practices

### 1. GPU Selection
- Use launcher script for automatic selection
- Ensure all GPUs have similar memory (mixed GPU types can cause issues)
- Check GPU utilization before training

### 2. Batch Size Selection
- Start with auto-optimized settings
- Increase batch size if memory allows
- Use gradient accumulation for effective larger batches

### 3. Learning Rate Scaling
- Linear scaling works well for Switch Transformers
- Monitor training loss for instability
- Use warmup to stabilize early training

### 4. Monitoring
- Enable wandb for distributed training visualization
- Monitor GPU memory usage across all devices
- Check for load balancing issues

## 🔄 Complete Workflow

### 1. Setup and Check
```bash
# Check system
python launch_training.py --status

# Verify data
ls -la ../data/train/train_data.json
```

### 2. Start Training
```bash
# Launch with confirmation
python launch_training.py --experts 128 --gpus 2

# Or auto-confirm
python launch_training.py --experts 128 --gpus 2 --yes
```

### 3. Monitor Progress
```bash
# Check logs
tail -f ../models/switch_128_experts_distributed/logs/train.log

# Monitor GPUs
nvidia-smi
```

### 4. Evaluate Results
```bash
# Quick evaluation
python simple_switch_eval.py --model_path ../models/switch_128_experts_distributed

# Comprehensive evaluation
python evaluate_model.py --model_path ../models/switch_128_experts_distributed --model_type switch
```

## 📚 Integration with Existing Scripts

The distributed training integrates seamlessly with existing evaluation scripts:

```bash
# All evaluation scripts work with distributed models
python simple_switch_eval.py --model_path ../models/switch_128_experts_distributed
python evaluate_model.py --model_path ../models/switch_128_experts_distributed --model_type switch
```

## 🚀 Quick Reference

### Most Common Commands
```bash
# Check GPUs
python launch_training.py --status

# Train Switch-128 on 2 GPUs
python launch_training.py --experts 128 --gpus 2 --disable_wandb

# Train Switch-256 on 4 GPUs
python launch_training.py --experts 256 --gpus 4 --num_epochs 5

# Direct distributed training
python train_switch_distributed.py --experts 128 --gpus 4 --disable_wandb
```

### Emergency Fallback
If distributed training fails, the scripts automatically fall back to single GPU training:
```bash
# This will use single GPU if distributed fails
python train_switch_distributed.py --experts 128 --gpus 1
```

---

*Distributed training optimized for Switch Transformers with automatic GPU selection and scaling.*