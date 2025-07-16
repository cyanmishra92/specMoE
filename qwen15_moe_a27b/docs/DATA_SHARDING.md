# Data Sharding for Memory-Efficient Training

## Overview

Data sharding is a memory optimization technique that splits large trace files into smaller, manageable chunks. This enables training on RTX 3090 (24GB) and other memory-constrained GPUs.

## Why Data Sharding?

### The Problem
- **Large trace files**: 68-80GB pickle files from medium/large trace collection
- **Memory constraints**: RTX 3090 has only 24GB VRAM
- **Training bottlenecks**: Cannot load entire dataset into memory

### The Solution
- **Automatic sharding**: Split large files into 400-500MB chunks
- **Streaming training**: Load one shard at a time during training
- **Layer-aware grouping**: Keep related traces together for better performance

## How It Works

### 1. Automatic Sharding During Collection

```bash
# Enable sharding during trace collection
python scripts/collection/collect_qwen15_moe_traces_medium.py \
  --target_traces 5000 \
  --shard_data \
  --shard_size_mb 400
```

This creates:
```
routing_data/
├── qwen15_moe_a27b_traces_medium.pkl          # Original file (68GB)
└── qwen15_moe_a27b_traces_medium_shards/       # Sharded directory
    ├── shard_000.pkl                           # 400MB
    ├── shard_001.pkl                           # 400MB
    ├── shard_002.pkl                           # 400MB
    ├── ...
    ├── shard_170.pkl                           # 400MB
    └── sharding_metadata.pkl                   # Metadata
```

### 2. Memory-Efficient Training

```bash
# Train with sharded data
python scripts/train_multi_expert_predictor.py \
  --shard_dir routing_data/qwen15_moe_a27b_traces_medium_shards \
  --batch_size 4 \
  --epochs 50
```

## Sharding Process

### Step 1: Trace Analysis
```python
# Estimate memory usage per trace
def estimate_trace_size(trace):
    size = 0
    size += trace.hidden_states.numel() * trace.hidden_states.element_size()
    size += trace.target_routing.numel() * trace.target_routing.element_size()
    # ... other tensors
    return size
```

### Step 2: Layer-Aware Grouping
```python
# Group traces by layer for better locality
layer_groups = {}
for trace in all_traces:
    layer_id = trace.layer_id
    if layer_id not in layer_groups:
        layer_groups[layer_id] = []
    layer_groups[layer_id].append(trace)
```

### Step 3: Smart Sharding
```python
# Create shards with target size
for layer_id in sorted(layer_groups.keys()):
    for trace in layer_groups[layer_id]:
        if current_size + trace_size > shard_size_bytes:
            # Save current shard and start new one
            save_shard(current_shard)
            current_shard = []
```

## Memory Benefits

### Before Sharding
```
Training Process:
├── Load entire dataset: 68GB → OOM on RTX 3090
├── PyTorch DataLoader: Additional 20GB+ memory
└── Model + Optimizer: 12GB
Total: ~100GB (Impossible on RTX 3090)
```

### After Sharding
```
Training Process:
├── Load one shard: 400MB
├── PyTorch DataLoader: 1GB
├── Model + Optimizer: 12GB
└── GPU cache: 2GB
Total: ~15GB (Comfortable on RTX 3090)
```

## Configuration Options

### Shard Size Recommendations

| GPU Model | VRAM | Recommended Shard Size | Batch Size |
|-----------|------|----------------------|------------|
| RTX 3090  | 24GB | 400MB                | 4-8        |
| A6000     | 48GB | 500MB                | 8-12       |
| A100      | 80GB | 1GB                  | 16-24      |

### Command Line Options

```bash
# Trace collection with sharding
--shard_data                      # Enable sharding
--shard_size_mb 400              # Target shard size (MB)
--output_suffix rtx3090          # Custom suffix

# Training with sharded data
--shard_dir path/to/shards       # Sharded data directory
--batch_size 4                   # Batch size
--gradient_accumulation_steps 4  # Effective larger batch size
```

## Advanced Usage

### Manual Sharding

```python
from scripts.utils.data_sharding import TraceDataSharder

# Create sharder
sharder = TraceDataSharder(shard_size_mb=400)

# Shard existing trace file
shard_files = sharder.shard_traces(
    input_file="routing_data/large_traces.pkl",
    output_dir="routing_data/sharded"
)
```

### Custom Data Loader

```python
from scripts.utils.data_sharding import ShardedDataLoader

# Create sharded loader
loader = ShardedDataLoader(
    shard_dir="routing_data/sharded",
    batch_size=4,
    shuffle=True
)

# Iterate over batches
for batch in loader.iterate_batches():
    # Process batch
    pass
```

## Performance Optimization

### Memory Usage Monitoring

```python
import torch

# Monitor GPU memory during training
def log_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    cached = torch.cuda.memory_reserved() / 1024**3      # GB
    print(f"GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")

# Clear cache periodically
if batch_idx % 10 == 0:
    torch.cuda.empty_cache()
```

### Batch Size Optimization

```python
# Find optimal batch size for your GPU
def find_optimal_batch_size(model, device):
    batch_sizes = [1, 2, 4, 8, 16]
    for batch_size in batch_sizes:
        try:
            # Test batch processing
            test_batch = create_dummy_batch(batch_size)
            model(test_batch)
            torch.cuda.empty_cache()
            print(f"Batch size {batch_size}: ✅ Success")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: ❌ OOM")
                break
```

## Troubleshooting

### Common Issues

1. **OOM during sharding**:
   ```bash
   # Use smaller shard size
   --shard_size_mb 200
   ```

2. **Slow training**:
   ```bash
   # Use fewer workers
   --num_workers 1
   
   # Enable gradient accumulation
   --gradient_accumulation_steps 8
   ```

3. **Disk space issues**:
   ```bash
   # Remove original file after sharding
   rm routing_data/original_traces.pkl
   ```

### Memory Tips

- **Start small**: Begin with small shard sizes and increase gradually
- **Monitor usage**: Use `nvidia-smi` to watch GPU memory
- **Clear cache**: Call `torch.cuda.empty_cache()` periodically
- **Reduce precision**: Use mixed precision training if available

## Example Workflow

```bash
# 1. Collect traces with sharding
python scripts/collection/collect_qwen15_moe_traces_medium.py \
  --target_traces 5000 \
  --shard_data \
  --shard_size_mb 400 \
  --output_suffix rtx3090

# 2. Verify sharding
ls -lh routing_data/qwen15_moe_a27b_traces_medium_rtx3090_shards/

# 3. Train with sharded data
python scripts/train_multi_expert_predictor.py \
  --shard_dir routing_data/qwen15_moe_a27b_traces_medium_rtx3090_shards \
  --batch_size 4 \
  --epochs 50 \
  --device cuda

# 4. Monitor GPU usage
watch -n 1 nvidia-smi
```

This approach makes it possible to train on datasets of any size using RTX 3090 and other memory-constrained GPUs!