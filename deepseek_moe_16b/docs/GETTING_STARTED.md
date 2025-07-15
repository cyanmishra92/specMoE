# Getting Started with DeepSeek-MoE-16B

## Quick Start (10 minutes)

### 1. Setup
```bash
cd deepseek_moe_16b
python setup.py
huggingface-cli login
```

### 2. Collect Traces
```bash
python scripts/collection/collect_deepseek_moe_traces.py
```

### 3. Expected Output
```
🚀 Starting DeepSeek-MoE-16B Trace Collection
✅ Selected GPU 0: NVIDIA GeForce RTX 3090
✅ deepseek-ai/deepseek-moe-16b-base loaded successfully
Target: 20000 total traces, ~3333 per dataset
Using batch size: 8 for NVIDIA GeForce RTX 3090

📊 Processing imdb (default)...
Expert diversity: 45/64 experts used (70.3%)
🎉 Collection complete! Total traces: 20,000
```

## Why Choose DeepSeek-MoE-16B?

### Research Value
- **64 experts**: Fine-grained specialization study
- **Complex routing**: More challenging prediction task
- **Expert diversity**: Rich patterns to analyze
- **Hierarchical structure**: Discover expert clustering

### RTX 3090 Compatible
- **Fits with optimization**: 4-bit quantization + CPU offload
- **Reasonable performance**: 1000-1500 traces/minute
- **Manageable memory**: 16-20GB VRAM with offload
- **Research-friendly**: Designed for academic use

### Unique Insights
- **Expert specialization**: What do 64 experts learn?
- **Routing patterns**: Complex expert interactions
- **Task distribution**: How tasks map to experts
- **Hierarchical clustering**: Expert organization

## Step-by-Step Guide

### Step 1: Environment Setup
```bash
# Install dependencies (more memory needed)
pip install torch transformers datasets tqdm bitsandbytes accelerate huggingface-hub GPUtil

# Or use setup script
python setup.py
```

### Step 2: Check GPU Compatibility
```bash
python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / 1024**3
    print(f'GPU: {props.name}')
    print(f'Memory: {memory_gb:.1f}GB')
    if memory_gb >= 24:
        print('✅ Good for DeepSeek-MoE-16B')
    elif memory_gb >= 16:
        print('⚠️ May work with aggressive optimization')
    else:
        print('❌ Insufficient memory')
"
```

### Step 3: Authentication
```bash
# Login to HuggingFace
huggingface-cli login

# Verify access to DeepSeek models
huggingface-cli whoami
```

### Step 4: Run Collection
```bash
python scripts/collection/collect_deepseek_moe_traces.py
```

### Step 5: Monitor Resources
```bash
# Monitor GPU and CPU usage
nvidia-smi -l 1
htop  # Should show high CPU usage due to offload
```

## What You'll See

### Model Loading (RTX 3090)
```
🚀 DeepSeek-MoE-16B Trace Collector
Selected GPU: NVIDIA GeForce RTX 3090 (24.0GB)
Device: cuda:0
Loading DeepSeek-MoE-16B model...
Optimal config for NVIDIA GeForce RTX 3090: {'quantization': '4bit', 'device_map': 'auto', 'max_memory': {'0': '20GB', 'cpu': '16GB'}, 'cpu_offload': True}
✅ deepseek-ai/deepseek-moe-16b-base loaded successfully
Number of experts: 64
```

### Dataset Processing
```
📊 Processing imdb (default)...
Using batch size: 8 for NVIDIA GeForce RTX 3090
Collecting imdb traces: 100%|████████| 3333/3333 [04:30<00:00, 12.3traces/s]
🚀 Starting GPU inference for batch of 8 samples...
🔥 Running model inference on 8 samples...
✅ Model inference completed in 3.2s (0.4s per sample)

Expert diversity metrics:
- Expert coverage: 45/64 (70.3%)
- Unique expert pairs: 1,247
- Routing entropy: 4.82
✅ imdb: collected 3333 traces
```

### Final Results
```
✅ Saved 20,000 traces to routing_data/deepseek_moe_16b_traces.pkl
📊 File size: 580.2 MB
📄 Metadata saved to routing_data/deepseek_moe_16b_traces.json
🎉 Collection complete! Total traces: 20,000

Final Expert Analysis:
- Total experts used: 58/64 (90.6%)
- Most active expert: Expert 23 (8.4% of traces)
- Least active expert: Expert 47 (0.2% of traces)
- Average experts per layer: 42.3
```

## GPU Memory Usage (RTX 3090)

### Resource Timeline
```
Model Loading:     16-18GB VRAM + 8GB CPU
Batch Processing:  18-20GB VRAM + 12GB CPU
Peak Usage:        20-22GB VRAM + 16GB CPU
Steady State:      18-20GB VRAM + 10GB CPU
```

### Expected nvidia-smi Output
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0  |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA GeForce RTX 3090    On   | 00000000:01:00.0 Off |                  N/A |
| 45%   65C    P2    280W / 350W |  19840MiB / 24576MiB |     70%      Default |
+-------------------------------+----------------------+----------------------+
```

### Expected htop Output
```
# High CPU usage due to offloading
CPU: 45-60% usage across cores
Memory: 12-16GB RAM used
Load: 4-8 (depends on CPU cores)
```

## File Structure After Collection

```
deepseek_moe_16b/
├── routing_data/
│   ├── deepseek_moe_16b_traces.pkl     # Main trace data (580MB)
│   ├── deepseek_moe_16b_traces.json    # Metadata
│   └── expert_analysis.json            # Expert diversity analysis
├── logs/
│   └── collection.log                  # Detailed logs
├── models/                             # (Empty initially)
└── results/                            # (Empty initially)
```

## Understanding the Traces

### Trace Structure
```python
# Each trace contains:
trace.layer_id          # Which MoE layer (0-27)
trace.hidden_states     # Hidden state tensor
trace.target_routing    # Expert routing probabilities (64-dim)
trace.target_top_k      # Top-2 selected experts
trace.token_ids         # Token ID
trace.dataset_name      # Source dataset
trace.sample_id         # Sample identifier
```

### Expert Routing (64 experts)
```python
# Top-2 routing example:
target_top_k = [23, 47]   # Experts 23 and 47 selected
target_routing = [0.0, 0.0, ..., 0.7, ..., 0.3, ...]  # 64-dim probabilities
```

## Expert Diversity Analysis

### Built-in Metrics
```python
# Expert coverage: How many experts are used?
expert_coverage = unique_experts_used / 64

# Routing entropy: How diverse is the routing?
routing_entropy = -sum(p * log(p) for p in expert_probabilities)

# Expert specialization: How specialized are experts?
specialization_score = calculate_specialization(expert_usage)
```

### Expert Clustering
```python
# Most common expert pairs
expert_pairs = Counter([(e1, e2) for e1, e2 in selected_expert_pairs])

# Example output:
# (23, 47): 1,247 occurrences
# (12, 35): 1,156 occurrences
# (8, 51): 1,089 occurrences
```

## Next Steps

### 1. Analyze Expert Patterns
```bash
python scripts/analysis/analyze_deepseek_expert_patterns.py
```

### 2. Train Hierarchical Predictors
```bash
python scripts/training/train_hierarchical_prediction.py
```

### 3. Compare with 8-Expert Models
```bash
# Compare complexity vs Qwen1.5-MoE or Mixtral
python scripts/analysis/compare_expert_complexity.py
```

## Performance Expectations

### RTX 3090 Performance
- **Collection time**: 20-30 minutes for 20,000 traces
- **Batch size**: 8 samples (vs 16 for Qwen1.5-MoE)
- **Memory pressure**: High (uses 20GB/24GB)
- **CPU offload**: Required (uses 12-16GB RAM)

### A6000 Performance
- **Collection time**: 15-20 minutes for 20,000 traces
- **Batch size**: 12 samples
- **Memory pressure**: Medium (uses 30GB/48GB)
- **CPU offload**: Optional

## Troubleshooting

### RTX 3090 Issues

**Out of Memory**
```python
# Reduce batch size and increase CPU offload
batch_size = 4  # Reduce from 8
max_memory = {"0": "18GB", "cpu": "20GB"}
```

**Slow Performance**
```bash
# Check if CPU offload is working
htop  # Should show 40-60% CPU usage
nvidia-smi -l 1  # GPU usage may be 60-80% (normal with offload)
```

**Model Loading Fails**
```bash
# Check available memory
nvidia-smi
# Ensure at least 16GB free before starting
```

### Performance Tips

**For RTX 3090**
1. **Use 4-bit quantization**: Essential for fitting
2. **Enable CPU offload**: Required for 24GB VRAM
3. **Monitor temperature**: May run hotter due to memory pressure
4. **Ensure adequate RAM**: Need 16GB+ system RAM

**For A6000**
1. **Use 4-bit quantization**: For optimal speed
2. **Larger batch sizes**: Can use batch_size=12
3. **Optional CPU offload**: More flexibility
4. **Better performance**: ~2x faster than RTX 3090

## Research Applications

### Expert Specialization Study
```python
# Analyze what each expert specializes in
expert_contexts = defaultdict(list)
for trace in traces:
    for expert_id in trace.target_experts:
        expert_contexts[expert_id].append(trace.context)

# Find expert specialization patterns
for expert_id, contexts in expert_contexts.items():
    print(f"Expert {expert_id}: {analyze_specialization(contexts)}")
```

### Hierarchical Routing Analysis
```python
# Discover expert hierarchies
expert_similarity = calculate_expert_similarity(traces)
expert_clusters = cluster_experts(expert_similarity, n_clusters=8)

# Analyze cluster characteristics
for cluster_id, experts in expert_clusters.items():
    print(f"Cluster {cluster_id}: {analyze_cluster(experts)}")
```

## Why This Is Valuable

### vs Qwen1.5-MoE-A2.7B
- **8x more experts**: 64 vs 8 experts
- **Fine-grained specialization**: More detailed expert roles
- **Complex routing**: More challenging prediction task
- **Research depth**: Deeper insights into MoE behavior

### vs Mixtral-8x7B
- **Better RTX 3090 fit**: 16.4B vs 45B parameters
- **More experts**: 64 vs 8 experts
- **Faster inference**: 2.8B vs 14B active parameters
- **Research novelty**: Unique architecture to study

## Summary

**DeepSeek-MoE-16B** is ideal for researchers who want:
- ✅ **Fine-grained expert analysis** (64 experts)
- ✅ **Complex routing patterns** to study
- ✅ **RTX 3090 compatibility** (with optimization)
- ✅ **Unique research insights** into MoE specialization

**Expected results**: 20,000 traces in 20-30 minutes with rich expert diversity analysis!

**Note**: Requires more resources than Qwen1.5-MoE but provides unique research value for understanding fine-grained expert specialization.