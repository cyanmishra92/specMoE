# DeepSeek-MoE-16B Deployment Guide

## Hardware Requirements

### Recommended GPUs
- **A6000 (48GB)**: Optimal performance with 4-bit quantization
- **A100 (40GB/80GB)**: Excellent performance
- **RTX 3090 (24GB)**: Possible but requires CPU offload + 4-bit quantization
- **RTX 4090 (24GB)**: Good performance with optimization

### System Requirements
- **RAM**: 32GB+ recommended (64GB for RTX 3090 with CPU offload)
- **Storage**: 30GB+ free space for models and traces
- **CUDA**: 11.8 or higher
- **Python**: 3.9, 3.10, or 3.11

## Quick Setup

### 1. Environment Setup
```bash
cd deepseek_moe_16b
python setup.py
```

### 2. Authentication
```bash
huggingface-cli login
```

### 3. Run Collection
```bash
python scripts/collection/collect_deepseek_moe_traces.py
```

## GPU-Specific Configurations

### A6000 (48GB) - Optimal Configuration
```python
config = {
    "quantization": "4bit",
    "device_map": "auto",
    "max_memory": {"0": "45GB"},
    "cpu_offload": False,
    "batch_size": 12,
    "max_length": 512
}
```

### RTX 3090 (24GB) - Challenging Configuration
```python
config = {
    "quantization": "4bit",
    "device_map": "auto",
    "max_memory": {"0": "20GB", "cpu": "16GB"},
    "cpu_offload": True,  # Required
    "batch_size": 8,
    "max_length": 512
}
```

### A100 (40GB) - Good Configuration
```python
config = {
    "quantization": "4bit",
    "device_map": "auto",
    "max_memory": {"0": "38GB"},
    "cpu_offload": False,
    "batch_size": 16,
    "max_length": 512
}
```

## Performance Expectations

### Collection Performance

| GPU | Memory | Batch Size | Rate (traces/min) | Total Time |
|-----|--------|------------|-------------------|------------|
| A6000 | 48GB | 12 | 1500-2500 | 15-20 min |
| RTX 3090 | 24GB | 8 | 1000-1500 | 20-30 min |
| A100 40GB | 40GB | 16 | 2000-3000 | 10-15 min |
| A100 80GB | 80GB | 20 | 2500-3500 | 8-12 min |

### Memory Usage (RTX 3090)
```
Model Loading:     16-18GB VRAM + 8GB CPU
Inference Peak:    20-22GB VRAM + 12GB CPU
Batch Processing:  18-20GB VRAM + 10GB CPU
```

## Expected Output

### Successful Collection
```
🚀 Starting DeepSeek-MoE-16B Trace Collection
✅ Selected GPU 0: NVIDIA GeForce RTX 3090
✅ deepseek-ai/deepseek-moe-16b-base loaded successfully
Target: 20000 total traces, ~3333 per dataset
Using batch size: 8 for NVIDIA GeForce RTX 3090

📊 Processing imdb (default)...
Collecting imdb traces: 100%|████████| 3333/3333 [04:30<00:00, 12.3traces/s]
🚀 Starting GPU inference for batch of 8 samples...
🔥 Running model inference on 8 samples...
✅ Model inference completed in 3.2s (0.4s per sample)
✅ imdb: collected 3333 traces

Expert diversity metrics:
- Expert coverage: 45/64 (70.3%)
- Unique expert pairs: 1,247
- Routing entropy: 4.82

🎉 Collection complete! Total traces: 20,000
```

## Troubleshooting

### Common Issues

1. **Out of Memory on RTX 3090**
   ```python
   # Solution: Reduce batch size and enable CPU offload
   batch_size = 4  # Reduce from 8
   cpu_offload = True
   max_memory = {"0": "18GB", "cpu": "20GB"}
   ```

2. **Slow Performance**
   ```bash
   # Check if CPU offload is being used
   htop  # Should show high CPU usage
   nvidia-smi -l 1  # GPU usage may be lower with CPU offload
   ```

3. **Model Loading Fails**
   ```bash
   # Check model access
   huggingface-cli whoami
   # DeepSeek models should be publicly available
   ```

### Memory Optimization for RTX 3090

1. **Aggressive quantization**:
   ```python
   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4",
       llm_int8_enable_fp32_cpu_offload=True,
   )
   ```

2. **CPU offload configuration**:
   ```python
   max_memory = {
       0: "18GB",    # Keep most on GPU
       "cpu": "20GB"  # Offload some to CPU
   }
   ```

3. **Batch size tuning**:
   ```python
   # Start small and increase if stable
   batch_size = 4  # Conservative
   # If stable, try 6 or 8
   ```

## Expert Analysis Features

### Expert Diversity Tracking
```python
# Built-in expert diversity metrics
expert_usage = {}
for trace in traces:
    for expert_id in trace.target_experts:
        expert_usage[expert_id] = expert_usage.get(expert_id, 0) + 1

# Expert coverage: How many of 64 experts are used?
expert_coverage = len(expert_usage) / 64

# Routing entropy: How diverse is the routing?
routing_entropy = calculate_entropy(expert_usage.values())
```

### Expert Clustering Analysis
```python
# Analyze expert co-occurrence patterns
expert_pairs = []
for trace in traces:
    expert_pairs.append(tuple(sorted(trace.target_experts)))

# Most common expert pairs
from collections import Counter
common_pairs = Counter(expert_pairs).most_common(10)
```

## Performance Comparison

### vs Qwen1.5-MoE-A2.7B
| Metric | Qwen1.5-MoE | DeepSeek-MoE | Trade-off |
|--------|-------------|--------------|-----------|
| Experts | 8 | 64 | 8x more complex |
| VRAM Usage | 8-12GB | 16-20GB | 2x more memory |
| Batch Size | 16 | 8 | 2x smaller batches |
| Collection Speed | 2000-3000/min | 1000-1500/min | 2x slower |
| Prediction Difficulty | Medium | High | Much harder |
| Research Value | Medium | High | More insights |

### vs Mixtral-8x7B
| Metric | Mixtral | DeepSeek-MoE | Advantage |
|--------|---------|--------------|-----------|
| Active Params | 14B | 2.8B | 5x fewer active |
| Total Params | 45B | 16.4B | 2.7x smaller |
| Experts | 8 | 64 | 8x more granular |
| RTX 3090 Fit | Tight | Possible | Better fit |

## Research Opportunities

### Expert Specialization
- **Domain analysis**: Which experts handle what domains?
- **Task specialization**: How do experts specialize by task?
- **Hierarchical clustering**: Are there expert hierarchies?

### Routing Patterns
- **Temporal patterns**: How does routing evolve?
- **Context sensitivity**: What triggers specific experts?
- **Load balancing**: How balanced is expert usage?

## Best Practices

### For RTX 3090 Users
1. **Start conservative**: Use batch_size=4 initially
2. **Monitor memory**: Watch both GPU and CPU usage
3. **Enable CPU offload**: Essential for 24GB VRAM
4. **Use 4-bit quantization**: Required for model fitting
5. **Expect slower speeds**: 2-3x slower than Qwen1.5-MoE

### For Research
1. **Track expert diversity**: Monitor which experts are used
2. **Analyze routing patterns**: Study expert co-occurrence
3. **Compare with baselines**: Use 8-expert models for comparison
4. **Study specialization**: What makes each expert unique?

## Advanced Configuration

### Custom Expert Analysis
```python
def analyze_expert_specialization(traces):
    expert_contexts = defaultdict(list)
    
    for trace in traces:
        for expert_id in trace.target_experts:
            expert_contexts[expert_id].append(trace.context)
    
    # Analyze what contexts each expert handles
    expert_specialization = {}
    for expert_id, contexts in expert_contexts.items():
        # Analyze context patterns, keywords, etc.
        expert_specialization[expert_id] = analyze_contexts(contexts)
    
    return expert_specialization
```

### Memory-Efficient Collection
```python
# For very memory-constrained setups
config = {
    "quantization": "4bit",
    "max_memory": {"0": "16GB", "cpu": "24GB"},
    "cpu_offload": True,
    "batch_size": 4,
    "max_length": 256,  # Shorter sequences
    "gradient_checkpointing": True
}
```

## Next Steps

1. **Analyze expert patterns**: Study the 64-expert routing
2. **Compare with 8-expert models**: Understand specialization differences
3. **Train hierarchical predictors**: Two-stage prediction models
4. **Research expert clustering**: Discover expert relationships

DeepSeek-MoE-16B offers **unique research opportunities** in fine-grained expert specialization, though it requires more computational resources than simpler MoE models.