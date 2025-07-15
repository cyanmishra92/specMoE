# Mixtral 8x7B Deployment Guide

## Hardware Requirements

### Recommended GPUs
- **A100 (80GB)**: Optimal performance, full model in GPU memory
- **A100 (40GB)**: Good performance, may need some CPU offload
- **A6000 (48GB)**: Good performance, recommended minimum
- **RTX 3090 (24GB)**: Minimum requirement, requires CPU offload

### System Requirements
- **RAM**: 64GB+ recommended (32GB minimum)
- **Storage**: 100GB+ free space for models and traces
- **CUDA**: 11.8 or higher
- **Python**: 3.10 or 3.11

## Quick Setup

### 1. Environment Setup
```bash
# Make setup script executable
chmod +x setup_environment.sh

# Run setup script
./setup_environment.sh

# Activate environment
conda activate mixtral_moe
```

### 2. Authentication
```bash
# Login to Hugging Face
huggingface-cli login

# Request access to gated models (if needed)
# Visit: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
```

### 3. Run Collection
```bash
# Start trace collection
python scripts/collection/collect_mixtral_traces.py

# Monitor progress
tail -f logs/collection.log
```

## GPU-Specific Configurations

### A100 (80GB) - Optimal
```python
# No special configuration needed
# Model will fit entirely in GPU memory
quantization = "4bit"  # Optional for faster inference
device_map = "auto"
```

### A100 (40GB) - Good
```python
# May need slight CPU offload
quantization = "4bit"  # Recommended
device_map = "auto"
max_memory = {0: "38GB", "cpu": "20GB"}
```

### A6000 (48GB) - Recommended Minimum
```python
# Requires 4-bit quantization
quantization = "4bit"  # Required
device_map = "auto"
max_memory = {0: "45GB", "cpu": "20GB"}
```

### RTX 3090 (24GB) - Minimum
```python
# Requires aggressive optimization
quantization = "4bit"  # Required
device_map = "sequential"  # Important for memory
max_memory = {0: "20GB", "cpu": "40GB"}
cpu_offload = True
```

## Performance Expectations

| GPU | Memory | Quantization | Speed | Model Fit |
|-----|--------|--------------|-------|-----------|
| A100 80GB | 80GB | None/4bit | Very Fast | Full GPU |
| A100 40GB | 40GB | 4bit | Fast | GPU + CPU |
| A6000 | 48GB | 4bit | Good | GPU + CPU |
| RTX 3090 | 24GB | 4bit | Slow | CPU Heavy |

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Solution: Enable CPU offload
   export CUDA_VISIBLE_DEVICES=0
   python scripts/collection/collect_mixtral_traces.py
   ```

2. **Model Access Denied**
   ```bash
   # Solution: Request access and re-login
   huggingface-cli login
   ```

3. **Slow Performance**
   ```bash
   # Solution: Check GPU utilization
   nvidia-smi -l 1
   ```

### Memory Optimization Tips

1. **Use 4-bit quantization** for all GPUs except A100-80GB
2. **Enable CPU offload** for GPUs with <48GB VRAM
3. **Use sequential device mapping** for RTX 3090
4. **Monitor memory usage** with `nvidia-smi`
5. **Clear cache** between runs: `torch.cuda.empty_cache()`

## Expected Output

### Successful Collection
```
🚀 Starting Mixtral 8x7B MoE Trace Collection
✅ mistralai/Mixtral-8x7B-Instruct-v0.1 loaded successfully
📊 Processing wikitext (wikitext-2-raw-v1)...
Processing wikitext: 100%|████████| 150/150 [02:30<00:00, 1.00it/s]
✅ Saved 45,000 traces to routing_data/mixtral_8x7b_traces.pkl
🎉 Collection complete! Total traces: 45,000
```

### File Structure After Collection
```
mixtral_8x7B/
├── routing_data/
│   ├── mixtral_8x7b_traces.pkl    # Main trace data
│   ├── mixtral_8x7b_traces.json   # Metadata
│   └── collection_stats.json       # Collection statistics
├── models/
│   └── (trained models will be saved here)
└── logs/
    └── collection.log              # Detailed logs
```

## Next Steps

1. **Analyze traces**: `python scripts/analysis/visualize_mixtral_routing.py`
2. **Train models**: `python scripts/training/train_mixtral_speculation.py`
3. **Compare results**: Compare with Switch Transformer implementation
4. **Scale experiments**: Try different datasets and model sizes

## Monitoring

### Real-time Monitoring
```bash
# GPU usage
watch -n 1 nvidia-smi

# Memory usage
htop

# Collection progress
tail -f logs/collection.log
```

### Performance Metrics
- **Trace collection rate**: ~300-500 traces/minute on A100
- **Memory usage**: ~30-40GB peak on A100
- **Collection time**: ~2-3 hours for full dataset
- **Output size**: ~500MB-1GB trace files

## Support

For issues:
1. Check GPU memory with `nvidia-smi`
2. Verify environment setup
3. Check HuggingFace authentication
4. Review logs in `logs/collection.log`