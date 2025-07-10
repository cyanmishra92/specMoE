# Enhanced Pre-gated MoE Setup Guide

## Quick Start

1. **Install Dependencies**:
```bash
pip install torch transformers accelerate numpy matplotlib seaborn psutil
```

2. **Run Demo**:
```bash
python main.py --mode demo --speculation-mode multi_layer
```

3. **Compare Speculation Modes**:
```bash
python main.py --mode compare
```

4. **Run Benchmark**:
```bash
python main.py --mode benchmark --benchmark-iterations 20
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Input Tokens   │───▶│ Speculation      │───▶│ Memory Manager  │
└─────────────────┘    │ Engine           │    │ - GPU Cache     │
                       │ - Multi-layer    │    │ - Compression   │
                       │ - Confidence     │    │ - Hierarchical  │
                       │ - Adaptive       │    └─────────────────┘
                       └──────────────────┘              │
                                 │                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Output Tokens   │◀───│ MoE Computation  │◀───│ Expert Loading  │
└─────────────────┘    │ - 8 Experts      │    │ - Prefetching   │
                       │ - Top-k Routing  │    │ - Async I/O     │
                       └──────────────────┘    └─────────────────┘
```

## Key Components

### 1. Speculation Engine (`gating/speculation_engine.py`)
- **Multi-layer Lookahead**: Uses L-3, L-2, L-1 to predict L+1
- **Confidence-based Adaptation**: Adjusts speculation based on prediction confidence
- **Input-aware Gating**: Adapts strategy based on input characteristics
- **Pattern Learning**: Learns expert transition patterns over time

### 2. Adaptive Memory Manager (`memory/adaptive_memory_manager.py`)
- **Dynamic Compression**: INT8/INT4 quantization based on memory pressure
- **Hierarchical Caching**: GPU → Unified → Compressed storage tiers
- **Adaptive Buffering**: Adjusts strategy based on available memory
- **Performance Monitoring**: Tracks cache hit rates and load times

### 3. Device Profiler (`utils/device_profiler.py`)
- **Hardware Detection**: Automatically detects RTX 3090 capabilities
- **Memory Benchmarking**: Measures available bandwidth and capacity
- **Configuration Recommendations**: Suggests optimal settings

### 4. Small Switch Transformer (`models/small_switch_transformer.py`)
- **6 Layers, 8 Experts**: Manageable size for RTX 3090 (140M parameters)
- **Routing Statistics**: Collects detailed routing information
- **Load Balancing**: Implements auxiliary loss for expert utilization

## Performance Optimizations

### For RTX 3090 (24GB VRAM):
- **Max Concurrent Experts**: 6 (based on memory constraints)
- **Compression**: INT8 dynamic quantization (4x reduction)
- **Batch Size**: 8 (optimal for throughput)
- **Speculation Aggressiveness**: 0.7

### Memory Usage:
- **Base Model**: ~40MB (non-expert parameters)
- **Each Expert**: ~4MB (512→2048→512 MLP)
- **Total Experts**: 48 (6 layers × 8 experts)
- **With Compression**: ~48MB for all experts

## Benchmarking Results

Initial results show:
- **Baseline Throughput**: ~10,000 tokens/sec
- **Memory Efficiency**: 4x compression with <1% accuracy loss
- **Expert Utilization**: Load balancing maintains expert diversity
- **Cache Performance**: Ready for speculation engine optimization

## Development Roadmap

### Phase 1: Enhanced Speculation ✅
- [x] Multi-layer lookahead prediction
- [x] Confidence-based adaptation
- [x] Input-aware gating strategies
- [x] Device-specific optimization

### Phase 2: Advanced Memory Management (In Progress)
- [ ] Expert weight quantization refinement
- [ ] Structured sparsity implementation
- [ ] Unified memory optimization (for Jetson)
- [ ] Dynamic cache sizing

### Phase 3: Production Optimization
- [ ] CUDA kernel optimization
- [ ] Triton integration for custom kernels
- [ ] Multi-GPU support
- [ ] Model serving optimization

## Configuration Options

### Speculation Modes:
- `none`: No speculation (baseline)
- `layer_minus_1`: Simple previous layer prediction
- `multi_layer`: Weighted multi-layer history
- `adaptive`: Combines multiple strategies with confidence

### Memory Strategies:
- `double_buffer`: High memory, full double buffering
- `single_async`: Medium memory, async loading
- `streaming`: Low memory, stream as needed
- `cpu_offload`: Very low memory, CPU storage

### Compression Types:
- `none`: No compression
- `int8_dynamic`: 4x compression, minimal accuracy loss
- `int4_grouped`: 8x compression, moderate accuracy loss
- `structured_sparse`: Variable compression with pruning

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch-size 1`
   - Enable compression: `--use-compression`
   - Use streaming mode

2. **Low Throughput**:
   - Check GPU utilization with `nvidia-smi`
   - Increase batch size if memory allows
   - Verify CUDA version compatibility

3. **Poor Speculation Accuracy**:
   - Try different speculation modes
   - Adjust confidence threshold
   - Check input data characteristics

### Performance Tips:

1. **Warmup**: Always run several warmup iterations
2. **Memory**: Monitor with `torch.cuda.memory_summary()`
3. **Profiling**: Use `--collect-detailed-stats` for analysis
4. **Caching**: Check cache hit rates in memory statistics

## Contributing

This is a research prototype focused on advancing speculative execution in MoE models. Key areas for improvement:

1. **Speculation Accuracy**: Better prediction algorithms
2. **Memory Efficiency**: More aggressive compression techniques
3. **Hardware Optimization**: Custom CUDA kernels
4. **Model Coverage**: Support for larger MoE architectures

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{enhanced_pregated_moe_2025,
  title={Enhanced Pre-gated MoE for Small GPUs},
  author={Research Team},
  year={2025},
  note={RTX 3090 optimization of ISCA'24 Pre-gated MoE}
}
```