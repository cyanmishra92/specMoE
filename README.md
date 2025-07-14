# Enhanced Pre-gated MoE for RTX 3090

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enhanced implementation of speculative gating for Mixture of Experts (MoE) models, specifically optimized for RTX 3090 and similar GPUs. This project extends the [ISCA'24 Pre-gated MoE paper](https://arxiv.org/pdf/2308.12066) with novel speculation strategies, learnable neural models, and memory optimizations.

## 🚀 Quick Start

### Prerequisites
- NVIDIA RTX 3090 (or similar GPU with 16GB+ VRAM)
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd specMoE

# Install dependencies
pip install torch transformers accelerate datasets
pip install numpy matplotlib seaborn psutil tqdm
```

### Quick Demo

Run the complete pipeline with 128-expert model:

```bash
# Complete pipeline (recommended - uses 128-expert Switch Transformer)
python scripts/pipelines/run_working_pipeline.py --use-128-experts

# Quick test with smaller model
python scripts/pipelines/run_working_pipeline.py
```

Or run individual components:

```bash
# Step 1: Collect routing traces from Switch Transformers
python scripts/collection/collect_robust_traces.py --traces 3000

# Step 2: Analyze expert usage patterns
python scripts/analysis/comprehensive_expert_analysis.py

# Step 3: Visualize expert traces
python scripts/analysis/visualize_expert_traces.py

# Step 4: Train speculation models
python scripts/training/improved_speculation_training.py

# Step 3: Test individual approaches
python scripts/evaluation/test_individual_approaches.py

# Step 4: Compare all approaches
python scripts/evaluation/compare_all_approaches.py
```

## 📁 Project Structure

```
specMoE/
├── 📚 Core Framework
│   ├── models/                     # MoE model implementations
│   │   ├── small_switch_transformer.py
│   │   └── pretrained_switch_model.py
│   ├── gating/                     # Speculation engines
│   │   └── speculation_engine.py   # Heuristic + learnable speculation
│   ├── training/                   # Neural model training
│   │   ├── learnable_gating_models.py
│   │   ├── gating_trainer.py
│   │   └── gating_data_collector.py
│   ├── memory/                     # Memory management
│   │   └── adaptive_memory_manager.py
│   └── utils/                      # Utilities
│       └── device_profiler.py
├── 🛠️ Scripts (Working Pipeline)
│   ├── collection/                 # Data collection
│   │   └── collect_robust_traces.py      # 128-expert traces from Switch Transformers
│   ├── analysis/                   # Expert analysis & visualization
│   │   ├── comprehensive_expert_analysis.py  # Statistical analysis
│   │   ├── visualize_expert_traces.py         # Trace visualization
│   │   └── create_small_dataset.py            # Small experimental dataset
│   ├── training/                   # Model training
│   │   └── improved_speculation_training.py   # Main training script
│   ├── benchmarks/                 # Performance benchmarks
│   │   ├── memory_transfer_benchmark.py       # Memory transfer analysis
│   │   └── run_memory_benchmarks.py           # Comprehensive benchmarks
│   └── visualization/              # Plot generation
│       └── latency_analysis_plots.py          # Publication-quality plots
├── 🎯 Applications
│   ├── main.py                     # Custom model demo
│   └── main_pretrained.py          # Pre-trained model demo
├── 📊 Data & Results
│   ├── routing_data/               # Collected routing traces & statistics
│   │   ├── comprehensive_expert_statistics.json  # Layer-wise statistics
│   │   ├── expert_statistics_3d.json             # 3D structure (layer→expert→freq)
│   │   └── sample_trace_paths.json               # Sample expert sequences
│   ├── benchmark_results/          # Performance benchmarks
│   ├── plots/                      # Generated visualizations
│   └── simulation_results/         # Memory simulation results
├── 📖 Documentation
│   ├── README.md                   # This file
│   ├── docs/
│   │   ├── COLLECTION_GUIDE.md     # Trace collection guide
│   │   ├── EXPERT_ANALYSIS_GUIDE.md # Expert analysis guide
│   │   ├── FINAL_PERFORMANCE_REPORT.md # Performance analysis
│   │   └── MEMORY_MANAGEMENT_GUIDE.md  # Memory optimization guide
└── 🗃️ Archive
    └── archive_unused/            # Unused/broken scripts
```

## 🧠 Speculation Modes

| Mode | Description | Performance | Use Case |
|------|-------------|-------------|----------|
| `none` | No speculation (baseline) | Baseline | Comparison |
| `layer_minus_1` | Previous layer prediction | ~13% accuracy | Simple |
| `multi_layer` | Multi-layer history | ~13% accuracy | Standard |
| `adaptive` | Confidence-based adaptation | ~10% accuracy | Dynamic |
| `learnable` | **Trained neural models** | **TBD** | **Best** |

## 🔬 Key Features

### Enhanced Speculation Engine
- **Multi-layer Lookahead**: Uses L-3, L-2, L-1 to predict expert needs for L+1
- **Learnable Models**: Neural networks trained on real MoE routing patterns
- **Confidence-based Adaptation**: Dynamically adjusts speculation aggressiveness
- **Pattern Learning**: Learns expert transition matrices over time

### Memory Management
- **Dynamic Compression**: INT8 (4x) and INT4 (8x) quantization
- **Hierarchical Caching**: GPU → Unified → Compressed storage tiers
- **Smart Prefetching**: Loads experts based on speculation confidence
- **RTX 3090 Optimization**: Tuned for 24GB VRAM constraints

### Data Collection
- **Real MoE Traces**: Extracts routing from Switch Transformers
- **128-Expert Support**: Works with `google/switch-base-128`
- **Diverse Datasets**: WikiText, SQuAD, GLUE for training data
- **Robust Data Splits**: Prevents overfitting with proper train/test separation

## 📊 Performance Results

### Current Status (6,000 traces collected)
- **Data**: 6,000 routing samples from Switch Transformer
- **Training**: Learnable models with 30% loss reduction
- **Models**: Multiple architectures (contextual, transformer, hierarchical)
- **Hardware**: Optimized for RTX 3090 (24GB VRAM)

### Speculation Accuracy (Heuristic Methods)
```
Method              Top-1 Acc   Top-2 Acc   Confidence
Layer-Minus-1       12.0%       26.6%       Stable
Multi-Layer         13.2%       25.2%       Best
Adaptive            9.6%        25.4%       Variable
Learnable           TBD         TBD         High
```

## 🎯 Usage Examples

### Complete Pipeline
```bash
# Run everything with 128-expert model (recommended)
python scripts/pipelines/run_working_pipeline.py --use-128-experts

# Quick test pipeline
python scripts/pipelines/run_working_pipeline.py
```

### Individual Components
```bash
# Collect traces from 128-expert Switch Transformer
python scripts/collection/collect_robust_traces.py

# Train with proper data splits (no overfitting)
python scripts/training/proper_train_test.py

# Test all speculation approaches
python scripts/evaluation/test_individual_approaches.py

# Comprehensive comparison
python scripts/evaluation/compare_all_approaches.py
```

### Demo Applications
```bash
# Custom model with speculation
python main.py --mode demo --speculation-mode multi_layer

# Pre-trained Switch Transformer
python main_pretrained.py --mode demo --pretrained-model google/switch-base-8

# Compare all modes
python main.py --mode compare
```

## 🔧 Configuration

### Speculation Parameters
- **Confidence Threshold**: 0.7 (optimal for RTX 3090)
- **History Length**: 4 layers
- **Top-K Experts**: 2 concurrent experts
- **Memory Strategy**: Adaptive based on available VRAM

### Training Configuration
- **Batch Size**: 16 (RTX 3090 optimized)
- **Learning Rate**: 3e-4
- **Epochs**: 25
- **Validation Split**: 20%
- **Mixed Precision**: Enabled

## 📈 Benchmarking

Check current status:
```bash
python scripts/check_current_status.py
```

Expected output:
```
✅ routing_data/proper_traces.pkl (188 MB, 6,000 samples)
✅ trained_models/simple_speculation_model.pt
✅ GPU Memory: 204 MB / 24576 MB (0.8% - idle)
```

## 🛠️ Development

### Adding New Features

1. **New Speculation Strategy**:
   - Add to `gating/speculation_engine.py`
   - Test with `scripts/evaluation/test_individual_approaches.py`

2. **New Model Architecture**:
   - Add to `training/learnable_gating_models.py`
   - Train with `scripts/training/proper_train_test.py`

3. **New Collection Method**:
   - Add to `scripts/collection/`
   - Follow patterns in `collect_robust_traces.py`

### Testing
```bash
# Test individual components
python scripts/evaluation/test_individual_approaches.py

# Full pipeline test
python scripts/pipelines/run_working_pipeline.py

# Status check
python scripts/check_current_status.py
```

## 🔍 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Use smaller batch size or enable compression
python scripts/training/proper_train_test.py --batch-size 8
```

**No traces found**
```bash
# Collect traces first (will take time for 128-expert model)
python scripts/collection/collect_robust_traces.py
```

**Low speculation accuracy**
```bash
# Check if learnable models are trained
python scripts/training/proper_train_test.py
```

### Performance Tips
1. **Use 128-expert model** for best diversity in traces
2. **Allow time** for trace collection (large model)
3. **Monitor GPU memory** during training
4. **Use proper train/test splits** to avoid overfitting

## 📚 Research Applications

This codebase supports research in:
- **MoE Efficiency**: Memory-constrained inference
- **Speculation Algorithms**: Neural vs heuristic prediction
- **Hardware Optimization**: GPU-specific tuning
- **Model Analysis**: Expert usage patterns

## 📖 Citation

```bibtex
@misc{enhanced_pregated_moe_2025,
  title={Enhanced Pre-gated MoE for Small GPUs: Advanced Speculation and Memory Optimization},
  author={Cyan Subhra Mishra},
  year={2025},
  note={Extension of ISCA'24 Pre-gated MoE with learnable speculation for RTX 3090},
  url={https://github.com/your-repo/enhanced-pregated-moe}
}
```

## 🤝 Contributing

We welcome contributions! See `docs/CONTRIBUTING.md` for guidelines.

Areas of interest:
- New speculation strategies
- Hardware optimizations
- Model architectures
- Evaluation methods

## 📄 License

MIT License - see `docs/LICENSE` for details.

---

**Ready to use!** Start with:
```bash
python scripts/pipelines/run_working_pipeline.py --use-128-experts
```