# SpecMoE Quick Reference Card

## 🚀 Essential Commands

### Installation
```bash
pip install -e .                    # Development install
python scripts/validation/validate_installation.py  # Verify setup
```

### Quick Start
```bash
python examples/quick_demo.py       # 5-minute demo
python -m src.evaluation.run_evaluation --architecture switch_transformer --strategy intelligent
```

### Evaluation
```bash
# Single architecture
python -m src.evaluation.run_evaluation --architecture switch_transformer --strategy intelligent --batch_sizes 1,8,16,32

# All architectures  
python -m src.evaluation.run_evaluation --architecture all --strategy intelligent

# Comparative analysis
python -m src.evaluation.run_evaluation --architecture comparative --include_baselines --enable_deduplication

# Expert deduplication
python -m src.evaluation.run_evaluation --architecture deduplication --batch_sizes 1,8,16,32,64
```

### Training
```bash
# Train neural predictor
python src/training/train_predictor.py --model_type dense_transformer --data_path data/routing_traces/

# Switch Transformer model
python scripts/training/train_switch_predictor.py --trace_file data/routing_traces/switch/traces.pkl

# Qwen MoE model  
python qwen15_moe_a27b/scripts/train_multi_expert_predictor.py --shard_dir data/routing_traces/qwen/shards/
```

### Data Collection
```bash
# Switch Transformer traces
python scripts/collection/collect_maximum_real_traces.py --model_name "switch-base-8" --num_samples 5000

# Qwen MoE traces
python qwen15_moe_a27b/collect_traces.py --model_path "Qwen/Qwen1.5-MoE-A2.7B" --num_samples 5000
```

### Testing
```bash
pytest tests/ -v                    # Run all tests
python scripts/validation/validate_models.py  # Validate models
```

## 📊 Key Results Summary

| Architecture | Experts | Routing | Best Strategy | Speedup | Hit Rate |
|--------------|---------|---------|---------------|---------|----------|
| Switch Transformer | 128 | Top-1 | Intelligent | **13.07×** | 99.43% |
| Qwen MoE | 64 | Top-8 | Intelligent | **1.62×** | 96.9% |
| Comparative | Various | Mixed | Deduplication | **1.29×** | 98%+ |

## ⚙️ Configuration

### Available Architectures
- `switch_transformer` - 128 experts, top-1 routing
- `qwen_moe` - 64 experts, top-8 routing  
- `comparative` - Cross-architecture comparison
- `deduplication` - Expert deduplication analysis

### Available Strategies
- `on_demand` - Baseline (no prefetching)
- `oracle` - Perfect prediction (upper bound)
- `topk` - Frequency-based caching
- `multilook` - Pattern recognition
- `intelligent` - Neural prediction (best)
- `intelligent_dedup` - Neural + deduplication

### Available Hardware
- `rtx_4090` - 24GB, consumer GPU
- `a100_80gb` - 80GB, enterprise GPU
- `h100_80gb` - 80GB, latest enterprise
- `jetson_agx_orin` - 32GB, edge device

## 📁 Directory Structure

```
specMoE/
├── src/                    # Core source code
│   ├── models/            # Prediction models & strategies  
│   ├── evaluation/        # Evaluation frameworks
│   ├── training/          # Training infrastructure
│   └── utils/             # Utilities & config
├── experiments/           # Architecture experiments
│   ├── switch_transformer/
│   ├── qwen_moe/
│   └── comparative/
├── results/               # Generated results
│   ├── data/             # JSON/CSV results
│   ├── plots/            # Visualizations
│   └── models/           # Trained models
├── config/               # Configuration files
│   ├── default.yaml
│   └── templates/
├── scripts/              # Utility scripts
│   ├── evaluation/
│   ├── training/
│   └── validation/
└── docs/                 # Documentation
```

## 🔧 Common Flags

### Evaluation Flags
```bash
--architecture {switch_transformer,qwen_moe,comparative,deduplication,all}
--strategy {on_demand,oracle,topk,multilook,intelligent,intelligent_dedup,all}
--batch_sizes "1,8,16,32"              # Comma-separated list
--cache_size 50                        # Cache size in MB  
--num_runs 10                          # Statistical runs
--hardware rtx_4090                    # Hardware config
--include_baselines                    # Include paper baselines
--enable_deduplication                 # Enable expert deduplication
--verbose                              # Detailed output
--output_dir results/my_experiment/    # Custom output directory
```

### Training Flags
```bash
--model_type {dense_transformer,enhanced,lightweight}
--data_path data/routing_traces/       # Training data directory
--epochs 120                           # Training epochs
--batch_size 32                        # Training batch size
--learning_rate 6e-5                   # Learning rate
--output_dir models/my_model/          # Model output directory
--use_wandb                           # Enable W&B logging
--gpu_id 0                            # GPU device ID
```

## 🚨 Troubleshooting

### Installation Issues
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # Fix CUDA
pip install -e .                       # Reinstall SpecMoE
python -c "import torch; print(torch.cuda.is_available())"  # Check CUDA
```

### Training Issues
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Fix OOM
nvidia-smi                             # Check GPU usage
python scripts/training/debug_training.py --model_dir models/current/
```

### Evaluation Issues
```bash
python -m src.evaluation.run_evaluation --debug_mode --verbose  # Debug mode
python scripts/validation/check_reproducibility.py  # Check consistency
ls -la results/                        # Check output files
```

## 📞 Getting Help

- **Validation**: `python scripts/validation/validate_installation.py`
- **Demo**: `python examples/quick_demo.py`
- **Full Guide**: `cat RUNNING_INSTRUCTIONS.md`
- **Issues**: [GitHub Issues](https://github.com/research/specMoE/issues)

## 🎯 Next Steps After Setup

1. **Validate**: `python scripts/validation/validate_installation.py`
2. **Demo**: `python examples/quick_demo.py`  
3. **Quick Eval**: `python -m src.evaluation.run_evaluation --architecture switch_transformer --strategy intelligent --batch_sizes 1,8`
4. **Full Eval**: `bash scripts/evaluation/run_complete_suite.sh`
5. **Train Model**: `python src/training/train_predictor.py --model_type dense_transformer`

---

*For complete instructions, see `RUNNING_INSTRUCTIONS.md`*