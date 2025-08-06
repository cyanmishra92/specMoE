# SpecMoE Running Instructions
===============================================================================

**Complete guide for training, testing, evaluation, and deployment of the SpecMoE expert prefetching framework.**

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
- [Training Neural Predictors](#training-neural-predictors)
- [Running Evaluations](#running-evaluations)
- [Testing & Validation](#testing--validation)
- [Data Collection](#data-collection)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Immediate Evaluation (5 minutes)
```bash
# Clone and setup
git clone https://github.com/research/specMoE.git
cd specMoE
pip install -e .

# Run quick evaluation with existing models
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --strategy intelligent \
    --batch_sizes 1,8,16 \
    --num_runs 3
```

### View Results
```bash
# Check generated results
ls results/switch_transformer/
cat results/switch_transformer/evaluation_summary.json

# View plots
open results/plots/switch_transformer_evaluation.png
```

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
# System requirements
python --version  # >= 3.8
nvidia-smi        # CUDA 11.8+ recommended
free -h           # 16GB+ RAM (32GB+ recommended)
```

### Basic Installation
```bash
# Standard installation
pip install specmoe

# Development installation
git clone https://github.com/research/specMoE.git
cd specMoE
pip install -e ".[dev]"
```

### Full Development Setup
```bash
# Clone repository
git clone https://github.com/research/specMoE.git
cd specMoE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dependencies
pip install -e ".[dev,wandb,jupyter]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import src; print('SpecMoE installed successfully')"
```

### Hardware Setup Verification
```bash
# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Check memory
python scripts/utils/check_system_resources.py
```

---

## 🎯 Training Neural Predictors

### Training Overview

The SpecMoE framework includes several neural prediction models:

| Model | Parameters | Training Time | Best Use Case |
|-------|------------|---------------|---------------|
| **Dense Transformer** | 8.4M | 3.5 hours | Production deployment |
| **Enhanced Model** | 24.5M | 3.0 hours | Maximum accuracy |
| **Lightweight** | 2.1M | 8 minutes | Fast prototyping |

### Data Preparation

#### 1. Collect Routing Traces
```bash
# Switch Transformer traces
python scripts/collection/collect_maximum_real_traces.py \
    --model_name "switch-base-8" \
    --num_samples 5000 \
    --output_dir data/routing_traces/switch/ \
    --max_length 512

# Qwen MoE traces  
python qwen15_moe_a27b/collect_traces.py \
    --model_path "Qwen/Qwen1.5-MoE-A2.7B" \
    --num_samples 5000 \
    --output_dir data/routing_traces/qwen/ \
    --create_shards
```

#### 2. Validate Data Quality
```bash
# Check trace quality
python scripts/analysis/analyze_trace_structure.py \
    --trace_dir data/routing_traces/switch/ \
    --min_sequence_length 32 \
    --min_expert_diversity 8

# Visualize routing patterns
python scripts/analysis/visualize_expert_traces.py \
    --trace_file data/routing_traces/switch/traces.pkl \
    --output_dir results/plots/routing_analysis/
```

### Training Commands

#### 1. Train Dense Transformer Model (Recommended)
```bash
python src/training/train_predictor.py \
    --model_type dense_transformer \
    --data_path data/routing_traces/switch/ \
    --config config/training/dense_transformer.yaml \
    --output_dir models/dense_transformer/ \
    --epochs 120 \
    --batch_size 32 \
    --learning_rate 6e-5 \
    --gpu_id 0
```

#### 2. Train Lightweight Model (Fast)
```bash
python src/training/train_predictor.py \
    --model_type lightweight \
    --data_path data/routing_traces/switch/ \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --output_dir models/lightweight/ \
    --quick_train
```

#### 3. Train Enhanced Model (Maximum Accuracy)
```bash
python src/training/train_predictor.py \
    --model_type enhanced \
    --data_path data/routing_traces/switch/ \
    --epochs 120 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --output_dir models/enhanced/ \
    --use_wandb \
    --project_name "specmoe-enhanced"
```

### Architecture-Specific Training

#### Switch Transformer Training
```bash
# Full training pipeline
python scripts/training/train_switch_predictor.py \
    --trace_file data/routing_traces/switch/maximum_real_traces.pkl \
    --model_config config/models/switch_transformer.yaml \
    --output_dir models/switch_predictor/ \
    --validation_split 0.2 \
    --early_stopping_patience 25
```

#### Qwen MoE Training
```bash
# Multi-expert predictor training
python qwen15_moe_a27b/scripts/train_multi_expert_predictor.py \
    --shard_dir data/routing_traces/qwen/shards/ \
    --model_type multi_expert \
    --output_dir models/qwen_predictor/ \
    --num_epochs 100 \
    --context_length 3 \
    --prediction_horizon 2
```

### Monitoring Training

#### Using Weights & Biases
```bash
# Install wandb
pip install wandb
wandb login

# Train with logging
python src/training/train_predictor.py \
    --model_type dense_transformer \
    --data_path data/routing_traces/switch/ \
    --use_wandb \
    --project_name "specmoe-training" \
    --experiment_name "dense-transformer-v1"
```

#### Manual Monitoring
```bash
# Monitor training logs
tail -f logs/training_*.log

# Check GPU utilization
nvidia-smi -l 1

# Monitor training progress
python scripts/training/monitor_training.py \
    --log_dir logs/ \
    --model_dir models/dense_transformer/
```

### Model Validation

#### Validate Trained Model
```bash
python src/training/validate_model.py \
    --model_path models/dense_transformer/best_model.pth \
    --test_data data/routing_traces/switch/test_traces.pkl \
    --output_dir results/validation/ \
    --compute_detailed_metrics
```

#### Compare Model Variants
```bash
python src/training/compare_models.py \
    --model_paths models/*/best_model.pth \
    --test_data data/routing_traces/switch/test_traces.pkl \
    --output_file results/model_comparison.json
```

---

## 📊 Running Evaluations

### Evaluation Overview

SpecMoE supports multiple evaluation modes:

| Evaluation Type | Command | Duration | Purpose |
|-----------------|---------|----------|---------|
| **Quick Test** | `--num_runs 3` | 5 min | Rapid validation |
| **Standard** | `--num_runs 10` | 30 min | Research results |
| **Publication** | `--num_runs 50` | 2 hours | Statistical significance |

### Single Architecture Evaluation

#### Switch Transformer Evaluation
```bash
# Basic evaluation
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --strategy intelligent,topk,multilook \
    --batch_sizes 1,8,16,32 \
    --cache_size 50 \
    --num_runs 10

# Comprehensive evaluation
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --strategy all \
    --batch_sizes 1,2,4,8,16,32,64 \
    --cache_size 50 \
    --num_runs 10 \
    --verbose
```

#### Qwen MoE Evaluation
```bash
# Standard Qwen evaluation
python -m src.evaluation.run_evaluation \
    --architecture qwen_moe \
    --strategy intelligent,topk \
    --batch_sizes 1,8,16,32 \
    --cache_size 160 \
    --num_runs 10
```

### Comparative Evaluation

#### Compare with Paper Baselines
```bash
# Include paper baseline methods
python -m src.evaluation.run_evaluation \
    --architecture comparative \
    --include_baselines \
    --enable_deduplication \
    --batch_sizes 1,8,16,32,64 \
    --output_dir results/comparative_analysis/
```

#### Cross-Architecture Comparison
```bash
# Run all architectures
python -m src.evaluation.run_evaluation \
    --architecture all \
    --strategy intelligent \
    --batch_sizes 1,8,16,32 \
    --output_dir results/cross_architecture/ \
    --generate_comparison_report
```

### Specialized Evaluations

#### Expert Deduplication Analysis
```bash
# Detailed deduplication study
python -m src.evaluation.run_evaluation \
    --architecture deduplication \
    --batch_sizes 1,2,4,8,16,32,64,128 \
    --output_dir results/deduplication_study/

# Alternative direct script
python expert_deduplication_analysis.py \
    --batch_sizes 1,4,8,16,32,64 \
    --num_experiments 100 \
    --output_dir results/deduplication/
```

#### Hardware-Specific Evaluation
```bash
# Evaluate on specific hardware
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --strategy intelligent \
    --hardware rtx_4090 \
    --batch_sizes 1,8,16,32 \
    --measure_actual_latency

# Multi-GPU evaluation
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --strategy intelligent \
    --multi_gpu \
    --gpu_ids 0,1,2,3
```

### Batch Evaluation Scripts

#### Run Complete Evaluation Suite
```bash
# Run all evaluations (long-running)
bash scripts/evaluation/run_complete_suite.sh

# Run evaluation suite with custom config
python scripts/evaluation/batch_evaluator.py \
    --config config/evaluation/complete_suite.yaml \
    --output_dir results/complete_evaluation/ \
    --parallel_jobs 4
```

#### Custom Evaluation Pipeline
```bash
# Create custom evaluation
python scripts/evaluation/custom_evaluation.py \
    --architectures switch_transformer,qwen_moe \
    --strategies intelligent,topk \
    --batch_sizes 1,8,16,32 \
    --cache_sizes 25,50,100 \
    --hardware rtx_4090,a100 \
    --output_dir results/custom/
```

---

## 🧪 Testing & Validation

### Unit Tests

#### Run All Tests
```bash
# Run complete test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_models.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_strategies.py -v
```

#### Test Categories
```bash
# Model tests
pytest tests/unit/test_neural_predictors.py
pytest tests/unit/test_strategy_implementations.py

# Evaluation tests  
pytest tests/unit/test_cache_framework.py
pytest tests/unit/test_hardware_modeling.py

# Integration tests
pytest tests/integration/test_full_pipeline.py
pytest tests/integration/test_cross_architecture.py
```

### Validation Scripts

#### Validate Installation
```bash
# Basic validation
python scripts/validation/validate_installation.py

# Comprehensive validation
python scripts/validation/validate_complete_setup.py \
    --check_gpu \
    --check_data \
    --check_models \
    --run_quick_test
```

#### Validate Models
```bash
# Test neural predictor models
python scripts/validation/validate_models.py \
    --model_dir models/ \
    --test_data data/routing_traces/switch/test_traces.pkl

# Test strategy implementations  
python scripts/validation/validate_strategies.py \
    --strategies all \
    --quick_test
```

#### Performance Regression Tests
```bash
# Check for performance regressions
python scripts/validation/regression_tests.py \
    --baseline_results results/baselines/ \
    --current_results results/latest/ \
    --tolerance 0.05

# Continuous integration tests
python scripts/validation/ci_tests.py \
    --fast_mode \
    --generate_report
```

---

## 📥 Data Collection

### Routing Trace Collection

#### Switch Transformer Traces
```bash
# Basic trace collection
python scripts/collection/collect_maximum_real_traces.py \
    --model_name "switch-base-8" \
    --dataset "c4" \
    --num_samples 10000 \
    --output_file data/routing_traces/switch_traces.pkl

# Diverse dataset collection
python scripts/collection/collect_robust_traces.py \
    --datasets "c4,openwebtext,bookcorpus" \
    --samples_per_dataset 5000 \
    --output_dir data/routing_traces/diverse/
```

#### Qwen MoE Traces
```bash
# Collect Qwen traces
python qwen15_moe_a27b/collect_traces.py \
    --model_path "Qwen/Qwen1.5-MoE-A2.7B" \
    --dataset_name "c4" \
    --num_samples 10000 \
    --output_dir data/routing_traces/qwen/

# Create training shards
python qwen15_moe_a27b/create_shards_from_existing.py \
    --input_file data/routing_traces/qwen/raw_traces.pkl \
    --output_dir data/routing_traces/qwen/shards/ \
    --traces_per_shard 500
```

#### Custom Model Traces
```bash
# Collect from custom MoE model
python scripts/collection/collect_custom_traces.py \
    --model_path "path/to/your/moe_model" \
    --tokenizer_path "path/to/tokenizer" \
    --num_samples 5000 \
    --output_file data/routing_traces/custom_traces.pkl \
    --batch_size 32
```

### Data Quality Control

#### Validate Collected Data
```bash
# Check trace quality
python scripts/analysis/validate_traces.py \
    --trace_file data/routing_traces/switch_traces.pkl \
    --min_length 32 \
    --min_expert_diversity 8 \
    --output_report data/trace_quality_report.json

# Analyze routing patterns
python scripts/analysis/analyze_routing_patterns.py \
    --trace_file data/routing_traces/switch_traces.pkl \
    --output_dir results/routing_analysis/
```

---

## ⚙️ Configuration

### Configuration Files

SpecMoE uses a hierarchical configuration system:

```bash
config/
├── default.yaml           # Base configuration
├── models/
│   ├── switch_transformer.yaml
│   ├── qwen_moe.yaml
│   └── custom_model.yaml
├── training/
│   ├── dense_transformer.yaml
│   ├── enhanced_model.yaml
│   └── lightweight.yaml
└── evaluation/
    ├── quick_test.yaml
    ├── standard.yaml
    └── publication.yaml
```

### Using Configurations

#### Load Configuration
```python
from src.utils import ConfigManager

# Load default config
config = ConfigManager()

# Load custom config
config = ConfigManager("config/custom.yaml")

# Get architecture-specific settings
arch_config = config.get_architecture_config("switch_transformer")
print(f"Experts: {arch_config.num_experts}")
print(f"Layers: {arch_config.num_layers}")
```

#### Override Configuration
```bash
# Command-line overrides
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --config config/evaluation/quick_test.yaml \
    --override evaluation.num_runs=5 \
    --override hardware.device=rtx_4090
```

#### Create Custom Configuration
```yaml
# config/my_experiment.yaml
architectures:
  my_switch:
    num_experts: 256
    num_layers: 16
    routing_type: "top_1"
    expert_size_mb: 5.0
    default_cache_size_mb: 100.0

evaluation:
  batch_sizes: [1, 4, 16, 64]
  statistical_runs: 20
  hardware_device: "a100"

strategies:
  my_intelligent:
    description: "Custom intelligent strategy"
    cache_enabled: true
    model_path: "models/my_custom_model.pth"
```

### Environment Variables
```bash
# Set environment variables
export SPECMOE_CONFIG_PATH="config/my_experiment.yaml"
export SPECMOE_DATA_DIR="data/"
export SPECMOE_OUTPUT_DIR="results/"
export SPECMOE_MODEL_DIR="models/"
export CUDA_VISIBLE_DEVICES="0,1"
```

---

## 🚀 Deployment

### Production Deployment

#### Docker Deployment
```bash
# Build Docker image
docker build -t specmoe:latest .

# Run evaluation in container
docker run --gpus all -v $(pwd)/results:/app/results \
    specmoe:latest \
    python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --strategy intelligent \
    --batch_sizes 1,8,16,32

# Deploy to production
docker-compose up -d
```

#### Kubernetes Deployment
```bash
# Deploy to Kubernetes cluster
kubectl apply -f deployment/k8s/specmoe-deployment.yaml

# Scale deployment
kubectl scale deployment specmoe --replicas=5

# Check status
kubectl get pods -l app=specmoe
```

### Cloud Deployment

#### AWS Deployment
```bash
# Launch EC2 instance with GPU
aws ec2 run-instances \
    --image-id ami-xxxxxxxxx \
    --instance-type p3.2xlarge \
    --key-name my-key \
    --security-groups sg-xxxxxxxxx

# Deploy using AWS Batch
aws batch submit-job \
    --job-name specmoe-evaluation \
    --job-queue specmoe-queue \
    --job-definition specmoe-job-def
```

#### Google Cloud Deployment
```bash
# Create GKE cluster with GPUs
gcloud container clusters create specmoe-cluster \
    --accelerator type=nvidia-tesla-v100,count=1 \
    --num-nodes=3

# Deploy workload
kubectl apply -f deployment/gcp/specmoe-workload.yaml
```

### Edge Deployment

#### Jetson Deployment
```bash
# Install on Jetson device
pip install specmoe[jetson]

# Run optimized evaluation
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --strategy lightweight \
    --hardware jetson_agx_orin \
    --optimization_level high
```

---

## 🔧 Troubleshooting

### Common Issues

#### Installation Issues
```bash
# Issue: CUDA not detected
nvidia-smi  # Check GPU availability
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Issue: Memory errors during training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python scripts/utils/check_memory_usage.py

# Issue: ImportError
pip install -e .  # Reinstall in development mode
python -c "import sys; print('\n'.join(sys.path))"  # Check Python path
```

#### Training Issues
```bash
# Issue: Training stuck or slow
nvidia-smi  # Check GPU utilization
htop  # Check CPU usage
python scripts/training/diagnose_training.py --model_dir models/current/

# Issue: NaN losses
python scripts/training/debug_training.py \
    --model_path models/problematic_model.pth \
    --check_gradients \
    --check_weights

# Issue: Out of memory
python scripts/training/optimize_memory.py \
    --batch_size 16 \  # Reduce batch size
    --gradient_checkpointing \
    --mixed_precision
```

#### Evaluation Issues
```bash
# Issue: Evaluation crashes
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --strategy intelligent \
    --debug_mode \
    --verbose

# Issue: Inconsistent results
python scripts/validation/check_reproducibility.py \
    --config_file config/evaluation/standard.yaml \
    --num_runs 5

# Issue: Missing results
ls -la results/  # Check output directory
python scripts/utils/diagnose_evaluation.py \
    --output_dir results/latest/
```

### Performance Optimization

#### Speed Up Training
```bash
# Use mixed precision
python src/training/train_predictor.py \
    --mixed_precision \
    --compile_model

# Use multiple GPUs
python src/training/train_predictor.py \
    --distributed \
    --gpu_ids 0,1,2,3

# Optimize data loading
python src/training/train_predictor.py \
    --num_workers 8 \
    --prefetch_factor 4
```

#### Speed Up Evaluation
```bash
# Parallel evaluation
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --parallel_strategies \
    --num_workers 4

# Reduce precision for speed
python -m src.evaluation.run_evaluation \
    --architecture switch_transformer \
    --fast_mode \
    --reduced_precision
```

### Getting Help

#### Debug Information
```bash
# Generate debug report
python scripts/utils/generate_debug_report.py \
    --include_system_info \
    --include_model_info \
    --include_recent_logs \
    --output_file debug_report.txt

# Check system compatibility
python scripts/utils/check_compatibility.py
```

#### Logging
```bash
# Enable verbose logging
export SPECMOE_LOG_LEVEL=DEBUG
python -m src.evaluation.run_evaluation --verbose

# Check log files
tail -f logs/specmoe.log
grep ERROR logs/specmoe.log
```

#### Community Support
- 📧 **Email**: [support@specmoe.ai]
- 💬 **Discord**: [SpecMoE Community]
- 🐛 **Issues**: [GitHub Issues](https://github.com/research/specMoE/issues)
- 📖 **Documentation**: [Full Docs](https://docs.specmoe.ai)

---

## 📚 Additional Resources

### Example Scripts
- **Quick Demo**: `examples/quick_demo.py`
- **Full Pipeline**: `examples/full_pipeline.py`  
- **Custom Strategy**: `examples/custom_strategy.py`
- **Batch Processing**: `examples/batch_processing.py`

### Configuration Templates
- **Research Setup**: `config/templates/research.yaml`
- **Production Setup**: `config/templates/production.yaml`
- **Development Setup**: `config/templates/development.yaml`

### Performance Benchmarks
- **Hardware Comparison**: `benchmarks/hardware_comparison.md`
- **Strategy Comparison**: `benchmarks/strategy_comparison.md`
- **Scaling Analysis**: `benchmarks/scaling_analysis.md`

---

**Last Updated**: August 6, 2025  
**Version**: 1.0.0  
**Tested Platforms**: Linux (Ubuntu 20.04+), macOS (12+), Windows 10+