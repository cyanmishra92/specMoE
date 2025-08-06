# Experiments Directory

Experiment-specific code and configurations for different MoE architectures.

## Structure
- `switch_transformer/` - Switch Transformer experiments
- `qwen_moe/` - Qwen MoE experiments  
- `comparative/` - Cross-architecture comparisons

## Usage
Run experiments using the unified evaluation framework:
```bash
python -m src.evaluation.run_evaluation --architecture switch_transformer
```
