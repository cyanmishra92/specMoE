# Expert Analysis Guide

This guide covers the comprehensive expert analysis system for understanding MoE routing patterns and statistics.

## 🎯 Overview

The expert analysis system provides detailed statistical analysis of expert usage patterns across different layers in MoE models. It calculates comprehensive metrics and generates visualizations to understand expert specialization and routing behavior.

## 📊 Statistics Calculated

For each layer, the system calculates:

### Basic Statistics
- **Mean**: Average expert usage
- **Standard Deviation**: Variability in expert usage
- **Coefficient of Variation**: Normalized variability (std/mean)
- **Median**: Middle value of expert usage
- **Min/Max Usage**: Range of expert usage

### Distribution Analysis
- **Skewness**: Asymmetry of the distribution
- **Kurtosis**: Tail heaviness of the distribution
- **Q25/Q75**: Quartiles for distribution shape
- **IQR**: Interquartile range

### Information Theory
- **Entropy (bits)**: Information content / diversity measure
- **Active Experts**: Number of experts with non-zero usage
- **Expert Diversity**: Percentage of experts actively used

## 🚀 Quick Start

### Basic Analysis
```bash
# Analyze existing traces
python scripts/analysis/comprehensive_expert_analysis.py

# Create small experimental dataset first
python scripts/analysis/create_small_dataset.py

# Visualize expert traces
python scripts/analysis/visualize_expert_traces.py
```

### Analysis Workflow
1. **Collect diverse traces**: Use `collect_maximum_real_traces.py` to gather routing data from 60+ datasets
2. **Analyze statistics**: Run `comprehensive_expert_analysis.py` for detailed metrics
3. **Visualize patterns**: Use `visualize_expert_traces.py` for trace visualization

### Basic Analysis
```bash
# 1. FIRST: Collect diverse real traces (RECOMMENDED)
python scripts/collection/collect_maximum_real_traces.py

# 2. Analyze comprehensive statistics with layer-wise metrics
python scripts/analysis/comprehensive_expert_analysis.py

# 3. Visualize expert traces and routing patterns
python scripts/analysis/visualize_expert_traces.py

# Alternative: Create small experimental dataset for testing
python scripts/analysis/create_small_dataset.py
```

## 📈 Generated Outputs

### Statistical Reports
- `comprehensive_expert_statistics.json`: Complete statistics for all layers
- `proper_3d_statistics.json`: 3D structure (layer → expert → frequency)
- `sample_trace_paths.json`: Sample expert sequences for visualization

### Visualizations
- `layer_X_expert_frequency.png`: Expert usage frequency for each layer
- `comparative_layer_statistics.png`: Cross-layer statistical comparison
- `expert_selection_traces.png`: Line plots of expert sequences
- `expert_transition_heatmap.png`: Expert transition patterns
- `expert_usage_timeline.png`: Timeline of expert usage

## 🔍 Understanding the Results

### Layer Patterns
- **Layer 1**: High diversity, exploration phase
- **Layer 3**: Specialization begins, some experts dominate
- **Layer 5**: Moderate specialization
- **Layer 7**: Task-specific expert selection
- **Layer 9**: Deep specialization
- **Layer 11**: Final routing decisions

### Key Metrics Interpretation
- **Low Entropy**: Highly specialized, few experts dominate
- **High Entropy**: Diverse usage, many experts active
- **High Skewness**: Heavily skewed toward few experts
- **Low CV**: Uniform usage across experts
- **High CV**: Uneven expert utilization

### Expert Transitions
- **High Transition Rate**: Frequent expert switching (good for prediction)
- **Low Persistence**: Experts rarely used consecutively
- **Transition Diversity**: Variety in expert switching patterns

## 📋 Example Results

From small experimental dataset:
```
Layer 1: Mean=1.60, Std=1.33, CV=0.831, Entropy=6.40, Active=95/128 (74.2%)
Layer 3: Mean=1.60, Std=3.24, CV=2.022, Entropy=4.83, Active=33/128 (25.8%)
Layer 11: Mean=1.60, Std=1.61, CV=1.004, Entropy=6.20, Active=88/128 (68.8%)
```

**Interpretation**: Layer 3 shows highest specialization (lowest entropy, highest CV), while Layers 1 and 11 show more diverse routing patterns.

## 🛠️ Advanced Usage

### Custom Analysis
```python
# Load and analyze specific traces
from scripts.analysis.comprehensive_expert_analysis import *

traces = load_traces("routing_data/my_traces.pkl")
stats, usage = analyze_layer_statistics(traces)
create_expert_frequency_plots(usage, stats)
```

### Trace Collection Integration
```bash
# Collect traces from all 170+ papers
python scripts/collection/collect_robust_traces.py --traces 10000 --mode real

# Analyze the collected traces
python scripts/analysis/comprehensive_expert_analysis.py
```

## 📊 Data Structure

### Trace Format
```json
{
  "layer_id": 1,
  "target_routing": "shape: (seq_len, 128)",
  "target_top_k": "shape: (seq_len, 1)",
  "sample_id": "dataset_sample_layer_1",
  "sequence_length": 32
}
```

### Statistics Format
```json
{
  "layer_id": 1,
  "mean": 1.60,
  "std": 1.33,
  "coefficient_of_variation": 0.831,
  "entropy_bits": 6.40,
  "active_experts": 95,
  "expert_diversity": 0.742,
  "skewness": 0.637,
  "kurtosis": 0.104
}
```

## 🔧 Troubleshooting

### Common Issues
1. **Large files**: Use small experimental dataset for testing
2. **Memory issues**: Process traces in batches
3. **Missing visualizations**: Check matplotlib backend settings

### Performance Tips
- Use `--mode real` for authentic routing patterns
- Collect 3000+ traces for reliable statistics
- Clean old traces before new collection

## 🎯 Next Steps

1. **Collect large dataset**: Run with 10,000+ traces from all papers
2. **Train speculation model**: Use trace statistics for model training
3. **Evaluate predictions**: Compare predicted vs actual routing patterns
4. **Optimize prefetching**: Use expert statistics for memory management