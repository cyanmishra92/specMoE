# Switch Transformer Prefetching Evaluation Suite

## Overview

Comprehensive evaluation of 5 prefetching strategies across 5 batch sizes (5×5 = 25 experiments) for Switch Transformer inference optimization.

## Experiment Matrix

### **Prefetching Strategies:**
- **A**: On-demand loading from CPU memory (baseline)
- **B**: Oracle one layer ahead prefetching (100% accuracy) 
- **C**: Our prefetching method with multi look-ahead
- **D**: Multi look-ahead with top-k expert loading (k=10 per layer)
- **E**: Strategy D + intelligent caching

### **Batch Sizes:** 1, 2, 4, 8, 16

### **Metrics:**
- Inference latency (mean ± std)
- Memory usage
- Cache hit rates
- Expert transfer times
- GPU utilization

## File Structure

```
evalSwitchB8/
├── configs/
│   ├── strategy_a_ondemand.json
│   ├── strategy_b_oracle.json
│   ├── strategy_c_multilook.json
│   ├── strategy_d_topk.json
│   └── strategy_e_intelligent.json
├── scripts/
│   ├── run_single_experiment.py      # Core experiment runner
│   ├── run_batch_evaluation.py       # Run all 25 experiments
│   ├── analyze_results.py            # Statistical analysis
│   └── generate_csv_report.py        # CSV/Excel output
├── results/
│   └── [experiment outputs]
└── README.md
```

## Usage

1. **Run single experiment:**
   ```bash
   python scripts/run_single_experiment.py --strategy A --batch-size 1 --runs 10
   ```

2. **Run all experiments:**
   ```bash
   python scripts/run_batch_evaluation.py --runs 10
   ```

3. **Generate analysis:**
   ```bash
   python scripts/analyze_results.py --output results/switch_analysis.csv
   ```

## Expected Runtime

- Single experiment: ~2-5 minutes (10 runs)
- Full evaluation: ~2-4 hours (25 experiments × 10 runs)
- Analysis: ~1 minute

## Hardware Requirements

- GPU: RTX 3090 (24GB) or equivalent
- RAM: 32GB+ recommended
- Storage: 10GB for traces and results