# 🚀 Quick Start Guide

## One-Command Setup

```bash
# Complete pipeline with 128-expert model (recommended)
python scripts/pipelines/run_working_pipeline.py --use-128-experts
```

This will:
1. ✅ Collect routing traces from `google/switch-base-128` 
2. ✅ Train speculation models with proper data splits
3. ✅ Test all speculation approaches
4. ✅ Generate comprehensive comparison results

## Step-by-Step Manual Execution

### Step 1: Collect Routing Traces
```bash
# Option A: 128-expert model (best quality, takes time)
python scripts/collection/collect_robust_traces.py

# Option B: Quick working collector (faster)
python scripts/collection/collect_working_final.py
```

**Expected**: This creates traces in `routing_data/` (will be large files)

### Step 2: Train Speculation Models
```bash
python scripts/training/proper_train_test.py
```

**Expected**: Creates trained models in `trained_models/`

### Step 3: Test Individual Approaches
```bash
python scripts/evaluation/test_individual_approaches.py
```

**Expected**: Tests all speculation modes individually

### Step 4: Compare All Approaches
```bash
python scripts/evaluation/compare_all_approaches.py
```

**Expected**: Comprehensive comparison with results

### Step 5: Check Status
```bash
python scripts/check_current_status.py
```

## Demo Applications

After the pipeline completes:

```bash
# Test custom model with best speculation
python main.py --mode demo --speculation-mode learnable

# Test pre-trained Switch Transformer
python main_pretrained.py --mode demo --pretrained-model google/switch-base-8

# Compare all modes
python main.py --mode compare
```

## Current Data Status

You currently have:
- ✅ **6,000 routing traces** in `routing_data/proper_traces.pkl` (188 MB)
- ✅ **Trained models** in `trained_models/`
- ✅ **GPU ready** (0.8% usage - idle)

## Expected Timeline

- **128-expert collection**: 10-30 minutes (large model)
- **Training**: 5-15 minutes
- **Testing**: 2-5 minutes
- **Total**: ~20-50 minutes

## File Structure After Completion

```
routing_data/
├── robust_traces.pkl       # 128-expert traces (new)
└── proper_traces.pkl       # Current traces (6K samples)

trained_models/
├── robust_speculation_model.pt
└── simple_speculation_model.pt

evaluation_results/
├── individual_results.json
└── comparison_results.json
```

## Troubleshooting

**Takes too long?**
```bash
# Use quick version instead
python scripts/pipelines/run_working_pipeline.py
```

**GPU memory issues?**
```bash
# Check GPU status
nvidia-smi

# Run status check
python scripts/check_current_status.py
```

**Missing files?**
Check that the working scripts are in place:
- ✅ `scripts/collection/collect_robust_traces.py`
- ✅ `scripts/training/proper_train_test.py`
- ✅ `scripts/evaluation/test_individual_approaches.py`

The working pipeline uses the **original scripts** that were proven to work with your setup.