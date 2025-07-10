# Pipeline Scripts

## Main Pipeline

### `run_working_pipeline.py`
Complete automated pipeline using the proven working scripts.

**Usage:**
```bash
# Full pipeline with 128-expert model
python run_working_pipeline.py --use-128-experts

# Quick pipeline with working collector
python run_working_pipeline.py

# Specific steps only
python run_working_pipeline.py --steps 1 2 3
```

**Options:**
- `--use-128-experts`: Use `google/switch-base-128` for trace collection
- `--force-recollect`: Force re-collection even if traces exist
- `--steps 1 2 3 4 5`: Run specific pipeline steps

**Steps:**
1. **Trace Collection**: Uses working collectors
2. **Training**: Proper train/test splits
3. **Individual Testing**: Test each approach
4. **Comparison**: Comprehensive evaluation
5. **Status Check**: Final verification

## Expected Behavior

The pipeline uses the **original working scripts** that were previously tested:
- `collect_robust_traces.py` - 128-expert Switch Transformer
- `proper_train_test.py` - Robust training with no data leakage
- `test_individual_approaches.py` - Individual algorithm testing
- `compare_all_approaches.py` - Full comparison

These scripts were moved from `archive_experimental/` because they contained the working implementation.