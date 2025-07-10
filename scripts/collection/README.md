# Data Collection Scripts

## Working Collectors

### `collect_robust_traces.py`
**Primary collector** - Uses `google/switch-base-128` for high-quality diverse traces.

**Features:**
- ✅ 128 experts (maximum diversity)
- ✅ Real Switch Transformer routing
- ✅ Multiple datasets (WikiText, SQuAD, GLUE)
- ✅ Confirmed working implementation

**Usage:**
```bash
python collect_robust_traces.py
```

**Expected Output:**
- File: `routing_data/robust_traces.pkl`
- Size: ~200-500 MB (depending on samples)
- Time: 10-30 minutes (large model)

### `collect_working_final.py`
**Secondary collector** - Confirmed working, faster alternative.

**Features:**
- ✅ Proven working implementation
- ✅ Faster execution
- ✅ Good for testing/development

**Usage:**
```bash
python collect_working_final.py
```

**Expected Output:**
- File: `routing_data/working_final_traces.pkl`
- Size: ~100-300 MB
- Time: 5-15 minutes

## Data Format

Both collectors produce traces in the same format:
```python
{
    'layer_id': int,
    'hidden_states': torch.Tensor,  # [seq_len, hidden_size]
    'target_routing': torch.Tensor, # [seq_len, num_experts]
    'prev_layer_gates': List[torch.Tensor],
    'sequence_length': int,
    'dataset_name': str,
    'sample_id': str
}
```

## Usage Notes

1. **Use `collect_robust_traces.py`** for final results (128 experts)
2. **Use `collect_working_final.py`** for quick testing
3. **Allow time** for the 128-expert model to load and process
4. **Monitor GPU memory** during collection

Both scripts are restored from the working archive and contain the proven implementations.