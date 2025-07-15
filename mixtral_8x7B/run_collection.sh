#!/bin/bash
# Mixtral 8x7B Trace Collection Runner
# Optimized for A100/A6000 deployment

set -e

echo "🚀 Starting Mixtral 8x7B MoE Trace Collection"
echo "============================================="

# Check if in correct directory
if [ ! -f "scripts/collection/collect_mixtral_traces.py" ]; then
    echo "❌ Error: Must run from mixtral_8x7B directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if conda environment exists
if ! conda info --envs | grep -q "mixtral_moe"; then
    echo "❌ Error: conda environment 'mixtral_moe' not found"
    echo "Please run: ./setup_environment.sh"
    exit 1
fi

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mixtral_moe

# Check GPU
echo "📊 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Check HuggingFace authentication
echo "🤗 Checking HuggingFace authentication..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "❌ Error: Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    exit 1
fi

# Create log directory
mkdir -p logs

# Set environment variables for optimal performance
# Note: CUDA_VISIBLE_DEVICES is set automatically by the script
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Run collection with logging
echo "📊 Starting trace collection..."
echo "Output will be logged to: logs/collection_$(date +%Y%m%d_%H%M%S).log"

python scripts/collection/collect_mixtral_traces.py 2>&1 | tee logs/collection_$(date +%Y%m%d_%H%M%S).log

# Check if collection was successful
if [ -f "routing_data/mixtral_8x7b_traces.pkl" ]; then
    echo "✅ Collection successful!"
    echo "📊 Trace file: routing_data/mixtral_8x7b_traces.pkl"
    echo "📄 Metadata: routing_data/mixtral_8x7b_traces.json"
    
    # Show file sizes
    echo "📈 File sizes:"
    ls -lh routing_data/mixtral_8x7b_traces.*
    
    # Show basic stats
    echo "📊 Basic statistics:"
    python -c "
import pickle
with open('routing_data/mixtral_8x7b_traces.pkl', 'rb') as f:
    traces = pickle.load(f)
print(f'Total traces: {len(traces):,}')
print(f'Unique datasets: {len(set(t.dataset_name for t in traces))}')
print(f'Unique layers: {len(set(t.layer_id for t in traces))}')
"
else
    echo "❌ Collection failed - no trace file found"
    exit 1
fi

echo ""
echo "🎯 Next steps:"
echo "1. Analyze traces: python scripts/analysis/visualize_mixtral_routing.py"
echo "2. Train models: python scripts/training/train_mixtral_speculation.py"
echo "3. View logs: tail -f logs/collection_*.log"
echo ""
echo "🎉 Mixtral 8x7B trace collection complete!"