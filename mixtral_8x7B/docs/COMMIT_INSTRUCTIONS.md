# Commit Instructions for Mixtral 8x7B MoE

## Files to Commit

### Core Implementation
- `scripts/collection/collect_mixtral_traces.py` - Main trace collection script
- `scripts/training/train_mixtral_speculation.py` - Training script
- `scripts/analysis/visualize_mixtral_routing.py` - Analysis and visualization

### Documentation
- `README.md` - Project overview
- `docs/GETTING_STARTED.md` - Setup guide
- `docs/TECHNICAL_DETAILS.md` - Deep technical details
- `docs/DEPLOYMENT_GUIDE.md` - A100/A6000 deployment guide

### Environment Setup
- `setup_environment.sh` - Automated environment setup
- `run_collection.sh` - Collection runner script
- `requirements.txt` - Python dependencies

### Configuration
- `.gitignore` - Ignore patterns (add if needed)

## Commit Message

```
feat: Add Mixtral 8x7B MoE expert speculation training

- Complete trace collection system for Mixtral 8x7B
- RTX 3090/A6000/A100 GPU optimization
- Real MoE routing extraction (no synthetic traces)
- Support for both top-1 and top-2 routing
- Automatic fallback to Switch Transformer
- Comprehensive documentation and deployment guide
- Automated environment setup scripts

Features:
- 4-bit quantization for memory efficiency
- CPU offloading for smaller GPUs
- Multi-dataset trace collection (8 datasets)
- Expert routing visualization
- Statistics-aware model training
- Performance monitoring and logging

Hardware support:
- RTX 3090 (24GB) - minimum with CPU offload
- A6000 (48GB) - recommended
- A100 (40GB/80GB) - optimal

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Git Commands

```bash
# Navigate to project root
cd /data/research/specMoE/specMoE

# Add all Mixtral files
git add mixtral_8x7B/

# Check what's being committed
git status

# Commit with message
git commit -m "$(cat <<'EOF'
feat: Add Mixtral 8x7B MoE expert speculation training

- Complete trace collection system for Mixtral 8x7B
- RTX 3090/A6000/A100 GPU optimization
- Real MoE routing extraction (no synthetic traces)
- Support for both top-1 and top-2 routing
- Automatic fallback to Switch Transformer
- Comprehensive documentation and deployment guide
- Automated environment setup scripts

Features:
- 4-bit quantization for memory efficiency
- CPU offloading for smaller GPUs
- Multi-dataset trace collection (8 datasets)
- Expert routing visualization
- Statistics-aware model training
- Performance monitoring and logging

Hardware support:
- RTX 3090 (24GB) - minimum with CPU offload
- A6000 (48GB) - recommended
- A100 (40GB/80GB) - optimal

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# Push to remote
git push origin main
```

## Deployment on A100/A6000

### 1. Clone Repository
```bash
git clone <repository-url>
cd specMoE/mixtral_8x7B
```

### 2. Environment Setup
```bash
# Run automated setup
./setup_environment.sh

# Activate environment
conda activate mixtral_moe
```

### 3. Authentication
```bash
# Login to HuggingFace
huggingface-cli login

# Request access to Mixtral models if needed
# https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
```

### 4. Run Collection
```bash
# Simple run
./run_collection.sh

# Or manual run with monitoring
python scripts/collection/collect_mixtral_traces.py
```

### 5. Expected Results
- `routing_data/mixtral_8x7b_traces.pkl` - 45,000+ traces
- `routing_data/mixtral_8x7b_traces.json` - metadata
- `logs/collection_*.log` - detailed logs

## File Structure After Deployment

```
mixtral_8x7B/
├── scripts/
│   ├── collection/collect_mixtral_traces.py
│   ├── training/train_mixtral_speculation.py
│   └── analysis/visualize_mixtral_routing.py
├── docs/
│   ├── GETTING_STARTED.md
│   ├── TECHNICAL_DETAILS.md
│   └── DEPLOYMENT_GUIDE.md
├── routing_data/           # Generated during collection
│   ├── mixtral_8x7b_traces.pkl
│   └── mixtral_8x7b_traces.json
├── models/                 # Generated during training
├── logs/                   # Generated during runs
├── setup_environment.sh    # Environment setup
├── run_collection.sh      # Collection runner
├── requirements.txt       # Dependencies
└── README.md             # Project overview
```

## Testing Checklist

Before pushing, verify:
- [ ] All scripts are executable
- [ ] Documentation is complete
- [ ] Requirements are up to date
- [ ] No hardcoded paths
- [ ] GPU memory requirements documented
- [ ] Error handling is comprehensive
- [ ] Logging is informative

## Performance Expectations

| GPU | Expected Time | Memory Usage | Trace Count |
|-----|---------------|--------------|-------------|
| A100 80GB | 1-2 hours | 35-40GB | 45,000+ |
| A100 40GB | 2-3 hours | 30-35GB | 45,000+ |
| A6000 48GB | 3-4 hours | 40-45GB | 45,000+ |
| RTX 3090 | 5-6 hours | 20GB + CPU | 30,000+ |

## Troubleshooting

Common issues and solutions documented in `docs/DEPLOYMENT_GUIDE.md`:
- Memory errors → CPU offload
- Authentication issues → HuggingFace login
- Slow performance → Check GPU utilization
- Model loading failures → Check quantization settings