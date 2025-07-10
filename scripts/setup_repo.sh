#!/bin/bash

# Setup script for creating the specMoE GitHub repository
# Run this script from the project root directory

set -e

echo "Setting up Enhanced Pre-gated MoE repository..."

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "models" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Make sure you have README.md and models/ directory"
    exit 1
fi

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    
    # Set main as default branch
    git branch -M main
fi

# Add all files
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "feat: initial commit - Enhanced Pre-gated MoE for RTX 3090

- Complete implementation of enhanced speculative gating
- Support for both pre-trained Switch Transformers and custom models  
- RTX 3090 optimizations with automatic device profiling
- Advanced memory management with INT8/INT4 compression
- Comprehensive benchmarking and evaluation suite
- Multi-layer speculation with confidence-based adaptation

Key features:
- Pre-trained model support (Google Switch Transformers)
- Custom 140M parameter model for experimentation
- 4x memory compression with minimal accuracy loss
- Adaptive buffering and hierarchical caching
- Hardware-aware optimization for RTX 3090
- Complete benchmarking infrastructure"

echo ""
echo "Repository setup complete!"
echo ""
echo "Next steps to create GitHub repository:"
echo "1. Go to https://github.com/new"
echo "2. Repository name: specMoE"
echo "3. Description: Enhanced Pre-gated MoE for RTX 3090: Advanced Speculation and Memory Optimization"
echo "4. Make it public"
echo "5. Do NOT initialize with README, .gitignore, or license (we already have them)"
echo "6. Click 'Create repository'"
echo ""
echo "Then run these commands to push to GitHub:"
echo "git remote add origin https://github.com/YOURUSERNAME/specMoE.git"
echo "git push -u origin main"
echo ""
echo "Replace YOURUSERNAME with your actual GitHub username"