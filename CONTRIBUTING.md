# Contributing to Enhanced Pre-gated MoE

We welcome contributions to enhance the speculation and memory optimization capabilities of this MoE implementation!

## 🎯 Areas of Interest

### High Priority
1. **Speculation Algorithms**: New expert prediction strategies
2. **Memory Optimization**: More efficient compression and caching
3. **Hardware Support**: Optimization for different GPU architectures
4. **Model Coverage**: Support for larger MoE models

### Medium Priority
1. **Performance Optimization**: CUDA kernels and Triton integration
2. **Evaluation Tools**: Better benchmarking and analysis
3. **Documentation**: Tutorials and examples
4. **Testing**: Unit tests and integration tests

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+
- Git

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/specMoE.git
cd specMoE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_speculation_engine.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality
```bash
# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8 .
```

## 📝 Contribution Guidelines

### 1. **Code Style**
- Use Black for code formatting
- Follow PEP 8 conventions
- Add type hints for all functions
- Write descriptive docstrings

### 2. **Testing**
- Add tests for new features
- Ensure all tests pass before submitting
- Aim for >80% code coverage
- Include integration tests for new components

### 3. **Documentation**
- Update README.md for new features
- Add docstrings to all public functions
- Include usage examples
- Update CHANGELOG.md

### 4. **Performance**
- Benchmark new features
- Include performance comparisons
- Document memory usage impacts
- Test on different hardware when possible

## 🔄 Development Workflow

### 1. **Issue Creation**
- Search existing issues first
- Use issue templates when available
- Provide clear reproduction steps
- Include system information (GPU, CUDA version, etc.)

### 2. **Branch Strategy**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new speculation algorithm"

# Push and create PR
git push origin feature/your-feature-name
```

### 3. **Pull Request Process**
- Fill out the PR template completely
- Include benchmarks for performance changes
- Add tests for new functionality
- Update documentation as needed
- Ensure CI passes

### 4. **Review Process**
- All PRs require at least one review
- Address reviewer feedback promptly
- Maintain clean commit history
- Squash commits before merging

## 🧪 Adding New Features

### New Speculation Mode
```python
# 1. Add to SpeculationMode enum
class SpeculationMode(Enum):
    YOUR_MODE = "your_mode"

# 2. Implement prediction method
def _predict_your_mode(self, target_layer: int) -> Tuple[torch.Tensor, float]:
    # Your implementation here
    return predicted_probs, confidence

# 3. Add to prediction dispatcher
def predict_next_experts(self, ...):
    if self.speculation_mode == SpeculationMode.YOUR_MODE:
        return self._predict_your_mode(target_layer)

# 4. Add tests
def test_your_mode_prediction():
    # Test implementation
    pass
```

### New Compression Method
```python
# 1. Add to CompressionType enum
class CompressionType(Enum):
    YOUR_COMPRESSION = "your_compression"

# 2. Implement compress/decompress
@staticmethod
def _compress_your_method(weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    # Compression logic
    return compressed_weights, metadata

@staticmethod
def _decompress_your_method(compressed: torch.Tensor, metadata: Dict) -> torch.Tensor:
    # Decompression logic
    return weights

# 3. Add to compression dispatcher
def compress(weights, compression_type):
    if compression_type == CompressionType.YOUR_COMPRESSION:
        return ExpertCompressor._compress_your_method(weights)
```

### New Hardware Profile
```python
# Add to DeviceProfiler
def _get_your_gpu_profile(self, memory_gb, bandwidth, compute_cap):
    return DeviceProfile(
        device_name="Your GPU",
        memory_capacity_gb=memory_gb,
        # ... other parameters
    )
```

## 📊 Benchmarking New Features

### Performance Testing
```bash
# Benchmark your changes
python main.py --mode benchmark --benchmark-iterations 50

# Compare with baseline
python scripts/compare_performance.py --baseline main --candidate your-branch

# Memory profiling
python scripts/profile_memory.py --mode your-feature
```

### Include Results
- Add benchmark results to PR description
- Include memory usage comparisons
- Test on different batch sizes and sequence lengths
- Document any performance regressions

## 🐛 Bug Reports

### Information to Include
1. **System Information**:
   - GPU model and VRAM
   - CUDA version
   - PyTorch version
   - Python version

2. **Reproduction Steps**:
   - Exact command that failed
   - Input parameters
   - Expected vs. actual behavior

3. **Error Information**:
   - Full error traceback
   - Log files if available
   - Memory usage when error occurred

### Bug Fix Guidelines
- Add regression tests
- Document root cause in commit message
- Test fix on multiple configurations
- Update documentation if needed

## 💡 Feature Requests

### Proposal Format
1. **Problem Statement**: What issue does this address?
2. **Proposed Solution**: How would you solve it?
3. **Alternatives Considered**: What other approaches exist?
4. **Implementation Plan**: High-level approach
5. **Testing Strategy**: How to validate the feature

### Evaluation Criteria
- Alignment with project goals
- Performance impact
- Maintenance burden
- Compatibility with existing features

## 🏷️ Release Process

### Version Numbering
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Performance benchmarks run
- [ ] Version number bumped
- [ ] Git tag created

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: For sensitive issues or collaboration

### Response Times
- Bug reports: 1-2 days
- Feature requests: 1 week
- Questions: 2-3 days

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Celebrate contributions

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation
- Invited to maintainer discussions (for significant contributions)

Thank you for contributing to Enhanced Pre-gated MoE! 🚀