# Src Directory

Core source code for SpecMoE expert prefetching framework.

## Structure
- `models/` - Neural prediction models and prefetching strategies
- `evaluation/` - Evaluation frameworks and metrics
- `training/` - Model training infrastructure  
- `data/` - Data processing utilities
- `utils/` - Common utilities and helpers

## Usage
Import modules using standard Python imports:
```python
from src.models import InterLayerSpeculationModel
from src.evaluation import IsoCacheFramework
```
