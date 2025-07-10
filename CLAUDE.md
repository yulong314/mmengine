# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMEngine is OpenMMLab's foundational library for training deep learning models based on PyTorch. It serves as the training engine for all OpenMMLab codebases and supports hundreds of algorithms in various research areas.

## PyTorch 2.7+ Compatibility

**Note**: This repository has been modified to support PyTorch 2.7+ compatibility. The original MMEngine supports PyTorch versions `>=1.6 <=2.1`, but this version includes fixes for PyTorch 2.7+ compatibility issues related to JIT compilation in distributed optimizers.

### Changes Made
- Modified `mmengine/optim/optimizer/zero_optimizer.py` to handle PyTorch 2.7+ JIT compilation issues
- Added safe import mechanism for `ZeroRedundancyOptimizer` that disables JIT compilation temporarily
- Fallback to graceful degradation when distributed optimizers are not available
- Modified `mmengine/runner/checkpoint.py` to handle PyTorch 2.6+ `weights_only` parameter changes
- Added `_safe_torch_load` function that provides backward compatibility for checkpoint loading
- Automatically handles numpy array serialization issues in PyTorch 2.6+

## Common Commands

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_runner/test_runner.py

# Run tests with coverage
python -m pytest tests/ --cov=mmengine --cov-report=html
```

### Linting and Code Quality
```bash
# Run pre-commit hooks (includes flake8, isort, yapf, mypy, etc.)
pre-commit run --all-files

# Run specific linting tools
flake8 mmengine/
isort mmengine/
yapf -r mmengine/
mypy mmengine/

# Check docstring coverage
interrogate -v --ignore-init-method --ignore-magic --ignore-module --ignore-nested-functions --ignore-regex "__repr__" --fail-under 80 mmengine
```

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e .[all]

# Install only test dependencies
pip install -e .[tests]

# Verify installation
python -c 'from mmengine.utils.dl_utils import collect_env;print(collect_env())'
```

## Code Architecture

### Core Components

1. **Runner System** (`mmengine/runner/`):
   - `Runner`: Main training orchestrator that manages the entire training process
   - `FlexibleRunner`: More flexible version for advanced use cases  
   - Training loops: `EpochBasedTrainLoop`, `IterBasedTrainLoop`
   - Evaluation loops: `ValLoop`, `TestLoop`

2. **Model System** (`mmengine/model/`):
   - `BaseModel`: Base class for all models with standardized forward interface
   - `BaseModule`: Enhanced nn.Module with initialization and checkpointing
   - Model wrappers: `MMDistributedDataParallel`, `MMFullyShardedDataParallel`
   - Weight initialization utilities

3. **Configuration System** (`mmengine/config/`):
   - `Config`: Powerful configuration management supporting Python, JSON, YAML
   - `ConfigDict`: Dictionary with attribute access
   - Supports inheritance, variable substitution, and lazy evaluation

4. **Registry System** (`mmengine/registry/`):
   - Central registry for components (models, optimizers, datasets, etc.)
   - `Registry`: Base registry class
   - Pre-defined registries: `MODELS`, `OPTIMIZERS`, `DATASETS`, etc.

5. **Hook System** (`mmengine/hooks/`):
   - Extensible hook mechanism for training customization
   - Built-in hooks: `CheckpointHook`, `LoggerHook`, `EMAHook`, etc.

6. **Dataset System** (`mmengine/dataset/`):
   - `BaseDataset`: Base class for datasets with lazy loading
   - Dataset wrappers: `ConcatDataset`, `RepeatDataset`, `ClassBalancedDataset`
   - Samplers: `DefaultSampler`, `InfiniteSampler`

7. **Evaluation System** (`mmengine/evaluator/`):
   - `Evaluator`: Manages multiple metrics evaluation
   - `BaseMetric`: Base class for custom metrics

8. **Optimization System** (`mmengine/optim/`):
   - `OptimWrapper`: Wrapper for optimizers with additional features
   - Parameter schedulers for learning rate, momentum, etc.
   - Support for gradient accumulation, mixed precision

### Key Design Patterns

- **Registry Pattern**: Components are registered and built dynamically from configs
- **Hook Pattern**: Training process is extensible through hooks
- **Strategy Pattern**: Different training strategies (single device, distributed, etc.)
- **Builder Pattern**: Complex objects built from configuration dictionaries

### File Structure Conventions

- `__init__.py`: Exports public API
- `base_*.py`: Base classes and abstract interfaces  
- `*_wrapper.py`: Wrapper classes for extending functionality
- `utils.py`: Utility functions specific to the module
- `registry_utils.py`: Registry-related utilities

### Configuration Structure

MMEngine uses a hierarchical configuration system:

```python
# Basic config structure
config = dict(
    model=dict(type='ModelName', param1=value1),
    optimizer=dict(type='SGD', lr=0.01),
    train_dataloader=dict(batch_size=32, dataset=dict(...)),
    train_cfg=dict(by_epoch=True, max_epochs=100),
    # ... other components
)
```

### Testing Patterns

- Tests are organized by module in `tests/test_*/`
- Use `pytest` for running tests
- Mock external dependencies when possible
- Test both success and failure cases
- Use parameterized tests for multiple scenarios

## Development Guidelines

### Code Style
- Follow PEP 8 with yapf formatting
- Use type hints where appropriate
- Line length limit: 79 characters
- Use pre-commit hooks to ensure code quality

### Documentation
- All public APIs must have docstrings
- Follow NumPy docstring format
- Include examples in docstrings when helpful
- Maintain minimum 80% docstring coverage

### Testing
- Write tests for new features and bug fixes
- Ensure tests pass locally before submitting
- Use appropriate test fixtures and utilities
- Test edge cases and error conditions

### Imports
- Use absolute imports for mmengine modules
- Sort imports using isort
- Avoid circular imports
- Import only what you need

## Large Model Training Support

MMEngine integrates with major distributed training frameworks:
- **ColossalAI**: For large-scale model training
- **DeepSpeed**: ZeRO optimization and mixed precision
- **FSDP**: PyTorch's Fully Sharded Data Parallel

## Common Patterns

### Creating a New Component
1. Define the component class inheriting from appropriate base class
2. Register it with the appropriate registry
3. Add it to the module's `__init__.py`
4. Write tests and documentation

### Extending Training Process
1. Create custom hooks for training modifications
2. Use the registry system for configuration-based instantiation
3. Leverage the Runner's flexible architecture

### Custom Metrics
1. Inherit from `BaseMetric`
2. Implement `process` and `compute_metrics` methods
3. Register with `METRICS` registry