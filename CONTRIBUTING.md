# Contributing to text-to-emoji

Thank you for your interest in contributing to text-to-emoji! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Getting Started

### Prerequisites

- Python 3.10+
- GPU with CUDA support (recommended) or MPS support (Apple Silicon)
- ~10GB disk space for dataset and checkpoints
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/text-to-emoji.git
cd text-to-emoji
```

2. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies:
```bash
uv sync
```

4. Install pre-commit hooks:
```bash
uv run pre-commit install
```

## Development Setup

### Project Structure

```
text-to-emoji/
├── src/
│   ├── config.py              # Configuration management
│   ├── data/                  # Dataset and data loading
│   ├── models/                # DiT and Diffusion models
│   ├── text_encoder/          # CLIP text encoder
│   ├── training/              # Training scripts and utilities
│   └── inference/             # Generation scripts
├── tests/                     # Unit tests
├── scripts/                   # Training and utility scripts
├── checkpoints/               # Model checkpoints (gitignored)
└── assets/                    # Generated samples
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest --cov=src tests/

# Run specific test file
uv run pytest tests/test_diffusion.py
```

### Code Formatting and Linting

We use `ruff` for linting and formatting:

```bash
# Check code style
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Format code
uv run ruff format src/
```

### Type Checking

We use `mypy` for static type checking:

```bash
uv run mypy src/
```

## Code Style

### General Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Keep functions focused and concise (< 50 lines when possible)
- Write self-documenting code with clear variable names
- Use docstrings for public APIs only

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Import Order

1. Standard library imports
2. Third-party imports
3. Local application imports

Example:
```python
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from src.config import ModelConfig
from src.models.dit import DiT
```

### Docstring Format

Use Google-style docstrings for public functions:

```python
def generate_image(
    prompt: str,
    model: DiT,
    num_steps: int = 50,
) -> torch.Tensor:
    """Generate an image from a text prompt.

    Args:
        prompt: Text description for image generation.
        model: DiT model for generation.
        num_steps: Number of diffusion sampling steps.

    Returns:
        Generated image tensor (C, H, W) in [0, 1] range.
    """
    ...
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names: `test_<functionality>_<scenario>`
- Use pytest fixtures for shared setup

Example:
```python
def test_diffusion_forward_process():
    """Test that forward diffusion adds noise correctly."""
    diffusion = Diffusion(num_timesteps=1000)
    x_0 = torch.randn(1, 3, 32, 32)
    t = torch.tensor([500])
    
    x_t, noise = diffusion.q_sample(x_0, t)
    
    assert x_t.shape == x_0.shape
    assert not torch.allclose(x_t, x_0)
```

### Test Coverage

- Aim for 80%+ test coverage
- Focus on critical paths and edge cases
- Test both success and failure scenarios

## Pull Request Process

### Before Submitting

1. Ensure all tests pass: `uv run pytest tests/`
2. Run linting: `uv run ruff check src/`
3. Run type checking: `uv run mypy src/`
4. Update documentation if needed
5. Add tests for new features

### PR Guidelines

1. **Title**: Use conventional commits format:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test additions/modifications
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements

2. **Description**: Include:
   - What changes were made and why
   - Link to related issues
   - Screenshots (for UI changes)
   - Breaking changes (if any)

3. **Commits**: 
   - Keep commits atomic and focused
   - Write clear commit messages
   - Squash fixup commits before merging

### Review Process

- At least one maintainer approval required
- All CI checks must pass
- Address review comments promptly
- Update PR description if scope changes

## Reporting Issues

### Bug Reports

Include:
- Python version and OS
- PyTorch and CUDA versions
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:
- Use case and motivation
- Proposed solution (if any)
- Alternative approaches considered
- Impact on existing functionality

### Good Examples

**Bug Report:**
```
**Bug**: Training crashes with CUDA OOM on batch_size=64

**Environment:**
- Python 3.10
- PyTorch 2.0.0
- CUDA 11.8
- GPU: RTX 3090 (24GB)

**Reproduction:**
```bash
uv run python src/training/train.py --batch-size 64 --model-size B
```

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB...
```

**Expected**: Training should work with 24GB GPU
**Actual**: OOM error
```

**Feature Request:**
```
**Feature**: Add support for LoRA fine-tuning

**Motivation**: Want to fine-tune on custom emoji styles without full model training

**Proposed Solution**: 
- Add LoRA adapters to DiT blocks
- Implement LoRA-specific training script
- Support merging LoRA weights into base model

**Alternatives**:
- Full fine-tuning (too expensive)
- DreamBooth (different use case)
```

## Development Workflow

### Branching Strategy

- `main` - stable, production-ready code
- `dev` - integration branch for features
- `feature/*` - new features
- `fix/*` - bug fixes
- `docs/*` - documentation updates

### Typical Workflow

1. Create feature branch from `main`:
```bash
git checkout -b feature/add-lora-support
```

2. Make changes and commit:
```bash
git add .
git commit -m "feat: add LoRA adapters to DiT blocks"
```

3. Push and create PR:
```bash
git push origin feature/add-lora-support
```

4. Address review feedback

5. Merge after approval

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## Questions?

- Open a [Discussion](https://github.com/YOUR_USERNAME/text-to-emoji/discussions)
- Join our [Discord](DISCORD_LINK) (if available)
- Check existing [Issues](https://github.com/YOUR_USERNAME/text-to-emoji/issues)

Thank you for contributing to text-to-emoji! 🎨
