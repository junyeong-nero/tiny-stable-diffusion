# AGENTS.md - Developer Guide for AI Coding Agents

This guide provides essential information for AI coding agents working on the text-to-emoji project.

## 🛠️ Build, Lint, and Test Commands

### Package Manager: uv
This project uses [uv](https://github.com/astral-sh/uv) - a fast Python package manager (10-100x faster than pip).

```bash
# Install dependencies
uv sync

# Run commands (uv automatically manages the virtual environment)
uv run <command>
```

### Testing

```bash
# Run all tests
uv run pytest tests/

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing tests/

# Run a single test file
uv run pytest tests/test_diffusion.py

# Run a single test function
uv run pytest tests/test_diffusion.py::TestDiffusion::test_initialization

# Run tests matching a pattern
uv run pytest -k "test_forward" tests/
```

### Linting and Formatting

```bash
# Check code style with ruff
uv run ruff check src/ tests/

# Format code with ruff
uv run ruff format src/ tests/

# Format code with black
uv run black src/ tests/

# Type checking
uv run mypy src/

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

### Training and Inference

```bash
# Train model with default emoji dataset
uv run python src/training/train.py --epochs 100 --batch-size 64

# Train with CIFAR-100 dataset (60,000 images)
uv run python src/training/train.py \
    --data-source cifar100 \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-4

# Train with CIFAR-100 using coarse labels (20 classes)
uv run python src/training/train.py \
    --data-source cifar100 \
    --use-coarse-labels \
    --epochs 200 \
    --batch-size 128

# Fine-tune on emoji dataset using pretrained checkpoint
uv run python src/training/train.py \
    --data-source huggingface \
    --dataset-name junyeong-nero/emoji-32 \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --pretrain-checkpoint checkpoints/pretrain_cifar100.pt

# Convert pretrained checkpoint for fine-tuning
python scripts/convert_pretrain_to_finetune.py \
    --input checkpoints/pretrain_cifar100.pt \
    --output checkpoints/finetune.pt \
    --reset-cross-attn

# Or use convenience script
./scripts/train.sh --epochs 100 --batch-size 64 --mixed-precision

# Generate images
uv run python src/inference/generate.py --prompt "rocket" --num-samples 4

# Interactive demo
uv run python demo.py --checkpoint checkpoints/model_best.pt
```

## 📝 Code Style Guidelines

### Import Order (isort)

```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import math
from pathlib import Path
from typing import Literal

# 3. Third-party packages
import torch
import torch.nn as nn
from PIL import Image

# 4. First-party (local) imports
from src.config import ModelConfig
from src.models.diffusion import Diffusion
```

### Formatting Rules (black)

- **Line length**: 100 characters max
- **Quote style**: Double quotes (`"`)
- **Indentation**: 4 spaces (no tabs)
- **Trailing commas**: Keep them for multi-line structures

Both `black` and `ruff format` produce consistent results. Use either:

### Type Hints (mypy)

**REQUIRED**: All function signatures must have type hints.

```python
# Good ✅
def generate_image(
    prompt: str,
    model: DiT,
    num_steps: int = 50,
) -> torch.Tensor:
    """Generate image from prompt."""
    ...

# Bad ❌
def generate_image(prompt, model, num_steps=50):
    ...
```

**Note**: Use `from __future__ import annotations` to enable PEP 585-style type hints (`list[str]` instead of `List[str]`).

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `DiT`, `Diffusion`, `EmojiDataset`)
- **Functions/methods**: `snake_case` (e.g., `generate_image`, `save_checkpoint`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`, `DEFAULT_LR`)
- **Private methods**: `_leading_underscore` (e.g., `_extract`, `_cosine_beta_schedule`)
- **Type variables**: `PascalCase` with `T` suffix (e.g., `ModelT`, `ConfigT`)

### Docstrings (Google Style)

Use Google-style docstrings for **public functions and classes only**.

```python
def sample(
    self,
    model: nn.Module,
    shape: tuple[int, ...],
    text_embeds: torch.Tensor,
    num_steps: int = 50,
) -> torch.Tensor:
    """Generate samples using DDIM sampling.

    Args:
        model: Denoising model (DiT).
        shape: Output tensor shape (B, C, H, W).
        text_embeds: Text conditioning embeddings.
        num_steps: Number of sampling steps.

    Returns:
        Generated images in [0, 1] range.
    """
    ...
```

**Do NOT add docstrings to**:
- Private methods (unless complex algorithms)
- Self-explanatory functions
- Test functions

### Error Handling

**NEVER suppress errors silently**:

```python
# Bad ❌
try:
    result = some_operation()
except Exception:
    pass

# Good ✅
try:
    result = some_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
```

**Do NOT use**:
- `as any` type casts
- `@ts-ignore` or `# type: ignore` (without specific error codes)
- `# noqa` (use specific error codes: `# noqa: E501`)

### Testing Conventions

```python
class TestDiffusion:
    """Test suite for Diffusion class."""

    @pytest.fixture
    def diffusion(self) -> Diffusion:
        """Create a Diffusion instance for testing."""
        return Diffusion(num_timesteps=1000, beta_schedule="cosine")

    def test_forward_process(self, diffusion: Diffusion) -> None:
        """Test forward diffusion adds noise correctly."""
        x_0 = torch.randn(1, 3, 32, 32)
        t = torch.tensor([500])
        
        x_t, noise = diffusion.q_sample(x_0, t)
        
        assert x_t.shape == x_0.shape
        assert not torch.allclose(x_t, x_0)
```

**Test naming**: `test_<functionality>_<scenario>`

## 🚫 Prohibited Patterns

1. **No commented-out code** - Delete or use git history
2. **No magic numbers** - Use named constants
3. **No mutable default arguments** - Use `None` and initialize inside function
4. **No bare `except:`** - Always specify exception types
5. **No `print()` in library code** - Use `logging` module

## 🔧 Pre-commit Hooks

Installed hooks (run automatically on `git commit`):
- `black` - Python code formatting
- `ruff` - Linting
- `ruff-format` - Code formatting- `mypy` - Type checking
 (excludes tests/)
- `trailing-whitespace` - Remove trailing spaces
- `end-of-file-fixer` - Ensure newline at EOF
- `check-yaml` - Validate YAML files
- `check-added-large-files` - Prevent large files (>1MB)
- `check-merge-conflict` - Detect merge conflicts
- `detect-private-key` - Prevent committing secrets

## 📁 Project Structure

```
src/
├── config.py           # Centralized configuration
├── data/               # Dataset loaders
├── models/             # DiT and Diffusion models
├── text_encoder/       # CLIP text encoder
├── training/           # Training loop and EMA
└── inference/          # Generation scripts

tests/                  # Unit tests (mirrors src/)
docs/                   # Documentation
scripts/                # Utility scripts
```

## 🎯 Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: Add mixed precision training support
fix: Correct diffusion posterior mean calculation
docs: Update training guide with new hyperparameters
test: Add tests for DDIM sampling
refactor: Simplify EMA implementation
perf: Optimize attention computation
chore: Update dependencies
```

## 📚 Key Dependencies

- **PyTorch** (≥2.0.0) - Deep learning framework
- **Transformers** (≥4.30.0) - CLIP text encoder
- **Datasets** (≥4.4.2) - Hugging Face dataset loader
- **Pillow** (≥10.0.0) - Image processing

## ⚡ Performance Tips

- Use `torch.no_grad()` for inference
- Enable mixed precision with `--mixed-precision` flag
- Use DDIM sampling (faster than DDPM: 50 steps vs 1000)
- Batch multiple prompts for efficiency

## 🔍 Debugging

```bash
# Verbose test output
uv run pytest -vv tests/

# Show print statements
uv run pytest -s tests/

# Drop into debugger on failure
uv run pytest --pdb tests/

# Run only failed tests from last run
uv run pytest --lf tests/
```

---

**Last Updated**: 2025-01-09
**Project**: text-to-emoji v0.1.0
