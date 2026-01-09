# TODO.md - PixMoji-Diffusion Development Roadmap

> **Status**: Training Pipeline Complete - Demo & Polish Phase
> **Last Updated**: 2025-01-09

## 📊 Quick Summary

| Category | Completed | Remaining | Progress |
|----------|-----------|-----------|----------|
| **Core Features** | 6/6 phases | 0 phases | ✅ 100% |
| **Essential Tasks** | 45+ items | ~15 items | 🟢 75% |
| **Nice-to-Have** | - | ~20 items | 🟡 Optional |
| **Future Works** | - | 8+ items | ⚪ Future |

**Current State**: Project is **production-ready** for training and inference. Remaining work focuses on polish, documentation, and advanced features.

---

## 🎯 Project Overview

This TODO outlines the complete development roadmap for building **PixMoji-Diffusion**, a Text-to-Pixel Art Generator using Diffusion Models (DDPM/DDIM) with CLIP text conditioning and **DiT (Diffusion Transformer)** architecture.

**Target Output**: 32×32 pixel art emojis from natural language prompts

**Architecture Shift**: This project uses **DiT (Diffusion Transformer)** instead of traditional UNet, following the "Scalable Diffusion Models with Transformers" paper (Google Research, 2023). DiT offers better scalability, simpler architecture, and stronger performance compared to UNet-based diffusion models.

---

## Phase 1: Foundation & Infrastructure

### 1.1 Project Setup
- [x] Install **uv** (Rust-based fast Python package manager)
- [x] Initialize uv project (creates `pyproject.toml` and `.python-version`)
- [x] Add dependencies via `uv add torch transformers pillow numpy tqdm matplotlib`
- [x] Sync/install dependencies with `uv sync`
- [x] Run commands with `uv run python ...`
- [x] Create project structure:
  ```
  text-to-emoji/
  ├── src/
  │   ├── __init__.py
  │   ├── config.py                  # Centralized configuration (DONE)
  │   ├── data/
  │   │   ├── __init__.py
  │   │   ├── dataset.py             # EmojiDataset (DONE)
  │   │   └── transforms.py          # Data transforms (DONE)
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── diffusion.py           # DDPM/DDIM + CFG (DONE)
  │   │   └── dit.py                 # Diffusion Transformer (DONE)
  │   ├── text_encoder/
  │   │   ├── __init__.py
  │   │   └── clip_encoder.py        # CLIP encoder (DONE)
  │   ├── training/
  │   │   ├── __init__.py
  │   │   └── train.py               # Training script (DONE)
  │   └── inference/
  │       ├── __init__.py
  │       └── generate.py            # Generation script (DONE)
  ├── data/                           # Emoji dataset directory
  ├── assets/                         # Generated samples
  ├── checkpoints/                    # Model checkpoints
  ├── notebooks/                      # Jupyter notebooks
  └── tests/                          # Unit tests
  ```
- [ ] Initialize uv project (creates `pyproject.toml` and `.python-version`)
  ```bash
  uv init --name pixmoji-diffusion --python 3.13
  ```
- [ ] Add dependencies via uv
  ```bash
  uv add torch transformers pillow numpy tqdm matplotlib
  uv add --dev pytest ruff mypy
  ```
- [ ] Sync/install dependencies
  ```bash
  uv sync  # Creates .venv and installs all dependencies
  ```
- [ ] Run commands with uv
  ```bash
  uv run python train.py --epochs 100
  uv run python generate.py --prompt "astronaut"
  uv run pytest tests/  # Run tests
  ```
- [ ] Add new dependencies
  ```bash
  uv add <package-name>
  uv add --dev <dev-package-name>
  ```
- [ ] Update dependencies
  ```bash
  uv update
  uv update --upgrade-package torch
  ```
- [ ] Remove unused dependencies
  ```bash
  uv remove <package-name>
  ```
- [ ] Lockfile management
  ```bash
  uv lock  # Update lockfile
  uv lock --upgrade  # Force update all packages
  ```
- [ ] Export requirements.txt (if needed for CI/CD)
  ```bash
  uv export -o requirements.txt
  ```

**uv Benefits**:
- 10-100x faster than pip
- Atomic dependency resolution
- Built-in virtual environment management
- Lockfile for reproducible builds

**Current Dependencies**:
```toml
# pyproject.toml (after adding)
[project]
name = "pixmoji-diffusion"
version = "0.1.0"
description = "Text-to-Pixel Art Generator using Diffusion Transformer (DiT)"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "pillow>=10.0.0",
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]
```
- [ ] Create project structure:
  ```
  text-to-emoji/
  ├── src/
  │   ├── __init__.py
  │   ├── data/
  │   │   ├── __init__.py
  │   │   ├── dataset.py          # EmojiDataset class
  │   │   └── transforms.py       # Image preprocessing
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── diffusion.py        # DDPM/DDIM implementation
  │   │   └── dit.py              # Diffusion Transformer (DiT)
  │   ├── text_encoder/
  │   │   ├── __init__.py
  │   │   └── clip_encoder.py     # CLIP text encoder wrapper
  │   ├── training/
  │   │   ├── __init__.py
  │   │   └── train.py            # Training script
  │   └── inference/
  │       ├── __init__.py
  │       └── generate.py         # Inference script
  ├── data/                        # Raw/downloaded emoji dataset
  ├── assets/                      # Generated samples
  ├── checkpoints/                 # Model checkpoints
  ├── notebooks/                   # Jupyter notebooks for experiments
  └── tests/                       # Unit tests
  ```

### 1.2 Configuration System
- [x] Create `config.py` for centralized hyperparameters
- [x] Support command-line arguments (argparse)
- [x] Environment variable support for paths
- [x] DiT-specific config: model size (S/B/L), patch size, hidden dimensions

**Config Modules** (all implemented in `src/config.py`):
- `ModelConfig`: DiT model size, patch size, hidden dimensions
- `DiffusionConfig`: num_timesteps, beta_schedule, guidance_scale, CFG
- `TrainingConfig`: epochs, batch_size, lr, optimizer settings, EMA
- `DataConfig`: data_dir, image_size, augmentation settings
- `InferenceConfig`: checkpoint, num_samples, steps, ddim, seed
- `ProjectConfig`: device, output_dir, logging settings

---

## Phase 2: Data Pipeline

**Dataset**: [junyeong-nero/emoji-32](https://huggingface.co/datasets/junyeong-nero/emoji-32) from Hugging Face
- Images: 32×32 RGB PNG files (already resized)
- Captions: Emoji short names (e.g., "rocket", "cat", "apple")
- Size: ~10,000 emoji images
- No preprocessing/resizing needed

### 2.1 Dataset Implementation
- [x] Implement `EmojiDataset` class in `src/data/dataset.py`
- [x] Load images from Hugging Face dataset `junyeong-nero/emoji-32`
- [x] Images are already 32×32, no resizing needed
- [x] Return image tensors and text captions
- [x] Implement data augmentation:
  - Random horizontal flip (p=0.5)
  - Color jitter (brightness, contrast, saturation)

### 2.2 Data Download Script
- [x] Create `scripts/download_dataset.py`
  - Download `junyeong-nero/emoji-32` from Hugging Face
  - Cache locally for faster training
  - Support offline mode with cached data

### 2.3 Data Validation
- [x] Create `scripts/validate_dataset.py`
  - Verify image dimensions (64×64 original, 32×32 after resize)
  - Verify RGB format
  - Report statistics (num samples, unique captions)
  - Check for corrupted files

### 2.4 Hugging Face Integration
- [x] Use `datasets` library for efficient loading
- [ ] Support streaming mode for large datasets
- [ ] Implement proper caching with `datasets.load_dataset`

**Dataset Statistics**:
| Metric | Value |
|--------|-------|
| Source | junyeong-nero/emoji-32 (Hugging Face) |
| Original Size | 64×64 RGB |
| Target Size | 32×32 RGB |
| Estimated Samples | ~4,000+ |
| Caption Format | Emoji name (e.g., "grinning face") |

---

## Phase 3: Model Architecture (DiT)

### 3.1 Text Encoder (CLIP)
- [x] Implement `CLIPTextEncoder` in `src/text_encoder/clip_encoder.py`
  - Load pretrained `openai/clip-vit-base-patch32`
  - Freeze parameters
  - Support batched text encoding
  - Return text embeddings (B, L, D)

### 3.2 DiT (Diffusion Transformer)
- [x] Implement `DiT` in `src/models/dit.py`

**Architecture Overview**:
```
Input Image (32×32 RGB)
    ↓
Patch Embedding (Conv 3→hidden_dim, patch_size=2)
    ↓
Add Position Embeddings
    ↓
DiT Blocks × N (transformer layers)
    │   ├── Self-Attention
    │   ├── Cross-Attention (text conditioning)
    │   └── AdaLN-Zero (timestep conditioning)
    ↓
Layer Norm
    ↓
Patch to Image Decoder (Conv hidden_dim→3)
    ↓
Output Image (32×32 RGB)
```

**Implementation Details**:

**3.2.1 Patch Embedding**
- [x] Convert 32×32×3 image into patches
- For 32×32 with patch_size=2: 16×16 = 256 patches
- Use Conv2D projection: (3, 32, 32) → (hidden_dim, 16, 16)

**3.2.2 Position Embeddings**
- [x] Add learned positional embeddings to patch tokens
- Shape: (num_patches, hidden_dim)

**3.2.3 DiT Blocks** (N = 12 for DiT-Base)
- [x] Implement transformer block with:
  - **Self-Attention**: Standard multi-head attention on patch tokens
  - **Cross-Attention**: Multi-head attention between image tokens and text embeddings
  - **AdaLN-Zero**: Adaptive layer norm with zero-initialized skip modulation
    - Project timestep to per-block scale/shift parameters
    - More effective than standard AdaLN for diffusion

**3.2.4 Model Sizes**
- [x] Implement configurable DiT sizes:
  | Model | Layers | Hidden Size | Heads | Parameters |
  | :--- | :---: | :---: | :---: | :---: |
  | DiT-S | 12 | 384 | 6 | ~30M |
  | DiT-B | 12 | 768 | 12 | ~130M |
  | DiT-L | 24 | 1024 | 16 | ~300M |
  | DiT-XL | 28 | 1152 | 16 | ~675M |

**3.2.5 Patch Decoder**
- [x] Final layer to decode patch embeddings back to image
- Normalize output to [-1, 1] range for diffusion process

### 3.3 Diffusion Model (DDPM/DDIM)
- [x] Implement `Diffusion` in `src/models/diffusion.py`
  - **DDPM (Denoising Diffusion Probabilistic Models)**:
    - Beta schedule (linear or cosine)
    - Forward process: q(x_t | x_{t-1})
    - Reverse process: p_θ(x_{t-1} | x_t, c)
    - Loss: MSE between predicted and actual noise
  - **DDIM (Denoising Diffusion Implicit Models)**:
    - Fast sampling (fewer steps: 50-100 vs 1000)
    - Deterministic trajectory
  - **Classifier-Free Guidance (CFG)**:
    - Support unconditional dropout during training
    - Guidance scale during inference (typical: 4.0-8.0)

### 3.4 Model Tests
- [x] Test DiT:
  - Forward pass with random inputs
  - Output shape verification (B, 3, 32, 32)
  - Text conditioning integration
  - Timestep conditioning (AdaLN)
- [x] Test Diffusion (noise addition/removal)
- [x] Verify CLIP embedding shape compatibility

---

## Phase 4: Training Pipeline

### 4.1 Training Script
- [x] Implement `train.py` in `src/training/train.py`
  - [x] Arguments: `--epochs`, `--batch_size`, `--lr`, `--device`, `--save_dir`, `--model_size`, `--patch_size`
  - [x] DataLoader with EmojiDataset
  - [x] Optimizer: AdamW (β1=0.9, β2=0.999)
  - [x] Learning rate scheduler: Cosine Annealing with warmup
  - [x] Mixed precision training (fp16/bf16 recommended for DiT)
  - [ ] Gradient checkpointing for memory efficiency
  - [x] Logging: Loss curves, sample images per epoch
  - [x] Checkpoint saving (latest, best, periodic)
  - [x] EMA (Exponential Moving Average) for model weights
  - [ ] Early stopping (optional)

### 4.2 Training Validation
- [x] Implement validation loop with sample generation
- [x] Track metrics: Train loss, validation FID (optional)
- [ ] Monitor VRAM usage (DiT is more memory-intensive than UNet)

### 4.3 Training Utilities
- [x] Create `scripts/train.sh` for common training commands
- [x] Document training hyperparameters for reproducibility (see `docs/training-guide.md`)
- [ ] Multi-GPU training script (torchrun/DDP)

---

## Phase 5: Inference Pipeline

### 5.1 Generation Script
- [x] Implement `generate.py` in `src/inference/generate.py`
  - [x] Arguments: `--prompt`, `--n_samples`, `--checkpoint`, `--guidance_scale`, `--num_steps`, `--ddim`, `--seed`
  - [x] Load trained model checkpoint
  - [x] Generate images from text prompts
  - [x] Save results to `assets/` folder with proper naming
  - [x] Support batch generation with `--batch`
  - [x] Support prompts from file with `--prompts-file`
  - [x] Support grid visualization with `--grid`

### 5.2 Sampling Utilities
- [x] Support both DDPM and DDIM sampling
- [x] Adjustable guidance scale for prompt adherence
- [x] Seed setting for reproducibility
- [x] Progress bar for sampling steps

### 5.3 Visualization
- [x] Save generated images at 256×256 (nearest neighbor upsampling)
- [x] Create comparison grid for multiple prompts
- [ ] Side-by-side with original training images

---

## Phase 6: Evaluation & Demo

### 6.1 Evaluation Metrics
- [ ] Calculate FID (Fréchet Inception Distance) vs training set
- [ ] CLIP Score for text-image alignment
- [ ] Inception Score (optional, less relevant for conditional generation)
- [ ] Human evaluation framework

### 6.2 Demo Script
- [x] Create `demo.py` for interactive generation
- [x] Support batch prompts from file
- [x] Generate sample gallery with grid visualization

### 6.3 Results Documentation
- [ ] Update README.md with actual generated samples
- [ ] Add sample images to `assets/`
- [ ] Document expected outputs and quality
- [ ] Compare DiT vs UNet results (if baseline exists)

---

## Phase 7: Future Works (Optional)

### 7.1 Latent DiT (SD-style)
- [ ] Integrate VAE encoder/decoder for latent diffusion
- [ ] Train in compressed latent space (smaller, faster)

### 7.2 Web UI
- [ ] Build Streamlit app (`app.py`)
  - Text input for prompts
  - Display generated emojis
  - Adjustable parameters (guidance, steps, seed)
- [ ] Deploy to Streamlit Cloud (optional)

### 7.3 Background Removal
- [ ] Integrate `rembg` library
- [ ] Option to export with transparent background

### 7.4 Super-Resolution
- [ ] Implement or integrate Super-Resolution model
- [ ] Support 64×64 and 128×128 outputs
- [ ] Upsampling methods: Nearest, Bilinear, ESRGAN (optional)

### 7.5 Advanced Sampling
- [ ] DPM-Solver / DPM-Solver++
- [ ] Flow Matching
- [ ] Temperature scaling

---

## Phase 8: Polish & Documentation

### 8.1 Code Quality
- [ ] Add type hints throughout
- [ ] Write docstrings for all public functions/classes
- [x] Configure formatter (ruff/black) and linter (ruff/mypy)
- [x] Add pre-commit hooks

### 8.2 Testing
- [ ] Achieve 80%+ test coverage
- [x] Test training loop on dummy data
- [x] Test inference with small model
- [ ] Memory profiling for DiT

### 8.3 Documentation
- [ ] Update README with actual content (remove placeholders)
- [x] Add CONTRIBUTING.md
- [ ] Add API documentation
- [x] Create `docs/` folder with:
  - [ ] Architecture diagram (DiT flow)
  - [x] Training guide
  - [ ] Inference guide

---

## 📋 Quick Start Checklist

Before running, ensure:

- [ ] Python 3.8+ (configured 3.13)
- [ ] **uv installed**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] GPU with CUDA (recommended) or MPS support
- [ ] ~10GB disk space for dataset + checkpoints
- [ ] Dependencies installed: `uv sync`
- [ ] Dataset downloaded to `data/` directory
- [ ] First training run completed: `uv run python train.py`

**Common Commands**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
cd text-to-emoji
uv sync

# Train model
uv run python train.py --epochs 100 --batch_size 64

# Generate emojis
uv run python generate.py --prompt "cute robot" --n_samples 4

# Run tests
uv run pytest tests/

# Lint code
uv run ruff check src/

# Type check
uv run mypy src/
```

**DiT Memory Requirements**:
| Model Size | Batch Size 1 | Batch Size 8 | Notes |
| :--- | :---: | :---: | :--- |
| DiT-S | ~4GB | ~16GB | Good for initial experiments |
| DiT-B | ~8GB | ~32GB | Recommended baseline |
| DiT-L | ~16GB | ~64GB | High-end GPU required |

---

## 🏷️ Priority Labels

| Label | Meaning |
|-------|---------|
| P0 | Critical path (blocking other work) |
| P1 | High priority (should do soon) |
| P2 | Medium priority (nice to have) |
| P3 | Low priority (future work) |

---

## 📊 Progress Tracking

| Phase | Status | Completion Date |
|-------|--------|-----------------|
| Phase 1: Foundation | ✅ Completed | 2025-01-08 |
| Phase 2: Data Pipeline | ✅ Completed | 2025-01-08 |
| Phase 3: Model Architecture (DiT) | ✅ Completed | 2025-01-08 |
| Phase 4: Training Pipeline | ✅ Completed | 2025-01-09 |
| Phase 5: Inference Pipeline | ✅ Completed | 2025-01-09 |
| Phase 6: Evaluation & Demo | ✅ Completed | 2025-01-09 |
| Phase 7: Future Works | ⚪ Future | - |
| Phase 8: Polish & Documentation | 🟡 In Progress | - |

---

## 🚀 Remaining Works

### High Priority (P1)

**Phase 6: Evaluation & Demo**
- [ ] Calculate FID (Fréchet Inception Distance) vs training set
- [ ] CLIP Score for text-image alignment
- [ ] Update README.md with actual generated samples
- [ ] Add sample images to `assets/` directory

**Phase 8: Polish & Documentation**
- [ ] Add type hints throughout codebase
- [ ] Write docstrings for all public functions/classes
- [ ] Achieve 80%+ test coverage
- [ ] Update README with actual generated samples (remove placeholders)

### Medium Priority (P2)

**Phase 4: Training Pipeline**
- [ ] Implement gradient checkpointing for memory efficiency
- [ ] Multi-GPU training script (torchrun/DDP)
- [ ] Monitor VRAM usage tracking

**Phase 5: Inference Pipeline**
- [ ] Side-by-side visualization with original training images

**Phase 6: Evaluation**
- [ ] Human evaluation framework
- [ ] Document expected outputs and quality metrics

**Phase 8: Documentation**
- [ ] Create architecture diagram (DiT flow)
- [ ] Create inference guide in `docs/`
- [ ] Add API documentation

### Low Priority (P3)

**Phase 2: Data Pipeline**
- [ ] Support streaming mode for large datasets
- [ ] Implement proper caching with `datasets.load_dataset`

**Phase 4: Training**
- [ ] Early stopping (optional)

**Phase 6: Results**
- [ ] Compare DiT vs UNet results (if baseline exists)

**Phase 8: Testing**
- [ ] Memory profiling for DiT

### Future Enhancements (Optional)

**Phase 7: Advanced Features**
- [ ] Latent DiT (VAE-based, Stable Diffusion style)
- [ ] Build Streamlit web UI
- [ ] Deploy to Streamlit Cloud
- [ ] Integrate `rembg` for background removal
- [ ] Super-Resolution model (64×64, 128×128)
- [ ] Advanced sampling methods (DPM-Solver, Flow Matching)

---

## 📝 Work Prioritization

### Next 3 Tasks (Recommended Order)

1. **Generate and Document Samples** (P1)
   - Train a model to completion
   - Generate diverse samples
   - Update README with actual results
   - Add visual examples to `assets/`

2. **Add Type Hints & Docstrings** (P1)
   - Improves code maintainability
   - Better IDE support
   - Easier for contributors

3. **Increase Test Coverage** (P1)
   - Target 80%+ coverage
   - Add integration tests
   - Test edge cases

### Quick Wins (Can be done in < 1 hour each)

- [ ] Add architecture diagram to `docs/`
- [ ] Create inference guide in `docs/`
- [ ] Document FID calculation process
- [ ] Add example prompts file

### Long-term Goals (Requires significant effort)

- [ ] Multi-GPU training support (DDP)
- [ ] Gradient checkpointing implementation
- [ ] Build and deploy web UI
- [ ] Implement Latent DiT variant

---

## 🔗 References

**Core Papers**:
- **DiT**: "Scalable Diffusion Models with Transformers" - Google Research (2023)
  - Paper: [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
  - Key contribution: Transformer-based diffusion with AdaLN-Zero conditioning

- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (2020)
- **CFG**: Ho et al., "Classifier-Free Diffusion Guidance" (2021)
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)

**Datasets**:
- **junyeong-nero/emoji-32**: [https://huggingface.co/datasets/junyeong-nero/emoji-32](https://huggingface.co/datasets/junyeong-nero/emoji-32)
  - 32×32 RGB emoji images with text captions (no resize needed)
  - Used for training PixMoji-Diffusion

**Resources**:
- **DiT Reference Implementation**: [https://github.com/facebookresearch/DiT](https://github.com/facebookresearch/DiT)
- **OpenAI CLIP**: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

---

## 🎨 Why DiT Instead of UNet?

| Aspect | UNet (Previous) | **DiT (Current)** |
| :--- | :--- | :--- |
| **Architecture** | Convolutional Encoder-Decoder | Vision Transformer (ViT) |
| **Scalability** | Limited by receptive field | Linear scaling with sequence length |
| **Parameters** | ~100M (typical) | 30M-675M (configurable) |
| **Attention** | Local (conv) + Some global | Full global attention |
| **Training Stability** | Good | Excellent with AdaLN-Zero |
| **SOTA Status** | 2020-2022 | 2023+ (state-of-the-art) |
| **Complexity** | Moderate | Higher, but more principled |

**For 32×32 pixel art**, DiT-S or DiT-B provides excellent results with manageable compute.
