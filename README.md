# 🎨 PixMoji-Diffusion: Text-to-Pixel Art Generator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

> **"Transform your imagination into retro 32x32 pixel art emojis."**
> A diffusion-based model that generates unique, custom emoji-style pixel art from text prompts.

## 📖 Introduction

PixMoji-Diffusion is a multi-modal generative AI model that transforms natural language descriptions into 32×32 pixel art emojis. Built with state-of-the-art diffusion transformer technology, it produces high-quality, text-conditioned imagery suitable for Discord/Slack emojis, game sprites, or any retro-styled creative project.

### ✨ Key Features

- **Text-to-Image Generation**: Create images from prompts like "astronaut", "cute robot", or "grinning cat"
- **Pixel Art Style**: Outputs optimized for 32×32 resolution with authentic retro/dot aesthetic
- **Diffusion Transformer (DiT)**: Modern transformer-based architecture with AdaLN-Zero conditioning
- **Fast Sampling**: DDIM (Denoising Diffusion Implicit Models) for rapid generation
- **Classifier-Free Guidance (CFG)**: Enhanced prompt adherence for more accurate results
- **Two-Stage Training**: Pretrain on CIFAR-100, fine-tune on emoji data

## 🧠 Model Architecture

PixMoji-Diffusion uses **DiT (Diffusion Transformer)**, a modern architecture from "Scalable Diffusion Models with Transformers" (Google Research, 2023).

```
Input Image (32×32 RGB)
    ↓
Patch Embedding (Conv 3→384, patch_size=2)
    ↓
Add Position Embeddings
    ↓
DiT Blocks × 12 (transformer layers)
    ├── Self-Attention (global attention on patches)
    ├── Cross-Attention (text conditioning)
    └── AdaLN-Zero (timestep conditioning)
    ↓
Patch Decoder (Conv 384→3)
    ↓
Output Image (32×32 RGB)
```

### 1. Text Encoder (CLIP)

Uses OpenAI's **CLIP** (`openai/clip-vit-base-patch32`) Text Encoder in frozen mode:
- Converts text prompts to 77×512 embeddings
- Provides text conditioning for the diffusion model
- Enables semantic understanding of natural language

### 2. Diffusion Process

- **Forward Process**: Gradually adds Gaussian noise over 1000 timesteps
- **Reverse Process**: DiT predicts and removes noise to reconstruct images
- **DDIM Sampling**: Fast deterministic sampling in 50-100 steps
- **Classifier-Free Guidance**: Enhanced prompt adherence (scale=7.5)

### 3. Model Configurations

| Model | Layers | Hidden Size | Heads | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| DiT-S | 12 | 384 | 6 | ~30M |
| DiT-B | 12 | 768 | 12 | ~130M |
| DiT-L | 24 | 1024 | 16 | ~300M |

Default: **DiT-S** (~30M parameters) for efficient training and inference.

## 📂 Dataset

Supports multiple datasets for flexible training:

### 1. Emoji Dataset (Fine-tuning)
- **Source**: [junyeong-nero/emoji-32](https://huggingface.co/datasets/junyeong-nero/emoji-32)
- **Size**: ~1,900 emoji images
- **Resolution**: 32×32 RGB

### 2. CIFAR-100 (Pretraining)
- **Source**: torchvision.datasets.CIFAR100
- **Size**: 60,000 images
- **Classes**: 100 (or 20 coarse categories)
- **Use Case**: Quick pretraining, general visual concepts

## 🚀 Quick Start

### Installation

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Two-Stage Training (Recommended)

**Stage 1: Pretrain** (CIFAR-100)
```bash
# CIFAR-100 (60K images)
./pretraining.sh
```

**Stage 2: Fine-tune on Emoji**
```bash
# First, edit main.py: TRAINING_STAGE = "finetune"
./finetuning.sh
```

### Direct Training

```bash
# Edit main.py to configure your settings
python main.py --train       # Training
python main.py --generate    # Generation
python main.py --demo        # Interactive demo
```

## 📖 Training Guide

### Two-Stage Training Pipeline

For best results with limited emoji data, use the two-stage approach:

**Stage 1: Pretraining** (CIFAR-100)

```python
# In main.py
TRAINING_STAGE = "pretrain"

PRETRAIN_CONFIG = {
    "data_source": "cifar100",
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "initial_cfg_prob": 0.0,     # Start unconditional
    "final_cfg_prob": 0.1,       # Gradually add conditioning
    "cfg_warmup_epochs": 10,
}
```

| Dataset | Images | Best For |
| :--- | :---: | :--- |
| **CIFAR-100** ⭐ | 60,000 | Quick experiments, limited storage |

**Stage 2: Fine-tuning**
```python
# In main.py
TRAINING_STAGE = "finetune"

FINETUNE_CONFIG = {
    "data_source": "huggingface",
    "dataset_name": "junyeong-nero/emoji-32",
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "cfg_prob": 0.1,
    "pretrain_checkpoint": "checkpoints/pretrain_cifar100.pt",
}
```

### Training Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--epochs` | Number of training epochs | 100 |
| `--batch-size` | Batch size | 64 |
| `--learning-rate` | Learning rate | 1e-4 |
| `--model-size` | DiT model size: XS, S, B, L, XL | S |
| `--data-source` | Dataset: huggingface, cifar100 | huggingface |

### Generating Images

```bash
# Single prompt
python main.py --generate --prompt "a cute robot"

# Multiple prompts (comma-separated)
python main.py --generate --prompt "rocket,cat,ghost"

# With custom checkpoint
python main.py --generate --prompt "star" --checkpoint checkpoints/model_best.pt

# Multiple samples
python main.py --generate --prompt "heart" --num-samples 4

# Custom sampling steps
python main.py --generate --prompt "fire" --steps 100
```

**Generation Arguments**:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--prompt` | Text description | required |
| `--checkpoint` | Model checkpoint path | checkpoints/model_best.pt |
| `--num-samples` | Number of images to generate | 1 |
| `--steps` | Diffusion sampling steps | 50 |
| `--guidance` | CFG scale | 7.5 |
| `--seed` | Random seed | None |

## 🛠️ Configuration

All configuration is done in `main.py`:

```python
# Training stage: "pretrain" or "finetune"
TRAINING_STAGE = "pretrain"

# Pretraining on CIFAR-100
PRETRAIN_CONFIG = {
    "data_source": "cifar100",
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "initial_cfg_prob": 0.0,
    "final_cfg_prob": 0.1,
    "cfg_warmup_epochs": 20,
}

# Fine-tuning on Emoji
FINETUNE_CONFIG = {
    "data_source": "huggingface",
    "dataset_name": "junyeong-nero/emoji-32",
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "cfg_prob": 0.1,
    "pretrain_checkpoint": "checkpoints/pretrain_cifar100.pt",
}

# Common settings
COMMON_CONFIG = {
    "model_size": "S",
    "patch_size": 2,
    "num_timesteps": 1000,
    "beta_schedule": "cosine",
    "guidance_scale": 7.5,
    "use_ema": True,
    "ema_decay": 0.9999,
}
```

## 📁 Project Structure

```
PixMoji-Diffusion/
├── main.py                 # Main entry point (train/generate/demo)
├── src/
│   ├── config.py          # Configuration classes
│   ├── data/
│   │   └── dataset.py     # Dataset loaders (emoji, CIFAR-100)
│   ├── models/
│   │   ├── dit.py         # DiT model implementation
│   │   └── diffusion.py   # Diffusion process
│   ├── text_encoder/
│   │   └── clip_encoder.py # CLIP text encoder
│   ├── training/
│   │   ├── train.py       # Training script
│   │   └── ema.py         # Exponential Moving Average
│   └── inference/
│       └── generate.py    # Generation script
├── scripts/
│   └── convert_pretrain_to_finetune.py  # Checkpoint conversion
├── checkpoints/           # Saved model checkpoints
├── samples/               # Generated samples
└── AGENTS.md             # Developer guide
```

## 🖼️ Results

Generated images are native 32×32 pixels. For display, they're upscaled using nearest-neighbor interpolation.

| Prompt | Result |
| :--- | :---: |
| **"rocket"** | <img src="assets/sample_rocket.png" width="100"> |
| **"cat"** | <img src="assets/sample_cat.png" width="100"> |
| **"robot"** | <img src="assets/sample_robot.png" width="100"> |

*(Add your generated images to the `assets/` folder)*

## 🔧 Development

### Code Quality

```bash
# Check code style
uv run ruff check src/

# Format code
uv run black src/

# Type checking
uv run mypy src/
```

### Running Tests

```bash
uv run pytest tests/
```

## 📚 References

- **DiT**: "Scalable Diffusion Models with Transformers" - Google Research (2023)
  - Paper: [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (2020)
- **CFG**: Ho et al., "Classifier-Free Diffusion Guidance" (2021)
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)

## 📄 License

MIT License - See LICENSE file for details.

---

**Note**: Generated images are native 32×32 pixels. For display, they're upscaled to larger sizes using nearest-neighbor interpolation to preserve the pixel art aesthetic.
