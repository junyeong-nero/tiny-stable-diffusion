# 🎨 text-to-emoji: Text-to-Pixel Art Generator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

> **"Transform your imagination into retro 32x32 pixel art emojis."**
> A diffusion-based model that generates unique, custom emoji-style pixel art from text prompts.

## 📖 Introduction

text-to-emoji is a multi-modal generative AI model that transforms natural language descriptions into 32×32 pixel art emojis. Built with state-of-the-art diffusion transformer technology, it produces high-quality, text-conditioned imagery suitable for Discord/Slack emojis, game sprites, or any retro-styled creative project.

### ✨ Key Features

- **Text-to-Image Generation**: Create images from prompts like "astronaut", "cute robot", or "grinning cat"
- **Pixel Art Style**: Outputs optimized for 32×32 resolution with authentic retro/dot aesthetic
- **Diffusion Transformer (DiT)**: Modern transformer-based architecture with AdaLN-Zero conditioning
- **Fast Sampling**: DDIM (Denoising Diffusion Implicit Models) for rapid generation
- **Classifier-Free Guidance (CFG)**: Enhanced prompt adherence for more accurate results
- **Two-Stage Training**: Pretrain on image-caption datasets (Flickr8k/CC3M), fine-tune on emoji data

## 🧠 Model Architecture

text-to-emoji supports two transformer architectures:

1. **DiT (Diffusion Transformer)** - From "Scalable Diffusion Models with Transformers" (Google Research, 2023)
2. **MMDiT (Multi-Modal DiT)** - From Stable Diffusion 3 (Esser et al., 2024)

```
Input Image (32×32 RGB)
    ↓
Patch Embedding (Conv 3→384, patch_size=2)
    ↓
Add Position Embeddings
    ↓
DiT Blocks × 12 (transformer layers)
    ├── Self-Attention (global attention on patches)
    ├── Cross-Attention OR Joint Attention (text conditioning)
    └── AdaLN-Zero (timestep conditioning)
    ↓
Patch Decoder (Conv 384→3)
    ↓
Output Image (32×32 RGB)
```

### Architecture Comparison

| Feature | DiT (Standard) | MMDiT (SD3) |
|---------|---------------|-------------|
| **Text Conditioning** | Cross-Attention | Joint Attention |
| **Parameters (DiT-S)** | ~30M | ~87M |
| **Attention** | Separate image/text attention | Unified text-image attention |
| **Library** | Custom implementation | [lucidrains/mmdit](https://github.com/lucidrains/mmdit) |

### 1. Text Encoder (CLIP)

Uses OpenAI's **CLIP** Text Encoder in frozen mode:
- Converts text prompts to pooled embeddings (B, D)
- Provides text conditioning for the diffusion model
- Enables semantic understanding of natural language

### 2. Diffusion Process

- **Forward Process**: Gradually adds Gaussian noise over 1000 timesteps
- **Reverse Process**: DiT predicts and removes noise to reconstruct images
- **DDIM Sampling**: Fast deterministic sampling in 50-100 steps
- **Classifier-Free Guidance**: Enhanced prompt adherence (scale=7.5)

### 3. Model Configurations

#### DiT Models (Standard)

| Model | Layers | Hidden Size | Heads | Parameters |
|-------|--------|-------------|-------|------------|
| DiT-S | 12 | 384 | 6 | ~30M |
| DiT-B | 12 | 768 | 12 | ~130M |
| DiT-L | 24 | 1024 | 16 | ~300M |
| DiT-XL | 28 | 1152 | 16 | ~675M |

#### MMDiT Models (Stable Diffusion 3)

| Model | Layers | Hidden Size | Heads | Parameters |
|-------|--------|-------------|-------|------------|
| MMDiT-S | 12 | 384 | 6 | ~87M |
| MMDiT-B | 12 | 768 | 12 | ~300M |
| MMDiT-L | 24 | 1024 | 16 | ~675M |
| MMDiT-XL | 28 | 1152 | 16 | ~1.6B |

**Default**: **DiT-S** (~30M parameters for DiT, ~87M for MMDiT) for efficient training and inference.

Configure model size in `config.yaml`:
```yaml
model_type: mmdit  # or "dit"
model_size: S      # Options: S, B, L, XL
```

### 4. MMDiT-Specific Features

- **qk_rmsnorm**: Use RMSNorm for QK attention (recommended: true)
- **register_tokens**: Register tokens from "Vision Transformers Need Registers" (0 or 4)

## 📂 Dataset

Supports multiple datasets for flexible training:

### 1. Emoji Dataset (Fine-tuning)
- **Source**: [junyeong-nero/emoji-32](https://huggingface.co/datasets/junyeong-nero/emoji-32)
- **Size**: ~1,900 emoji images
- **Resolution**: 32×32 RGB

### 2. Pretrain Datasets (Text-to-Image)

**Recommended Datasets** (all have image-caption pairs):

| Dataset | Images | Captions | Best For | HuggingFace |
|---------|--------|----------|----------|-------------|
| **Flickr8k** ⭐ | 8K | Human-written, 5/image | Fast pretrain, testing | `ariG23498/flickr8k` |
| **Pokemon BLIP** 🎮 | 833 | BLIP-generated | Pixel art style, small | `reach-vb/pokemon-blip-captions` |
| **CC3M** 🌐 | 3.3M | Web alt-text | Large-scale pretrain | `pixparse/cc3m-wds` |
| **CC12M** 🌍 | 12M | Web alt-text | Production-scale | `laion/conceptual-captions-12m-webdataset` |

**Not Recommended**:
- **CIFAR-100**: Image classification dataset (no captions) - poor for text-to-image learning

**Configuration** (in `config.yaml`):
```yaml
pretrain:
    data_source: caption
    dataset_name: ariG23498/flickr8k  # or other options above
    image_field: image
    caption_field: caption
    streaming: false
```

## 🚀 Quick Start

### Installation

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Two-Stage Training (Recommended)

**Stage 1: Pretrain** (Text-to-Image Dataset)
```bash
# Default: Flickr8k (8K images with captions)
uv run main.py --pretrain --epochs 100 --batch-size 32

# With Pokemon BLIP (pixel art style)
uv run main.py --pretrain --dataset reach-vb/pokemon-blip-captions

# With CC3M (large-scale, requires more time/resources)
uv run main.py --pretrain --dataset pixparse/cc3m-wds --epochs 50
```

**Stage 2: Fine-tune on Emoji**
```bash
# Default emoji dataset with pretrained checkpoint
uv run main.py --finetune \
    --checkpoint checkpoints/pretrain_flickr8k.pt \
    --epochs 100 \
    --batch-size 16

# With custom dataset
uv run main.py --finetune \
    --dataset user/my-emoji-dataset \
    --checkpoint checkpoints/pretrain_flickr8k.pt
```

### Interactive Demo

```bash
uv run main.py --demo
```

Enter prompts to generate images interactively. Type 'quit' to exit.

## 📖 Training Guide

### Two-Stage Training Pipeline

For best results with limited emoji data, use the two-stage approach:

**Stage 1: Pretraining** (Image-Caption Dataset)

```bash
# Default: Flickr8k (recommended for initial testing)
uv run main.py --pretrain --epochs 100 --batch-size 32

# Pokemon BLIP (pixel art style, very fast)
uv run main.py --pretrain \
    --dataset reach-vb/pokemon-blip-captions \
    --epochs 200 \
    --batch-size 16

# CC3M (large-scale, requires more resources)
uv run main.py --pretrain \
    --dataset pixparse/cc3m-wds \
    --epochs 50 \
    --batch-size 64
```

| Dataset | Images | Captions | Best For |
| :--- | :---: | :---: | :--- |
| **Flickr8k** ⭐ | 8K | Human-written | Fast testing, quality captions |
| **Pokemon BLIP** 🎮 | 833 | BLIP | Pixel art style, very fast |
| **CC3M** 🌐 | 3.3M | Web | Large-scale pretrain |

**Stage 2: Fine-tuning**
```bash
# Default: Emoji-32 dataset with pretrained checkpoint
uv run main.py --finetune \
    --checkpoint checkpoints/pretrain_flickr8k.pt \
    --epochs 100 \
    --batch-size 16

# Override hyperparameters
uv run main.py --finetune \
    --checkpoint checkpoints/pretrain_flickr8k.pt \
    --epochs 200 \
    --batch-size 16 \
    --learning-rate 1e-5

# Use custom dataset
uv run main.py --finetune \
    --dataset user/my-emoji-dataset \
    --checkpoint checkpoints/pretrain_flickr8k.pt
```

### Training Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--epochs` | Number of training epochs | config default |
| `--batch-size` | Batch size | config default |
| `--learning-rate` | Learning rate | config default |
| `--dataset` | Dataset path or name | config default |
| `--checkpoint` | Pretrained checkpoint path (finetune only) | config default |
| `--wandb` | Enable wandb logging | disabled |
| `--wandb-project` | Wandb project name | text-to-emoji |
| `--wandb-run-name` | Wandb run name | auto-generated |

### Weights & Biases Logging

Enable experiment tracking with [Weights & Biases](https://wandb.ai):

```bash
# Basic usage
uv run main.py --pretrain --wandb

# With custom project and run name
uv run main.py --pretrain --wandb --wandb-project my-project --wandb-run-name exp-1

# Fine-tuning with wandb
uv run main.py --finetune --checkpoint checkpoints/pretrain.pt --wandb
```

**Logged Metrics:**

| Metric | Description | Frequency |
| :--- | :--- | :--- |
| `train/loss` | Training loss | Every step |
| `train/learning_rate` | Current learning rate | Every step |
| `train/global_step` | Global step counter | Every step |
| `epoch/avg_loss` | Average epoch loss | Every epoch |
| `epoch/epoch` | Current epoch | Every epoch |
| `epoch/cfg_probability` | CFG dropout probability | Every epoch |

### Generating Images

```bash
# Single prompt
uv run main.py --generate --prompt "a cute robot"

# Multiple prompts (comma-separated)
uv run main.py --generate --prompt "rocket,cat,ghost"

# With custom checkpoint
uv run main.py --generate --prompt "star" --checkpoint checkpoints/model_best.pt

# Multiple samples
uv run main.py --generate --prompt "heart" --num-samples 4

# Custom sampling steps
uv run main.py --generate --prompt "fire" --steps 100
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

All configuration is managed via `config.yaml`. Edit this file to change training/generation settings:

```yaml
# config.yaml

# Training stage: "pretrain" or "finetune"
training_stage: pretrain

# Model type: "dit" (standard DiT) or "mmdit" (Multi-Modal DiT from SD3)
model_type: mmdit

# Pretraining Settings (Stage 1)
pretrain:
  data_source: cifar100
  epochs: 200
  batch_size: 4
  learning_rate: 1.0e-4
  ...

# Fine-tuning Settings (Stage 2)
finetune:
  data_source: huggingface
  dataset_name: junyeong-nero/emoji-32
  epochs: 100
  ...

# Common Settings (shared by both stages)
common:
  model_size: S
  patch_size: 2
  guidance_scale: 7.5
  use_ema: true

  # MMDiT-specific settings (only used when model_type: mmdit)
  qk_rmsnorm: true          # RMSNorm for QK attention (SD3-style)
  register_tokens: 0        # Register tokens (0 or 4)
```

**CLI overrides config.yaml settings:**
```bash
# Override any config value
uv run main.py --pretrain --epochs 50 --batch-size 32 --learning-rate 1e-4
uv run main.py --finetune --checkpoint checkpoints/pretrain.pt --epochs 200
```

## 📁 Project Structure

```
text-to-emoji/
├── main.py                 # Main entry point with CLI arguments
├── config.yaml            # Configuration file (pretrain/finetune/common settings)
├── src/
│   ├── config.py          # Configuration classes
│   ├── data/
│   │   ├── dataset.py     # Dataset loaders (emoji, CIFAR-100)
│   │   └── loader.py      # Data loading utilities
│   ├── models/
│   │   ├── diffusion.py   # Diffusion process
│   │   ├── factory.py     # Model factory (DiT/MMDiT)
│   │   ├── vanilla_dit.py # Standard DiT implementation
│   │   └── mmdit.py       # MMDiT implementation (SD3)
│   ├── text_encoder/
│   │   └── clip_encoder.py # CLIP text encoder
│   ├── training/
│   │   ├── train.py       # Training script
│   │   ├── ema.py         # Exponential Moving Average
│   │   └── checkpoint.py  # Checkpoint saving/loading
│   └── inference/
│       └── generate.py    # Generation script
├── scripts/
│   ├── install.sh         # Installation script
│   ├── train-pretrain.sh  # Pretraining script
│   └── train-finetune.sh  # Fine-tuning script
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
- **MMDiT**: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" - Stability AI (2024)
  - Paper: [https://arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)
  - Implementation: [lucidrains/mmdit](https://github.com/lucidrains/mmdit)
- **Vision Transformers Need Registers**: "Vision Transformers Need Registers" - Meta AI (2023)
  - Paper: [https://arxiv.org/abs/2309.16588](https://arxiv.org/abs/2309.16588)
- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (2020)
- **CFG**: Ho et al., "Classifier-Free Diffusion Guidance" (2021)
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)

## 📄 License

MIT License - See LICENSE file for details.

---

**Note**: Generated images are native 32×32 pixels. For display, they're upscaled to larger sizes using nearest-neighbor interpolation to preserve the pixel art aesthetic.
