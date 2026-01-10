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

#### 📊 Dataset Scale Analysis

**How many images do you need for pretrain?**

Large-scale research typically uses:
- **DiT-XL/2** (256×256): 1.3M images (ImageNet)
- **Stable Diffusion**: 600M-2.3B images (LAION)
- **Scaling Law studies**: 108M+ images

But for **32×32 resolution**, requirements are much lower:
- 32×32 has **64× less pixels** than 256×256 (8² = 64)
- Simpler patterns, smaller model (~30M params for DiT-S)
- **Estimate: 20K-100K image-caption pairs** can produce meaningful results

#### ✅ Immediately Usable Datasets (Images Included)

| Dataset | Images | Captions | Effective Pairs | Format | Status |
|---------|--------|----------|-----------------|--------|--------|
| **jxie/flickr8k** ⭐ | 8,000 | 5 per image | **40,000** | Well-organized columns | ✅ Recommended |
| **jxie/flickr8k** | 8,000 | 5 per image | **40,000** | Captions as list | ✅ Alternative |
| **Pokemon BLIP** 🎮 | 833 | 1 per image | 833 | Single caption | ✅ Tested |

**Why jxie/flickr8k?**
- Captions organized as separate columns (`caption_0` to `caption_4`)
- Images pre-loaded as PIL objects (faster)
- Cleaner data structure for easy access
- One caption randomly selected per training iteration (automatic data augmentation)

#### ⚠️ Large-Scale Datasets (URLs Only - Download Required)

| Dataset | Images | Note |
|---------|--------|------|
| **CC3M** | 3.3M | Requires `img2dataset` to download actual images |
| **CC12M** | 12M | Requires `img2dataset` to download actual images |
| **LAION** | 5B | Requires `img2dataset` to download actual images |

Most large datasets only provide image URLs, not actual images. You'll need to:
1. Download images using tools like [`img2dataset`](https://github.com/rom1504/img2dataset)
2. Handle broken URLs (~20-40% failure rate)
3. Wait several hours to days for downloads

#### 🎯 Why Flickr8k is Sufficient

**For 32×32 DiT pretrain, Flickr8k (8K images, 40K captions) is a solid choice:**

✅ **Advantages**:
- **High-quality captions**: Human-written, 5 per image
- **Effective 40K training pairs**: With data augmentation, even more
- **Instant availability**: No download scripts needed
- **32×32 resolution advantage**: Simpler patterns, less data needed
- **Fast experimentation**: 4-8 hour training time

⚠️ **Limitations**:
- Smaller than typical pretrain datasets (50K-100K+ recommended)
- Limited scene/concept diversity
- Text-image alignment may be weaker than larger datasets

**Performance Expectations**:

```
Pokemon (833)     ▓░░░░░░░░░  10% - Fine-tuning only
Flickr8k (8K)     ▓▓▓░░░░░░░  30% - Basic concepts, color, shape
Flickr30k (30K)   ▓▓▓▓▓░░░░░  50% - Decent pretrain
CC3M (3.3M)       ▓▓▓▓▓▓▓▓░░  80% - Strong pretrain (requires download)
CC12M (12M)       ▓▓▓▓▓▓▓▓▓▓ 100% - Production-ready (requires download)
```

#### 📝 Recommended Strategy

**Option 1: Start with Flickr8k** (Recommended for beginners) ⚡
```bash
python main.py --pretrain --epochs 200 --batch-size 32
```
- Training time: 4-8 hours
- Good for validation and testing
- If results are promising, proceed to fine-tuning

**Option 2: Download CC3M** (For serious pretrain) 🔧
```bash
# Install img2dataset
pip install img2dataset

# Download CC3M images (takes several hours)
img2dataset --url_list cc3m.parquet \
    --output_folder data/cc3m \
    --resize_mode no \
    --thread_count 16

# Then train
python main.py --pretrain --dataset data/cc3m
```

**Option 3: Hybrid Approach** (Best of both worlds) 🎯
```bash
# Quick test with Flickr8k
python main.py --pretrain --dataset jxie/flickr8k --epochs 50

# If results are good, extend training or switch to CC3M
python main.py --pretrain --dataset data/cc3m --epochs 100
```

#### ⚙️ Configuration

Edit `config.yaml`:
```yaml
pretrain:
    data_source: caption
    dataset_name: jxie/flickr8k  # Recommended: well-organized format
    # Alternative: jxie/flickr8k (captions as list)
    image_field: image
    caption_field: caption  # Auto-detects caption_0, caption_1, etc.
    streaming: false
    epochs: 200  # More epochs for smaller dataset
```

**Not Recommended**:
- **CIFAR-100**: Image classification dataset (no captions) - poor for text-to-image learning

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
# Default: Flickr8k (8K images, 40K captions - tested and working)
uv run main.py --pretrain --epochs 200 --batch-size 32

# Pokemon BLIP (very small, pixel art style - for quick testing only)
uv run main.py --pretrain --dataset reach-vb/pokemon-blip-captions --epochs 500

# Note: CC3M/CC12M require separate image download (see Dataset section)
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
# Flickr8k: 8K images with 40K human-written captions (recommended starting point)
uv run main.py --pretrain --epochs 200 --batch-size 32

# Pokemon BLIP: 833 pixel art images (for quick testing)
uv run main.py --pretrain \
    --dataset reach-vb/pokemon-blip-captions \
    --epochs 500 \
    --batch-size 16
```

**Why use 200+ epochs for Flickr8k?**
- Smaller dataset (8K vs typical 100K+) requires more iterations
- High-quality captions enable effective learning even with fewer images
- Each image has 5 captions = 40K effective training pairs
- 32×32 resolution is simpler to learn than higher resolutions

**Dataset Comparison:**

| Dataset | Images | Effective Pairs | Training Time | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Flickr8k** ⭐ | 8K | 40K | 4-8 hours | ✅ Ready to use |
| **Pokemon BLIP** 🎮 | 833 | 833 | 30-60 min | ✅ Ready to use |
| **CC3M** 🌐 | 3.3M | 3.3M | 1-2 days | ⚠️ Requires img2dataset download |

See the **Dataset** section above for detailed analysis of why Flickr8k is sufficient for 32×32 pretrain.

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
