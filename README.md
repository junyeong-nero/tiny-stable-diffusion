# tiny-stable-diffusion

> **Stable Diffusion 3 from Scratch** - A minimal educational implementation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview

**tiny-stable-diffusion** is a project that implements the Stable Diffusion 3 pipeline from scratch for educational purposes. It follows the same architecture as the actual SD3 while being lightweight at 64x64 resolution, making it trainable on consumer GPUs.

### Core Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Stable Diffusion 3 Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. VAE Training (Stage 1)                                              │
│     Image → Encoder → Latent Space → Decoder → Reconstructed Image      │
│                                                                         │
│  2. Diffusion Training (Stage 2)                                        │
│     Image → [Frozen VAE] → Latent → DiT + Text → Noise Prediction       │
│                                                                         │
│  3. Generation (Inference)                                              │
│     Noise → DiT Denoise → Clean Latent → [VAE Decoder] → Image          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Latent Space Diffusion?

| Pixel Space | Latent Space |
|-------------|--------------|
| 64×64×3 = 12,288 dimensions | 8×8×16 = 1,024 dimensions |
| High computation | **12x more efficient** |
| High memory usage | **Memory efficient** |
| High resolution difficult | **High resolution possible** |

By compressing images with VAE and performing diffusion in latent space, training becomes much more efficient.

---

## Architecture

### Overall System Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    tiny-stable-diffusion                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: VAE Training                                           │
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │  Image  │  ->  │ Encoder │  ->  │ Latent  │                │
│   │ (64×64) │      │         │      │(16,8,8) │                │
│   └─────────┘      └─────────┘      └────┬────┘                │
│                                          │                      │
│                                          v                      │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │  Recon  │  <-  │ Decoder │  <-  │ Sample  │                │
│   │  Image  │      │         │      │  z~N    │                │
│   └─────────┘      └─────────┘      └─────────┘                │
│                                                                 │
│   Loss = MSE(Image, Recon) + β × KL(q(z|x) || p(z))            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Diffusion Training (Latent Space)                      │
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │  Image  │  ->  │   VAE   │  ->  │ Latent  │                │
│   │ (64×64) │      │ Encoder │      │(16,8,8) │                │
│   └─────────┘      └─────────┘      └────┬────┘                │
│                     (frozen)              │                     │
│                                          v                      │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │Predicted│  <-  │   DiT   │  <-  │ Noisy   │                │
│   │  Noise  │      │ + Text  │      │ Latent  │                │
│   └─────────┘      └─────────┘      └─────────┘                │
│                                                                 │
│   Loss = MSE(Predicted Noise, Actual Noise) × SNR Weight        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Generation                                             │
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌───────┐ │
│   │ Random  │  ->  │   DiT   │  ->  │  Clean  │  ->  │ Image │ │
│   │  Noise  │      │ Denoise │      │ Latent  │      │(64×64)│ │
│   └─────────┘      └─────────┘      └─────────┘      └───────┘ │
│                    (DDIM 50 steps)       │                     │
│                         ^                 v                     │
│                    ┌─────────┐      ┌─────────┐                │
│                    │  Text   │      │   VAE   │                │
│                    │  CLIP   │      │ Decoder │                │
│                    └─────────┘      └─────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### VAE (Variational AutoEncoder)

SD3-style AutoencoderKL implementation:

```
Encoder:
  Conv3x3(3→64) → ResBlock×2 → Downsample
                → ResBlock×2 → Downsample
                → ResBlock×2 → Downsample
                → ResBlock×2 → SelfAttention → ResBlock
                → Conv3x3(512→32) → [mean, logvar]

Decoder:
  Conv3x3(16→512) → ResBlock → SelfAttention → ResBlock
                  → ResBlock×3 → Upsample
                  → ResBlock×3 → Upsample
                  → ResBlock×3 → Upsample
                  → Conv3x3(64→3)
```

| Setting | Value |
|---------|-------|
| Input | 64×64×3 RGB |
| Latent | 8×8×16 |
| Compression | f8 (8x compression) |
| Base channels | 64 |
| Channel multipliers | [1, 2, 4, 4] |
| Parameters | ~21M |

### Diffusion Transformer (DiT / MMDiT)

Two architectures are supported:

#### DiT (Vanilla) - Cross-Attention Method
```
Image Tokens → Self-Attention → Cross-Attention(with Text) → MLP → Output
```

| Size | Layers | Hidden | Heads | Params | Use Case |
|------|--------|--------|-------|--------|----------|
| **S** | 12 | 384 | 6 | **39.9M** | Default, quick experiments |
| B | 12 | 768 | 12 | **158.8M** | Medium scale |
| L | 24 | 1024 | 16 | **559.0M** | High quality |
| XL | 28 | 1152 | 16 | **824.2M** | Best quality |

#### MMDiT (SD3 Style) - Joint Attention Method
```
[Text Tokens, Image Tokens] → Joint Self-Attention → Separate MLPs → Output
```

| Size | Layers | Hidden | Heads | Params | Use Case |
|------|--------|--------|-------|--------|----------|
| **S** | 12 | 384 | 6 | **87.0M** | Default, recommended |
| B | 12 | 768 | 12 | **186.9M** | Medium scale |
| L | 24 | 1024 | 16 | **558.9M** | High quality |
| XL | 28 | 1152 | 16 | **780.1M** | Best quality |

**DiT vs MMDiT Comparison:**
| Feature | DiT | MMDiT |
|---------|-----|-------|
| Text Processing | Cross-Attention | Joint Attention |
| Architecture | Separate attention | Unified attention |
| Training Stability | Good | Better (QK-RMSNorm) |
| Actual SD3 | No | Yes |

**DiT Block Structure:**
```
Input → LayerNorm → Self-Attention → + → LayerNorm → Cross-Attention → + → LayerNorm → MLP → + → Output
         ↑                           |                                  |               |
         └── AdaLN-Zero (timestep) ──┴──────────────────────────────────┴───────────────┘
                                     (text conditioning)
```

**MMDiT Block Structure:**
```
[Text, Image] → Joint LayerNorm → Joint Self-Attention → Split → Separate MLPs → Output
                     ↑                                              |
                     └──────── Time Conditioning ───────────────────┘
```

### Comparison with Actual SD3

| Component | Stable Diffusion 3 | tiny-stable-diffusion |
|-----------|-------------------|----------------------|
| Image Size | 1024×1024 | **64×64** |
| VAE Latent Channels | 16 | **16** |
| VAE Compression | f8 | **f8** |
| Diffusion Architecture | MMDiT | **DiT / MMDiT** |
| Text Encoder | T5-XXL + CLIP-G + CLIP-L | **CLIP ViT-B/32** |
| Total Parameters | 2B+ | **~60M** |
| Training Time | Thousands of GPU-hours | **A few hours** |

---

## Quick Start

### Installation

```bash
# Install uv package manager (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Or use pip
pip install -e .
```

### Full Training Pipeline

```bash
# Step 1: VAE training (learn image compression)
uv run main.py --train-vae --epochs 100 --batch-size 32

# Step 2: Diffusion training (learn noise removal in latent space)
uv run main.py --train-diffusion --epochs 200 --batch-size 32

# Step 3: Generate images
uv run main.py --generate --prompt "a cute cat sitting on a couch"
```

### Quick Test

```bash
# Quick test with small dataset
uv run main.py --train-vae --epochs 10 --dataset reach-vb/pokemon-blip-captions
uv run main.py --train-diffusion --epochs 20 --dataset reach-vb/pokemon-blip-captions
```

---

## Training Guide

### Stage 1: VAE Training

VAE learns to compress images to latent space and reconstruct them.

```bash
uv run main.py --train-vae \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4
```

**Key Settings (config.yaml):**
```yaml
vae_train:
    image_size: 64           # Input image size
    latent_channels: 16      # Latent channel count (SD3 style)
    vae_ch: 64               # VAE base channels
    vae_ch_mult: [1, 2, 4, 4]  # Channel multiplier ratio
    kl_weight: 1.0e-6        # KL divergence weight
    epochs: 100
    batch_size: 32
    learning_rate: 1.0e-4
    checkpoint_path: checkpoints/vae.pt
```

**Training Tips:**
- If `kl_weight` is too large, reconstruction quality degrades
- If `kl_weight` is too small, posterior collapse may occur
- Recommended starting value: `1e-6`

**Output:**
- `checkpoints/vae.pt`: Trained VAE checkpoint
- `samples/vae_epoch_N/`: Reconstruction sample images

### Stage 2: Diffusion Training

Train the diffusion model in latent space using the pre-trained VAE.

```bash
uv run main.py --train-diffusion \
    --vae-checkpoint checkpoints/vae.pt \
    --epochs 200 \
    --batch-size 32
```

**Key Settings (config.yaml):**
```yaml
diffusion_train:
    image_size: 64           # Original image size
    latent_size: 8           # Latent space size (64/8)
    in_channels: 16          # Latent channels (match VAE)

    # CFG (Classifier-Free Guidance) settings
    initial_cfg_prob: 0.0    # Initial unconditional dropout probability
    final_cfg_prob: 0.1      # Final unconditional dropout probability
    cfg_warmup_epochs: 10    # CFG warmup period

    # VAE
    vae_checkpoint: checkpoints/vae.pt

    epochs: 200
    batch_size: 32
    learning_rate: 1.0e-4
    checkpoint_path: checkpoints/diffusion.pt
```

**Training Process:**
1. Convert image to latent using frozen VAE encoder
2. Add noise to latent
3. DiT predicts the noise
4. Train with MSE loss

**Output:**
- `checkpoints/diffusion.pt`: Trained diffusion checkpoint
- `samples/epoch_N/`: Generated sample images

---

## Generation

### Basic Generation

```bash
# Single prompt
uv run main.py --generate --prompt "a photo of a cat"

# Multiple prompts
uv run main.py --generate --prompt "cat,dog,sunset,mountain"

# Multiple samples per prompt
uv run main.py --generate --prompt "a robot" --num-samples 4
```

### Advanced Options

```bash
uv run main.py --generate \
    --prompt "a beautiful landscape with mountains" \
    --checkpoint checkpoints/diffusion.pt \
    --vae-checkpoint checkpoints/vae.pt \
    --steps 100 \           # diffusion steps (default: 50)
    --guidance 7.5 \        # CFG scale (default: 7.5)
    --seed 42 \             # seed for reproducibility
    --output my_image.png
```

### VAE Reconstruction

Test VAE quality by reconstructing images:

```bash
# Single image reconstruction
uv run main.py --reconstruct-vae \
    --input samples/original/sample_000_cattle.png \
    --output samples/reconstructed.png \
    --vae-checkpoint checkpoints/vae.pt

# Using the convenience script
./scripts/inference-vae.sh samples/original/sample_000_cattle.png

# Batch reconstruct all sample images
./scripts/inference-vae.sh --all
```

### Interactive Demo

```bash
uv run main.py --demo
```

Enter prompts to generate images in real-time.

---

## Configuration

All settings are managed in `config.yaml`:

```yaml
# tiny-stable-diffusion Configuration

# Current training stage: "vae_train" or "diffusion_train"
training_stage: vae_train

# ═══════════════════════════════════════════════════════════════
# Stage 1: VAE Training
# ═══════════════════════════════════════════════════════════════
vae_train:
    data_source: streaming_caption
    dataset_name: hmu013/LAION-300k
    image_field: png
    caption_field: json
    image_size: 64
    latent_channels: 16
    vae_ch: 64
    vae_ch_mult: [1, 2, 4, 4]
    kl_weight: 1.0e-6
    epochs: 100
    batch_size: 128
    learning_rate: 4.0e-4
    checkpoint_path: checkpoints/vae.pt

# ═══════════════════════════════════════════════════════════════
# Stage 2: Diffusion Training
# ═══════════════════════════════════════════════════════════════
diffusion_train:
    model_type: mmdit        # "dit" or "mmdit"
    data_source: streaming_caption
    dataset_name: visual-layer/oxford-iiit-pet-vl-enriched
    image_field: image
    caption_field: caption_enriched
    image_size: 64
    latent_size: 8
    in_channels: 16
    initial_cfg_prob: 0.0
    final_cfg_prob: 0.1
    cfg_warmup_epochs: 10
    vae_checkpoint: checkpoints/vae.pt
    epochs: 200
    batch_size: 32
    learning_rate: 1.0e-4
    checkpoint_path: checkpoints/diffusion.pt
    # Model settings
    model_size: S           # S, B, L, XL
    patch_size: 2
    num_timesteps: 1000
    beta_schedule: cosine
    guidance_scale: 7.5
    use_ema: true
    ema_decay: 0.9999
    device: auto
    seed: 42
    validation_prompts:
        - a photo of a cat
        - a rocket flying in space
        - a robot with blue eyes
    validation_interval: 10
```

---

## Dataset

### Recommended Datasets

| Dataset | Size | Features | Use Case |
|---------|------|----------|----------|
| **hmu013/LAION-300k** | 300K images | Large-scale, diverse | VAE training |
| **visual-layer/oxford-iiit-pet-vl-enriched** | 7.4K images | Pet images, enriched captions | Diffusion training |
| reach-vb/pokemon-blip-captions | 833 images | Pixel art style | Quick testing |

### Changing Dataset

```bash
# Change via CLI
uv run main.py --train-vae --dataset hmu013/LAION-300k

# Or modify config.yaml
vae_train:
    data_source: streaming_caption
    dataset_name: hmu013/LAION-300k
    image_field: png
    caption_field: json
```

---

## Project Structure

```
tiny-stable-diffusion/
├── main.py                         # CLI entry point
├── config.yaml                     # Configuration file
├── pyproject.toml                  # Project metadata
├── README.md                       # This document
│
├── src/
│   ├── models/
│   │   ├── vae.py                  # VAE (AutoencoderKL)
│   │   │   ├── Encoder             # Image → latent
│   │   │   ├── Decoder             # Latent → image
│   │   │   └── training_loss()     # VAE loss function
│   │   │
│   │   ├── diffusion.py            # DDPM/DDIM process
│   │   │   ├── q_sample()          # Forward diffusion
│   │   │   ├── p_sample()          # Reverse (DDPM)
│   │   │   ├── ddim_sample()       # Reverse (DDIM)
│   │   │   └── sample()            # Full generation loop
│   │   │
│   │   ├── factory.py              # DiT model factory
│   │   ├── vanilla_dit.py          # Standard DiT implementation
│   │   ├── mmdit.py                # Multi-Modal DiT (SD3)
│   │   └── layers.py               # Common layers
│   │
│   ├── training/
│   │   ├── vae_trainer.py          # VAE training loop
│   │   ├── trainer.py              # Diffusion training loop
│   │   ├── ema.py                  # Exponential Moving Average
│   │   └── checkpoint.py           # Checkpoint management
│   │
│   ├── inference/
│   │   ├── generator.py            # Image generation
│   │   └── vae_inference.py        # VAE reconstruction
│   │
│   ├── text_encoder/
│   │   └── clip_encoder.py         # CLIP text encoder
│   │
│   ├── data/
│   │   ├── dataset.py              # Dataset loader
│   │   └── loader.py               # DataLoader utilities
│   │
│   ├── config/
│   │   ├── loader.py               # config.yaml loader
│   │   └── dataclasses.py          # Configuration dataclasses
│   │
│   └── utils/
│       └── common.py               # Common utilities
│
├── checkpoints/                    # Saved models
│   ├── vae.pt                      # VAE checkpoint
│   └── diffusion.pt                # Diffusion checkpoint
│
├── samples/                        # Generated samples
│   ├── original/                   # Original sample images (64x64)
│   ├── vae_reconstructed/          # VAE reconstruction outputs
│   ├── vae_epoch_N/                # VAE training samples
│   └── epoch_N/                    # Diffusion generation results
│
├── scripts/                        # Utility scripts
│   ├── inference-vae.sh            # VAE inference script
│   ├── inference-diffusion.sh      # Diffusion inference script
│   ├── train-vae.sh                # VAE training script
│   ├── train-diffusion.sh          # Diffusion training script
│   └── download-samples.py         # Download sample images
│
└── tests/                          # Test code
```

---

## CLI Reference

```
usage: main.py [-h] [--train-vae] [--train-diffusion] [--train]
               [--generate] [--demo] [options]

tiny-stable-diffusion - Stable Diffusion 3 from Scratch

Training:
  --train-vae           Stage 1: Train VAE
  --train-diffusion     Stage 2: Train Diffusion (requires VAE)
  --train               Use training_stage from config.yaml

  --epochs N            Number of epochs
  --batch-size N        Batch size
  --learning-rate F     Learning rate
  --dataset NAME        Dataset name
  --vae-checkpoint P    VAE checkpoint path

Generation:
  --generate            Generate images
  --reconstruct-vae     Reconstruct image through VAE
  --demo                Interactive demo

  --prompt TEXT         Prompt (comma-separated)
  --input PATH          Input image (for --reconstruct-vae)
  --num-samples N       Samples per prompt
  --steps N             Diffusion steps (default: 50)
  --guidance F          CFG scale (default: 7.5)
  --seed N              Random seed
  --checkpoint P        Diffusion checkpoint
  --output PATH         Output file path

Logging:
  --wandb               Enable Wandb logging
  --wandb-project NAME  Wandb project name
  --wandb-run-name NAME Wandb run name
```

---

## Technical Details

### Diffusion Process

**Forward Process (adding noise):**
```
x_t = √(α̅_t) × x_0 + √(1 - α̅_t) × ε
```

**Reverse Process (DDIM):**
```
x_{t-1} = √(α̅_{t-1}) × pred_x_0 + √(1 - α̅_{t-1} - σ²) × ε_θ + σ × z
```

**Min-SNR Weighting:**
```python
snr = α̅_t / (1 - α̅_t)
weight = min(snr, γ) / snr  # γ = 5.0
loss = weight × MSE(ε_θ, ε)
```

### Classifier-Free Guidance (CFG)

```python
# During training: drop text condition with 10% probability
if random() < 0.1:
    text_embed = uncond_embed  # Empty string embedding

# During inference: combine conditional and unconditional predictions
noise_pred = uncond_pred + guidance_scale × (cond_pred - uncond_pred)
```

---

## References

- **Stable Diffusion 3**: [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)
- **DiT**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **DDIM**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- **VAE**: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Min-SNR**: [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

Issues and PRs are welcome. As this is an educational project, we prioritize code clarity and easy-to-understand implementations.
