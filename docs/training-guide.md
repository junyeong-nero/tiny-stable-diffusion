# Training Quick Start Guide

> A quick start guide for training tiny-stable-diffusion.
> For more details, see [training-pipeline.md](./training-pipeline.md).

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Hyperparameters](#hyperparameters)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Environment Setup

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Or use pip
pip install -e .
```

### 2. Stage 1: VAE Training

```bash
# Basic VAE training
uv run main.py --train-vae --epochs 100 --batch-size 32

# With Wandb logging
uv run main.py --train-vae --epochs 100 --batch-size 32 --wandb

# With custom dataset
uv run main.py --train-vae --epochs 100 --dataset reach-vb/pokemon-blip-captions
```

### 3. Stage 2: Diffusion Training

```bash
# Basic Diffusion training (requires VAE)
uv run main.py --train-diffusion --epochs 200 --batch-size 32

# Specify VAE checkpoint
uv run main.py --train-diffusion --vae-checkpoint checkpoints/vae.pt --epochs 200
```

### 4. Image Generation

```bash
# Generate single image
uv run main.py --generate --prompt "a cute cat"

# Generate multiple images
uv run main.py --generate --prompt "a robot,a sunset,a mountain" --num-samples 4

# With options
uv run main.py --generate \
    --prompt "a beautiful landscape" \
    --steps 50 \
    --guidance 7.5 \
    --seed 42
```

### 5. Upload to HuggingFace Hub

```bash
# Upload after VAE training
uv run main.py --train-vae --push-to-hub --hub-model-id username/my-vae

# Upload after Diffusion training
uv run main.py --train-diffusion --push-to-hub --hub-model-id username/my-diffusion
```

---

## Configuration

### config.yaml Structure

```yaml
# Current training stage
training_stage: vae_train  # or diffusion_train

# VAE training settings
vae_train:
    data_source: streaming_caption
    dataset_name: hmu013/LAION-300k
    image_size: 64
    latent_channels: 16
    epochs: 100
    batch_size: 128
    learning_rate: 4.0e-4
    kl_weight: 1.0e-6
    checkpoint_path: checkpoints/vae.pt

# Diffusion training settings
diffusion_train:
    model_type: mmdit  # dit or mmdit
    model_size: S      # S, B, L, XL
    epochs: 200
    batch_size: 32
    learning_rate: 1.0e-4
    guidance_scale: 7.5
    use_ema: true
    ema_decay: 0.9999
    vae_checkpoint: checkpoints/vae.pt
    checkpoint_path: checkpoints/diffusion.pt
```

### CLI Priority

CLI arguments override config.yaml values:

```bash
# Even if epochs=100 in config.yaml, CLI takes priority
uv run main.py --train-vae --epochs 50
```

---

## Hyperparameters

### VAE Training

| Parameter | Default | Recommended Range | Description |
|-----------|---------|-------------------|-------------|
| `epochs` | 100 | 50-200 | Number of training epochs |
| `batch_size` | 128 | 32-256 | Batch size |
| `learning_rate` | 4e-4 | 1e-4 ~ 1e-3 | Learning rate |
| `kl_weight` | 1e-6 | 1e-7 ~ 1e-5 | KL loss weight |

### Diffusion Training

| Parameter | Default | Recommended Range | Description |
|-----------|---------|-------------------|-------------|
| `epochs` | 200 | 100-500 | Number of training epochs |
| `batch_size` | 32 | 16-64 | Batch size |
| `learning_rate` | 1e-4 | 5e-5 ~ 3e-4 | Learning rate |
| `guidance_scale` | 7.5 | 3.0-15.0 | CFG scale |
| `cfg_probability` | 0.1 | 0.05-0.2 | CFG dropout probability |
| `ema_decay` | 0.9999 | 0.999-0.9999 | EMA decay rate |

### Model Sizes

| Size | Layers | Hidden | Heads | Params | VRAM |
|------|--------|--------|-------|--------|------|
| **S** | 12 | 384 | 6 | ~40M | ~4GB |
| B | 12 | 768 | 12 | ~160M | ~8GB |
| L | 24 | 1024 | 16 | ~560M | ~16GB |
| XL | 28 | 1152 | 16 | ~820M | ~24GB |

---

## Hardware Requirements

### GPU Memory

| Stage | Model Size | Batch Size | VRAM |
|-------|------------|------------|------|
| VAE | - | 32 | ~4GB |
| VAE | - | 128 | ~8GB |
| Diffusion | S | 32 | ~6GB |
| Diffusion | B | 32 | ~12GB |
| Diffusion | L | 16 | ~20GB |

### Recommended Specifications

**Minimum:**
- GPU: RTX 3060 12GB or higher
- RAM: 16GB
- Storage: 20GB SSD

**Recommended:**
- GPU: RTX 3090 24GB or higher
- RAM: 32GB
- Storage: 50GB SSD

### Apple Silicon (MPS)

```bash
# MPS auto-detection
uv run main.py --train-vae --batch-size 32

# Or set in config.yaml
# device: mps
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Solution 1: Reduce batch size
--batch-size 16

# Solution 2: Enable mixed precision
# In config.yaml
mixed_precision: true

# Solution 3: Use smaller model
model_size: S
```

### Loss Not Decreasing

1. **Check learning rate**: Too high = unstable, too low = slow
2. **Check KL weight**: 1e-6 recommended for VAE
3. **Check dataset**: Ensure images are loading correctly

### NaN Loss

```bash
# Solution: Lower learning rate
--learning-rate 5e-5

# Or add gradient clipping (requires code modification)
```

### Poor Generation Quality

1. Train for **more epochs**
2. Adjust **guidance scale**: 7.5-10.0
3. Increase **steps**: 50-100
4. Verify **EMA weights** are being used

### CLIP Installation Error

```bash
# Install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git
```

---

## Best Practices

### 1. Progressive Training

```bash
# Step 1: Test with small dataset
uv run main.py --train-vae --epochs 10 --dataset reach-vb/pokemon-blip-captions

# Step 2: Full training with large dataset
uv run main.py --train-vae --epochs 100 --dataset hmu013/LAION-300k
```

### 2. Checkpoint Management

```bash
# Auto-save during training: based on best loss
# Location: checkpoints/vae.pt, checkpoints/diffusion.pt

# Backup to HuggingFace Hub
--push-to-hub --hub-model-id username/model-name
```

### 3. Monitoring

```bash
# Monitor training with Wandb
--wandb --wandb-project tiny-stable-diffusion

# Check samples
# samples/vae_epoch_N/: VAE reconstruction
# samples/epoch_N/: Diffusion generation
```

### 4. Reproducibility

```bash
# Fix seed
# In config.yaml
seed: 42

# Or during generation
--seed 42
```

---

## Additional Documentation

- [Architecture Deep Dive](./architecture.md) - Model architecture details
- [Training Pipeline Deep Dive](./training-pipeline.md) - Training process details
- [Inference Deep Dive](./inference.md) - Image generation details

---

## References

- [Stable Diffusion 3 Paper](https://arxiv.org/abs/2403.03206)
- [DiT Paper](https://arxiv.org/abs/2212.09748)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [VAE Paper](https://arxiv.org/abs/1312.6114)
