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

For script wrappers and examples, see [`scripts/README.md`](../scripts/README.md).

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

### 4. Stage 3: Motion Module Training (Optional GIF Extension)

```bash
# Basic Motion training (requires VAE and Diffusion)
uv run main.py --train-motion --epochs 100 --batch-size 8

# With memory optimizations (gradient checkpointing)
uv run main.py --train-motion --epochs 100 --batch-size 2 --gradient-checkpointing
```

### 5. Image & GIF Generation

```bash
# Generate single image
uv run main.py --generate --prompt "a cute cat"

# Generate GIF animation
uv run main.py --generate-gif --prompt "a cat walking" --frames 16 --fps 8

# With options
uv run main.py --generate \
    --prompt "a beautiful landscape" \
    --steps 50 \
    --guidance 7.5 \
    --seed 42
```

### 6. Upload to HuggingFace Hub

```bash
# Upload after training
uv run main.py --train-vae --push-to-hub --hub-model-id username/my-vae
uv run main.py --train-diffusion --push-to-hub --hub-model-id username/my-diffusion
uv run main.py --train-motion --push-to-hub --hub-model-id username/my-motion
```

---

## Configuration

### config.yaml Structure

```yaml
# Current training stage
training_stage: vae_train  # vae_train, diffusion_train, or motion_train

# VAE training settings
vae_train:
    ...

# Diffusion training settings
diffusion_train:
    ...

# Motion Module training settings
motion_train:
    base_checkpoint: checkpoints/diffusion.pt
    motion_num_layers: 2
    num_frames: 16
    batch_size: 8
    learning_rate: 1.0e-4
```

---

## Hyperparameters

### VAE Training
...

### Diffusion Training
...

### Motion Module Training

| Parameter | Default | Recommended Range | Description |
|-----------|---------|-------------------|-------------|
| `epochs` | 100 | 50-200 | Number of training epochs |
| `batch_size` | 8 | 2-16 | Batch size (smaller than image due to VRAM) |
| `learning_rate` | 1e-4 | 5e-5 ~ 2e-4 | Learning rate |
| `num_frames` | 16 | 8-32 | Number of frames in a video clip |

---

## Hardware Requirements

### GPU Memory

| Stage | Model Size | Batch Size | VRAM |
|-------|------------|------------|------|
| VAE | - | 128 | ~8GB |
| Diffusion | B | 32 | ~12GB |
| Motion (16 frames)| B | 4 | ~16GB |
| Motion (16 frames)| B | 1 (w/ Grad CKPT) | ~8GB |

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

- [Architecture Deep Dive](./architecture.md) - Overall system architecture
- [VAE Documentation](./models/VAE.md) - VAE details
- [MMDiT Documentation](./models/MMDiT.md) - Transformer details
- [Diffusion Documentation](./models/Diffusion.md) - Rectified Flow details
- [Inference Deep Dive](./inference.md) - Image generation details

---

## References

- [Stable Diffusion 3 Paper](https://arxiv.org/abs/2403.03206)
- [DiT Paper](https://arxiv.org/abs/2212.09748)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [VAE Paper](https://arxiv.org/abs/1312.6114)
