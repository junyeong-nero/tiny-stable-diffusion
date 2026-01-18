# tiny-stable-diffusion

> **Stable Diffusion 3 from Scratch** - A minimal educational implementation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## Introduction

A lightweight, educational implementation of **Stable Diffusion 3** built from scratch in PyTorch. This project is designed to help you understand how modern text-to-image diffusion models work by providing a minimal yet complete implementation.

### Key Features

- **64×64 Resolution**: Optimized for fast training and experimentation on consumer GPUs
- **SD3 Architecture**: Implements the core components of Stable Diffusion 3
  - AutoencoderKL (VAE) for latent space compression
  - MMDiT (Multi-Modal Diffusion Transformer) for text-conditioned generation
  - CLIP text encoder for prompt understanding
- **Two-Stage Training**: Train VAE first, then diffusion model in latent space
- **Beginner-Friendly**: Clean, readable code with minimal dependencies

### Example Generation

> **Prompt**: "a Siamese cat with blue eyes"

<p align="center">
  <img src="assets/sample_0.png" alt="Sample 0" width="128">
  <img src="assets/sample_1.png" alt="Sample 1" width="128">
  <img src="assets/sample_2.png" alt="Sample 2" width="128">
  <img src="assets/sample_3.png" alt="Sample 3" width="128">
</p>

> ⚠️ **Note**: The current model is trained on [Oxford Pets dataset](https://huggingface.co/datasets/visual-layer/oxford-iiit-pet-vl-enriched) (7.4K images of cats & dogs). It works best with **pet-related prompts**.

---

## How to Use

### 1. Environment Setup

See `setup.sh` for detailed setup instructions.

```bash
# Quick setup
bash setup.sh
```

### 2. Download Pretrained Weights

Download the pretrained VAE and Diffusion model checkpoints:

```bash
# Download pretrained weights (coming soon)
# Place checkpoints in the checkpoints/ directory
```

### 3. Inference

#### VAE Reconstruction

Test the VAE encoder-decoder by reconstructing images:

```bash
uv run main.py --reconstruct --image path/to/image.png
```

#### Diffusion (Text-to-Image Generation)

Generate images from text prompts using the diffusion model:

```bash
uv run main.py --generate --prompt "a cute cat" --steps 50 --guidance 7.5
```

### 4. Web Demo

Launch the interactive Streamlit web interface:

```bash
uv run streamlit run src/demo/app.py
```

---

## How It Works

```
Training Pipeline:
1. VAE: Learn to compress images (64×64 → 8×8 latent)
2. Diffusion: Learn to generate images from text in latent space

Generation:
Text Prompt → CLIP → DiT Denoising → VAE Decode → Image
```

---

## Model Architecture

| Component | Parameters | Description |
|-----------|------------|-------------|
| VAE | ~21M | Image encoder/decoder (f8 compression) |
| MMDiT-B | ~187M | Text-conditioned diffusion transformer |
| CLIP | - | Text encoder (pretrained, frozen) |

**Comparison with SD3:**

| | Stable Diffusion 3 | tiny-stable-diffusion |
|---|---|---|
| Resolution | 1024×1024 | 64×64 |
| Parameters | 2B+ | ~200M |
| Training | 1000s GPU-hours | A few hours |

---

## Configuration

All settings in `config.yaml`:

```yaml
# Key settings
vae_train:
    dataset_name: hmu013/LAION-300k
    epochs: 100
    batch_size: 256

diffusion_train:
    dataset_name: visual-layer/oxford-iiit-pet-vl-enriched
    model_type: mmdit  # or "dit"
    model_size: B      # S, B, L, XL
    epochs: 200
    batch_size: 64
```

---

## CLI Commands

```bash
# Training
uv run main.py --train-vae              # Train VAE
uv run main.py --train-diffusion        # Train Diffusion

# Generation
uv run main.py --generate --prompt "..."
uv run main.py --generate --prompt "..." --steps 100 --guidance 7.5

# Demo
uv run main.py --demo                   # CLI demo
uv run streamlit run src/demo/app.py    # Web UI
```

---

## Project Structure

```
tiny-stable-diffusion/
├── main.py              # CLI entry point
├── config.yaml          # Configuration
├── src/
│   ├── models/          # VAE, DiT, Diffusion
│   ├── training/        # Training loops
│   ├── inference/       # Generation
│   └── demo/            # Streamlit app
├── checkpoints/         # Saved models
└── samples/             # Generated images
```

---

## References

- [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) - Rectified Flow Transformers
- [DiT](https://arxiv.org/abs/2212.09748) - Diffusion Transformers
- [DDPM](https://arxiv.org/abs/2006.11239) / [DDIM](https://arxiv.org/abs/2010.02502) - Diffusion Models

---

## License

MIT License
