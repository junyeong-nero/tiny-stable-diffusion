# tiny-stable-diffusion

> **Stable Diffusion 3 from Scratch** - A minimal educational implementation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## ⚡ TL;DR

A **200M parameter** implementation of Stable Diffusion 3 (SD3) trained on consumer GPUs.
It uses **Rectified Flow** and **MMDiT** architecture to generate **64×64 images**.

**Quick Start:**
```bash
# 1. Setup
bash setup.sh

# 2. Train VAE (or download weights)
uv run main.py --train-vae

# 3. Train Diffusion
uv run main.py --train-diffusion

# 4. Generate
uv run main.py --generate --prompt "a cute cat"
```

---

## Introduction

A lightweight, educational implementation of **Stable Diffusion 3** built from scratch in PyTorch. This project is designed to help you understand how modern text-to-image diffusion models work by providing a minimal yet complete implementation.

### Key Features

- **64×64 Resolution**: Optimized for fast training and experimentation on consumer GPUs
- **SD3 Architecture**: Implements the core components of Stable Diffusion 3
  - **VAE**: AutoencoderKL with f8 compression (16 latent channels)
  - **MMDiT**: Multi-Modal Diffusion Transformer with Joint Attention
  - **Rectified Flow**: Linear interpolation based diffusion training
- **Two-Stage Training**: Train VAE first, then diffusion model in latent space
- **Beginner-Friendly**: Clean, readable code with minimal dependencies

---

## Overall Pipeline

The system works in two distinct stages, mirroring the standard Latent Diffusion Model (LDM) approach but with SD3 improvements.

### 1. Training Pipeline

```mermaid
graph LR
    subgraph Stage 1: VAE
    I[Image 64px] --> E[Encoder]
    E --> L[Latent 8x8x16]
    L --> D[Decoder]
    D --> R[Recon Image]
    end
    
    subgraph Stage 2: Diffusion
    T[Text "a cat"] --> CLIP[CLIP Encoder]
    CLIP --> Emb[Text Embeds]
    L2[Latent] --> Noise[Add Noise]
    Noise --> MMDiT
    Emb --> MMDiT
    MMDiT --> Pred[Predict Velocity]
    end
```

### 2. Inference Pipeline

```
Prompt "a cat" ──► CLIP ──► Text Embeds ──┐
                                          ▼
Random Noise ──► MMDiT (Rectified Flow) ──► Denoised Latent ──► VAE Decoder ──► Image
```

---

## Usage

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

## Model Architecture Details

| Component | Parameters | Description |
|-----------|------------|-------------|
| **VAE** | ~21M | **AutoencoderKL**: Compresses 64×64 images to 8×8×16 latents. Uses a "f8" compression factor. |
| **MMDiT** | ~187M (Base) | **Multi-Modal DiT**: Uses Joint Attention to process text and image tokens simultaneously. |
| **CLIP** | 123M | **Text Encoder**: Frozen CLIP ViT-B/32 model for text embeddings. |

**Comparison with SD3:**

| Feature | Stable Diffusion 3 | tiny-stable-diffusion |
|---|---|---|
| **Resolution** | 1024×1024 | 64×64 |
| **Latent Channels** | 16 | 16 |
| **Model Size** | 2B+ | ~200M |
| **Training Cost** | Massive Cluster | Consumer GPU |

For detailed documentation, check the `docs/` folder:
- [**VAE Architecture**](docs/models/VAE.md)
- [**MMDiT Architecture**](docs/models/MMDiT.md)
- [**Diffusion Process**](docs/models/Diffusion.md)

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
├── docs/                # Documentation
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
