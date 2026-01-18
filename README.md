# tiny-stable-diffusion

> **Stable Diffusion 3 from Scratch** - A minimal educational implementation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## What is this?

A lightweight implementation of Stable Diffusion 3 for learning purposes. Train your own text-to-image model on a consumer GPU!

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

## Quick Start

### 1. Installation

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using pip
pip install -e .
```

### 2. Training

```bash
# Step 1: Train VAE (image compression)
uv run main.py --train-vae

# Step 2: Train Diffusion (text-to-image)
uv run main.py --train-diffusion

# Step 3: Generate images!
uv run main.py --generate --prompt "a cute cat"
```

### 3. Web Demo

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

## HuggingFace Hub

Share your trained models and use pre-trained checkpoints from HuggingFace Hub.

### Upload Checkpoints

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Upload VAE checkpoint
python scripts/upload_to_hub.py --model-type vae --repo-id username/tiny-sd-vae

# Upload Diffusion checkpoint
python scripts/upload_to_hub.py --model-type diffusion --repo-id username/tiny-sd-diffusion

# Upload both models to a single repository
python scripts/upload_to_hub.py --model-type all --repo-id username/tiny-sd-models

# Create a private repository
python scripts/upload_to_hub.py --model-type all --repo-id username/tiny-sd-private --private
```

### Download Checkpoints

```bash
# Download VAE checkpoint
python scripts/download_from_hub.py --repo-id username/tiny-sd-vae --model-type vae

# Download Diffusion checkpoint
python scripts/download_from_hub.py --repo-id username/tiny-sd-diffusion --model-type diffusion

# Download both from a combined repository
python scripts/download_from_hub.py --repo-id username/tiny-sd-models --model-type all

# Download to custom directory
python scripts/download_from_hub.py --repo-id username/tiny-sd-models --model-type all --output-dir ./models
```

### Use Downloaded Models

```bash
# Generate with downloaded checkpoints
uv run main.py --generate \
    --prompt "a cute cat" \
    --vae-checkpoint checkpoints/vae.pt \
    --checkpoint checkpoints/diffusion.pt
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
