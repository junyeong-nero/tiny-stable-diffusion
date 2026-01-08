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

Training uses the **junyeong-neo/emoji-32** dataset from Hugging Face:

- **Source**: [junyeong-nero/emoji-32](https://huggingface.co/datasets/junyeong-nero/emoji-32)
- **Size**: ~10,000 emoji images
- **Resolution**: 32×32 RGB (pre-processed, no resize needed)
- **Captions**: Emoji short names (e.g., "rocket", "cat", "apple")
- **Format**: PIL Images from Hugging Face `datasets` library

### Dataset Structure

```python
{
    "image_apple": PIL.Image (32×32, RGBA),
    "short_name": str,  # e.g., "rocket"
    "category": str,    # e.g., "Activities"
    "subcategory": str, # e.g., "event"
}
```

## ⚙️ Configuration

PixMoji-Diffusion uses a centralized configuration system in `src/config.py`:

### Model Config (Default)
```python
ModelConfig(
    model_size="S",      # S, B, L, XL
    patch_size=2,        # Patch size for tokenization
    image_size=32,       # Input resolution
    in_channels=3,       # RGB
    clip_embed_dim=512,  # CLIP embedding dimension
)
```

### Diffusion Config (Default)
```python
DiffusionConfig(
    num_timesteps=1000,  # Total diffusion steps
    beta_schedule="cosine",  # linear, cosine, quadratic
    guidance_scale=7.5,  # CFG scale
    cfg_probability=0.1, # Probability of unconditional dropout
)
```

### Training Config (Default)
```python
TrainingConfig(
    epochs=100,
    batch_size=64,
    learning_rate=1e-4,
    use_mixed_precision=True,
    ema_decay=0.9999,
    checkpoint_interval=10,
)
```

All configurations can be overridden via command-line arguments.

## 🚀 Installation & Usage

### 1. Clone Repository

```bash
git clone https://github.com/your-username/PixMoji-Diffusion.git
cd PixMoji-Diffusion
```

### 2. Install Dependencies with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager (10-100x faster than pip).

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Or using `uv run` (no activation needed):

```bash
uv run python src/training/train.py --epochs 100 --batch_size 64
```

### 3. Training

Train your own model from scratch:

```bash
uv run python src/training/train.py \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --model-size S
```

Or using the convenience script:

```bash
./scripts/train.sh --epochs 100 --batch-size 64 --model-size S
```

**Training Arguments**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 64)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--model-size`: DiT model size: S, B, L (default: S)
- `--dataset-name`: Hugging Face dataset name (default: junyeong-nero/emoji-32)
- `--output-dir`: Checkpoint output directory (default: checkpoints)
- `--wandb`: Enable Weights & Biases logging

### 4. Inference (Generation)

Generate emojis using a trained model:

```bash
uv run python src/inference/generate.py \
    --prompt "a cute robot" \
    --num-samples 4 \
    --checkpoint checkpoints/model_final.pt \
    --guidance-scale 7.5 \
    --steps 50
```

**Generation Arguments**:
- `--prompt`: Text description of the image (English recommended)
- `--num-samples`: Number of images to generate (default: 4)
- `--checkpoint`: Path to model checkpoint
- `--guidance-scale`: CFG scale (default: 7.5)
- `--steps`: Sampling steps, DDIM (default: 50), DDPM (default: 1000)
- `--ddim`: Use DDIM sampling (default: True)
- `--seed`: Random seed for reproducibility (default: 42)

## 🖼️ Results (Demo)

> **Note**: Generated images are native 32×32 pixels. For display, they're upscaled to 256×256 using nearest-neighbor interpolation.

| Prompt | Generated Result |
| :--- | :---: |
| **"Astronaut in space"** | <img src="assets/sample_astronaut.png" width="100"> |
| **"Red apple"** | <img src="assets/sample_apple.png" width="100"> |
| **"Ghost with hat"** | <img src="assets/sample_ghost.png" width="100"> |

*(Add your generated images to the `assets/` folder and update paths above)*

## 🛠️ Future Works

- **Web UI**: Interactive demo page using Streamlit
- **Background Removal**: Automatic transparent background processing with `rembg`
- **High-Res Upgrade**: Super-Resolution model for 64×64 or 128×128 outputs
- **Latent DiT**: VAE-based latent diffusion for faster training

## 🤝 References

- **DiT**: "Scalable Diffusion Models with Transformers" - Google Research (2023)
  - Paper: [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (2020)
- **CFG**: Ho et al., "Classifier-Free Diffusion Guidance" (2021)
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)

---

### 📬 Contact

- **Name**: [Your Name/Nickname]
- **Email**: [your.email@example.com]
- **GitHub**: [https://github.com/your-username](https://github.com/your-username)
