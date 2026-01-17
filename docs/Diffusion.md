# Diffusion Model Documentation

> Comprehensive guide to Diffusion training, architecture, and configuration in tiny-stable-diffusion.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Rectified Flow](#rectified-flow)
4. [Training Configuration](#training-configuration)
5. [Dataset](#dataset)
6. [Training Pipeline](#training-pipeline)
7. [Classifier-Free Guidance (CFG)](#classifier-free-guidance-cfg)
8. [Sampling / Generation](#sampling--generation)
9. [Evaluation](#evaluation)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Overview

The Diffusion model is the **second stage** of the Stable Diffusion pipeline. It learns to generate images from text prompts by:

1. **Training**: Learning to predict noise/velocity from noisy latents + text
2. **Generation**: Iteratively denoising random noise → clean latent → image

### Key Specifications

| Component | Value | Description |
|-----------|-------|-------------|
| Architecture | MMDiT (or DiT) | Multi-Modal Diffusion Transformer |
| Input | 8×8×16 latent | From frozen VAE encoder |
| Output | 8×8×16 latent | Decoded by VAE to 64×64 image |
| Text Encoder | CLIP ViT-B/32 | 512-dim text embeddings |
| Diffusion Type | Rectified Flow | SD3-style linear interpolation |
| Timesteps | 1000 | Continuous sampling |

### Pipeline Flow

```
Training:
Image (64×64) → VAE Encoder → Latent (8×8×16) → Add Noise → MMDiT → Predict Velocity

Generation:
Random Noise → MMDiT Denoise (50 steps) → Clean Latent → VAE Decoder → Image (64×64)
```

---

## Architecture

### DiT vs MMDiT

tiny-stable-diffusion supports two architectures:

| | DiT (Vanilla) | MMDiT (SD3-style) |
|---|---|---|
| Text Conditioning | Cross-Attention | Joint Attention |
| Architecture | Separate text/image paths | Unified attention |
| Training Stability | Good | Better (QK-RMSNorm) |
| SD3 Compatible | No | Yes |

**Recommendation**: Use MMDiT (default) for best results.

### Model Sizes

#### MMDiT (Recommended)

| Size | Layers | Hidden | Heads | Parameters | Use Case |
|------|--------|--------|-------|------------|----------|
| **S** | 12 | 384 | 6 | **87M** | Quick experiments |
| **B** | 12 | 768 | 12 | **187M** | Default, balanced |
| L | 24 | 1024 | 16 | **559M** | High quality |
| XL | 28 | 1152 | 16 | **780M** | Best quality |

#### DiT (Vanilla)

| Size | Layers | Hidden | Heads | Parameters |
|------|--------|--------|-------|------------|
| S | 12 | 384 | 6 | 40M |
| B | 12 | 768 | 12 | 159M |
| L | 24 | 1024 | 16 | 559M |
| XL | 28 | 1152 | 16 | 824M |

### MMDiT Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MMDiT Block                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐                          │
│   │ Text Tokens  │    │ Image Tokens │                          │
│   │   (B, 1, D)  │    │  (B, N, D)   │                          │
│   └──────┬───────┘    └──────┬───────┘                          │
│          │                   │                                   │
│          └─────────┬─────────┘                                   │
│                    ▼                                             │
│          ┌─────────────────┐                                     │
│          │ Joint Attention │ ← Text and Image attend together   │
│          │   (QK-RMSNorm)  │                                     │
│          └────────┬────────┘                                     │
│                   │                                              │
│          ┌────────┴────────┐                                     │
│          ▼                 ▼                                     │
│   ┌────────────┐    ┌────────────┐                              │
│   │  Text MLP  │    │ Image MLP  │ ← Separate MLPs              │
│   └────────────┘    └────────────┘                              │
│          │                 │                                     │
│          └────────┬────────┘                                     │
│                   ▼                                              │
│          ┌─────────────────┐                                     │
│          │ Time Modulation │ ← AdaLN from timestep embedding    │
│          └─────────────────┘                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              × L layers (12 for B size)
                                        ↓
                            ┌─────────────────┐
                            │   Final Layer   │
                            │  Unpatchify     │
                            └─────────────────┘
                                        ↓
                            Output (B, 16, 8, 8)
```

### Key Components

#### 1. Patch Embedding

Converts latent image to sequence of tokens:

```python
# Input: (B, 16, 8, 8) latent
# Patch size: 2×2

PatchEmbed:
    Conv2d(16, hidden, kernel=2, stride=2)
    # (B, 16, 8, 8) → (B, hidden, 4, 4) → (B, 16, hidden)

# Output: 16 tokens of hidden dimension
```

#### 2. Position Embedding

Learnable positional encoding for image tokens:

```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
```

#### 3. Timestep Embedding

Sinusoidal embedding → MLP:

```python
def get_timestep_embedding(timesteps):
    # Sinusoidal encoding
    half = hidden_size // 2
    freqs = exp(-log(10000) * arange(half) / half)
    emb = cat([cos(t * freqs), sin(t * freqs)])

    # MLP
    emb = Linear(emb)
    emb = SiLU(emb)
    emb = Linear(emb)
    return emb  # (B, hidden)
```

#### 4. Text Conditioning

CLIP text embeddings projected to match hidden dimension:

```python
# CLIP: (B, 512) → Text projection: (B, hidden)
text_tokens = text_embed.unsqueeze(1)  # (B, 1, hidden)
```

---

## Rectified Flow

### Overview

tiny-stable-diffusion uses **Rectified Flow** (SD3-style) instead of DDPM:

| | DDPM | Rectified Flow |
|---|---|---|
| Forward Process | Gaussian noise schedule | Linear interpolation |
| Target | Noise (ε) | Velocity (v) |
| Sampling | DDPM/DDIM | Euler ODE |
| Efficiency | 50-1000 steps | 20-50 steps |

### Forward Process (Training)

Linear interpolation between clean and noise:

```python
x_t = (1 - t) * x_0 + t * noise

# Where:
# x_0: clean latent
# noise: random Gaussian noise
# t: timestep in [0, 1]
```

### Velocity Prediction

Model predicts the velocity (direction from clean to noise):

```python
v = noise - x_0  # Target velocity

# Model predicts: v_pred = model(x_t, t, text)
# Loss: MSE(v_pred, v)
```

### Reverse Process (Sampling)

Euler ODE solver:

```python
def euler_step(x_t, t_curr, t_next):
    v_pred = model(x_t, t_curr, text)
    dt = t_next - t_curr
    x_next = x_t + v_pred * dt
    return x_next
```

### Logit-Normal Timestep Sampling

SD3-style sampling concentrates on middle timesteps:

```python
def sample_timesteps_logit_normal(batch_size, mean=0.0, std=1.0):
    # Sample from normal distribution
    u = torch.randn(batch_size) * std + mean

    # Apply sigmoid to get [0, 1] with concentration in middle
    t = torch.sigmoid(u)

    return (t * num_timesteps).long()
```

**Why?** Middle timesteps are most informative for learning.

---

## Training Configuration

### Full Configuration (config.yaml)

```yaml
diffusion_train:
    # ===== Model Type =====
    model_type: mmdit               # "dit" or "mmdit"

    # ===== Dataset =====
    data_source: huggingface        # "huggingface" or "streaming_caption"
    dataset_name: visual-layer/oxford-iiit-pet-vl-enriched
    image_field: image
    caption_field: caption_enriched
    split: train

    # ===== Image/Latent Settings =====
    image_size: 64                  # Original image size
    latent_size: 8                  # 64 / 8 = 8
    in_channels: 16                 # Matches VAE latent channels

    # ===== Training Hyperparameters =====
    epochs: 200
    batch_size: 64                  # Adjust based on GPU memory
    learning_rate: 3.0e-4

    # ===== CFG Settings =====
    initial_cfg_prob: 0.0           # Start with no dropout
    final_cfg_prob: 0.1             # 10% dropout after warmup
    cfg_warmup_epochs: 10           # Epochs to reach final_cfg_prob

    # ===== VAE =====
    vae_checkpoint: checkpoints/vae.pt
    scaling_factor: auto            # Auto-compute from data

    # ===== Checkpointing =====
    checkpoint_path: checkpoints/diffusion.pt
    checkpoint_dir: checkpoints
    checkpoint_interval: 10
    resume: false

    # ===== Model Configuration =====
    model_size: B                   # S, B, L, XL
    patch_size: 2

    # ===== Diffusion Settings =====
    num_timesteps: 1000
    guidance_scale: 7.5

    # ===== Timestep Sampling =====
    use_logit_normal_sampling: true
    logit_mean: 0.0
    logit_std: 1.0

    # ===== EMA =====
    use_ema: true
    ema_decay: 0.9999

    # ===== Training Settings =====
    mixed_precision: false
    device: auto
    seed: 42

    # ===== DataLoader =====
    num_workers: 4
    prefetch_factor: 2

    # ===== Validation =====
    validation_prompts:
        - a photo of a fluffy Persian cat
        - a golden retriever sitting on grass
        - a black and white Border Collie
        - a cute tabby kitten playing
        - a Siamese cat with blue eyes
    validation_interval: 10
    sample_dir: samples

    # ===== MMDiT Specific =====
    qk_rmsnorm: true                # QK normalization for stability
    register_tokens: 0              # Additional register tokens
```

### Key Configuration Decisions

#### 1. Learning Rate + Batch Size

| Dataset Size | Batch Size | Learning Rate |
|--------------|------------|---------------|
| < 10k | 32-64 | 3e-4 |
| 10k-100k | 128-256 | 1e-4 |
| > 100k | 256+ | 1e-4 |

#### 2. Model Size Selection

| Model Size | VRAM Usage | Training Time | Quality |
|------------|------------|---------------|---------|
| S (87M) | ~4GB | Fast | Good |
| **B (187M)** | ~8GB | Moderate | **Recommended** |
| L (559M) | ~16GB | Slow | High |
| XL (780M) | ~24GB | Very Slow | Best |

#### 3. Scaling Factor

```yaml
scaling_factor: auto  # Recommended
```

Normalizes latent space to std ≈ 1 for stable training.

---

## Dataset

### Recommended Datasets

| Dataset | Size | Description | Use Case |
|---------|------|-------------|----------|
| **oxford-iiit-pet-vl-enriched** | 7.4K | Cats & dogs with captions | Default |
| LAION-300k | 300K | Diverse images | Large-scale |
| pokemon-blip-captions | 833 | Pokemon images | Quick testing |

### Dataset Requirements

- **Image field**: Contains PIL Image or image path
- **Caption field**: Contains text description

### Current Dataset: Oxford Pets

```yaml
data_source: huggingface
dataset_name: visual-layer/oxford-iiit-pet-vl-enriched
image_field: image
caption_field: caption_enriched
```

**Statistics:**
- ~7,400 images
- 37 breeds (12 cats, 25 dogs)
- Enriched captions with detailed descriptions

---

## Training Pipeline

### Command

```bash
# Basic training (uses config.yaml)
uv run main.py --train-diffusion

# With custom parameters
uv run main.py --train-diffusion \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 3e-4

# Resume training
uv run main.py --train-diffusion --resume

# With wandb logging
uv run main.py --train-diffusion --wandb
```

### Training Loop

```python
# Load frozen VAE
vae = load_vae(checkpoint)
vae.eval()
vae.freeze()

# Load CLIP text encoder
clip = CLIPTextEncoder()
uncond_embed = clip.encode("")  # For CFG

# Initialize diffusion model
model = MMDiT(model_size="B", ...)
diffusion = Diffusion(num_timesteps=1000, ...)

for epoch in range(epochs):
    for batch in dataloader:
        images = batch["image"]      # (B, 3, 64, 64)
        captions = batch["caption"]  # List of strings

        # Encode to latent space
        with torch.no_grad():
            latents = vae.encode_to_latent(images)
            text_embeds = clip.encode(captions)

        # Sample timesteps (logit-normal)
        timesteps = diffusion.sample_timesteps_logit_normal(B)

        # Compute loss
        loss = diffusion.training_loss(model, latents, timesteps, text_embeds)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # EMA update
        if ema:
            ema.update()

    # Generate validation samples
    if (epoch + 1) % validation_interval == 0:
        generate_samples(model, prompts, ...)
```

### Output Files

```
checkpoints/
├── diffusion.pt                  # Latest (full state)
├── diffusion_epoch_10.pt         # Epoch 10 (weights only)
├── diffusion_epoch_20.pt         # Epoch 20 (weights only)
└── ...

samples/
├── epoch_10/
│   ├── 00_a_photo_of_a_fluff.png
│   ├── 01_a_golden_retrieve.png
│   └── ...
├── epoch_20/
└── ...
```

---

## Classifier-Free Guidance (CFG)

### Overview

CFG improves text-image alignment by combining conditional and unconditional predictions:

### Training

```python
# During training: randomly drop text condition with probability cfg_prob
if random() < cfg_prob:
    text_embed = uncond_embed  # Empty string embedding ""
```

### CFG Warmup

```yaml
initial_cfg_prob: 0.0    # Start with no dropout
final_cfg_prob: 0.1      # 10% after warmup
cfg_warmup_epochs: 10    # Linear increase over 10 epochs
```

### Inference

```python
def cfg_sample(model, x_t, t, text_embed, guidance_scale=7.5):
    # Conditional prediction
    cond_pred = model(x_t, t, text_embed)

    # Unconditional prediction
    uncond_pred = model(x_t, t, uncond_embed)

    # CFG combination
    pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

    return pred
```

### Guidance Scale

| Scale | Effect |
|-------|--------|
| 1.0 | No guidance (ignores CFG) |
| 3.0-5.0 | Soft guidance |
| **7.5** | **Default (recommended)** |
| 10.0-15.0 | Strong guidance (may oversaturate) |

---

## Sampling / Generation

### Command

```bash
# Basic generation
uv run main.py --generate --prompt "a cute cat"

# With options
uv run main.py --generate \
    --prompt "a golden retriever on the beach" \
    --steps 50 \
    --guidance 7.5 \
    --seed 42 \
    --output my_image.png
```

### Euler ODE Sampling

```python
def sample(model, text_embeds, num_steps=50):
    # Start from pure noise
    x = randn(B, 16, 8, 8)

    # Timestep schedule: 1.0 → 0.0
    timesteps = linspace(1.0, 0.0, num_steps + 1)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]

        # Euler step
        v_pred = model(x, t_curr, text_embeds)
        dt = t_next - t_curr
        x = x + v_pred * dt

    # Decode latent to image
    image = vae.decode_from_latent(x)
    return image
```

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_steps` | 50 | Denoising steps (more = better quality, slower) |
| `guidance_scale` | 7.5 | CFG strength |
| `seed` | None | Random seed for reproducibility |

---

## Evaluation

### Qualitative Evaluation

Monitor generated samples during training:

```
samples/epoch_N/
├── 00_prompt1.png
├── 01_prompt2.png
└── ...
```

**What to look for:**
1. **Early epochs (1-20)**: Blurry, color blobs
2. **Mid epochs (20-100)**: Shapes forming, unclear details
3. **Late epochs (100+)**: Clear objects, recognizable features

### Quantitative Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| FID | Fréchet Inception Distance | Lower is better |
| CLIP Score | Text-image alignment | Higher is better |

### Training Metrics

| Metric | Healthy Range |
|--------|---------------|
| Loss | 0.5 - 1.0 (after warmup) |
| Learning Rate | Should follow cosine schedule |

---

## Troubleshooting

### Common Issues

#### 1. Loss Not Decreasing

**Symptoms:** Loss stays flat or increases
**Causes:**
- Learning rate too low
- Batch size too large for dataset
- VAE checkpoint issue

**Solutions:**
```yaml
# Increase learning rate
learning_rate: 3.0e-4

# Decrease batch size
batch_size: 64  # or 32

# Verify VAE checkpoint
uv run main.py --reconstruct-vae --input sample.png
```

#### 2. Generated Images Look Like Noise

**Symptoms:** No recognizable shapes
**Causes:**
- Too few epochs
- Learning rate too low
- CFG not working

**Solutions:**
- Train for more epochs (100+)
- Check that CFG dropout is working
- Increase guidance_scale during inference

#### 3. Generated Images Are Blurry

**Symptoms:** Recognizable shapes but no detail
**Causes:**
- Not enough training
- VAE reconstruction quality poor
- Guidance scale too low

**Solutions:**
- Train longer
- Check VAE quality first
- Increase `guidance_scale` to 10-12

#### 4. Out of Memory

**Solutions:**
```yaml
# Reduce batch size
batch_size: 32

# Use smaller model
model_size: S

# Enable mixed precision
mixed_precision: true

# Reduce workers
num_workers: 2
```

#### 5. Text Not Affecting Output

**Symptoms:** Same output regardless of prompt
**Causes:**
- CFG probability too high
- CLIP encoder issue
- Guidance scale too low

**Solutions:**
```yaml
# Check CFG probability
final_cfg_prob: 0.1  # Not higher

# Increase guidance scale
guidance_scale: 10.0
```

---

## References

### Papers

- [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) - Rectified Flow Transformers
- [DiT](https://arxiv.org/abs/2212.09748) - Scalable Diffusion Models with Transformers
- [DDPM](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [DDIM](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) - CFG paper
- [CLIP](https://arxiv.org/abs/2103.00020) - Text encoder

### Code References

- [src/models/mmdit.py](../src/models/mmdit.py) - MMDiT implementation
- [src/models/diffusion.py](../src/models/diffusion.py) - Rectified Flow
- [src/training/trainer.py](../src/training/trainer.py) - Training loop
- [src/text_encoder/clip_encoder.py](../src/text_encoder/clip_encoder.py) - CLIP encoder
