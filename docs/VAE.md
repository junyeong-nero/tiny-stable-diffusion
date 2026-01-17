# VAE (Variational AutoEncoder) Documentation

> Comprehensive guide to VAE architecture, training, and configuration in tiny-stable-diffusion.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Training Configuration](#training-configuration)
4. [Dataset](#dataset)
5. [Training Pipeline](#training-pipeline)
6. [Loss Function](#loss-function)
7. [Evaluation](#evaluation)
8. [Inference](#inference)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Overview

The VAE (Variational AutoEncoder) is the **first stage** of the Stable Diffusion pipeline. Its purpose is to:

1. **Compress** images to a compact latent representation
2. **Decompress** latents back to images
3. Enable efficient diffusion training in the latent space

### Key Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| Architecture | AutoencoderKL | SD3-style VAE |
| Input Resolution | 64×64 RGB | 3 channels, normalized to [-1, 1] |
| Latent Resolution | 8×8×16 | 16 channels, f8 compression |
| Compression Ratio | 12:1 | 12,288 → 1,024 dimensions |
| Parameters | ~21M | Lightweight for consumer GPUs |

### Comparison with SD3 VAE

| | Stable Diffusion 3 | tiny-stable-diffusion |
|---|---|---|
| Input | 1024×1024 | 64×64 |
| Latent | 128×128×16 | 8×8×16 |
| Compression | f8 | f8 |
| Parameters | ~84M | ~21M |

---

## Architecture

### High-Level Structure

```
Input Image (64×64×3)
        ↓
   ┌─────────┐
   │ Encoder │ ← Conv3x3 + ResBlocks + Downsample×3 + Attention
   └────┬────┘
        ↓
   ┌─────────┐
   │ Latent  │ ← (mean, logvar) → Reparameterization → z
   │ (8×8×16)│
   └────┬────┘
        ↓
   ┌─────────┐
   │ Decoder │ ← Conv3x3 + ResBlocks + Upsample×3 + Attention
   └────┬────┘
        ↓
Output Image (64×64×3)
```

### Encoder Architecture

```python
# Input: (B, 3, 64, 64)

# Initial convolution
Conv3x3(3 → 64)

# Resolution 64×64 → 32×32
ResBlock(64 → 64) × 2
Downsample → Conv3x3 stride=2

# Resolution 32×32 → 16×16
ResBlock(64 → 128) × 2
Downsample → Conv3x3 stride=2

# Resolution 16×16 → 8×8
ResBlock(128 → 256) × 2
Downsample → Conv3x3 stride=2

# Bottleneck (8×8)
ResBlock(256 → 256) × 2
AttnBlock(256)              # Self-attention for global context
ResBlock(256 → 256)

# Output to latent
GroupNorm → SiLU → Conv3x3(256 → 32)  # 32 = 2 × z_channels (mean + logvar)

# Output: (B, 32, 8, 8) → split into mean (B, 16, 8, 8) and logvar (B, 16, 8, 8)
```

### Decoder Architecture

```python
# Input: z (B, 16, 8, 8)

# Initial convolution
Conv3x3(16 → 256)

# Bottleneck (8×8)
ResBlock(256 → 256)
AttnBlock(256)              # Self-attention
ResBlock(256 → 256) × 2

# Resolution 8×8 → 16×16
ResBlock(256 → 256) × 3
Upsample → nearest + Conv3x3

# Resolution 16×16 → 32×32
ResBlock(256 → 128) × 3
Upsample → nearest + Conv3x3

# Resolution 32×32 → 64×64
ResBlock(128 → 64) × 3
Upsample → nearest + Conv3x3

# Output
GroupNorm → SiLU → Conv3x3(64 → 3)

# Output: (B, 3, 64, 64)
```

### Key Components

#### ResBlock (Residual Block)

```python
class ResnetBlock:
    """ResNet Block: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + Skip"""

    def forward(x):
        h = GroupNorm(x)
        h = SiLU(h)
        h = Conv3x3(h)
        h = GroupNorm(h)
        h = SiLU(h)
        h = Conv3x3(h)
        return x + h  # Skip connection
```

#### AttnBlock (Self-Attention)

```python
class AttnBlock:
    """Self-Attention for global context (used in bottleneck only)"""

    def forward(x):
        # x: (B, C, H, W)
        q = Conv1x1(GroupNorm(x))  # Query
        k = Conv1x1(x)              # Key
        v = Conv1x1(x)              # Value

        # Reshape: (B, C, H*W)
        attn = softmax(q @ k.T / sqrt(C))
        out = attn @ v

        return x + Conv1x1(out)
```

#### Reparameterization Trick

```python
def reparameterize(mean, logvar):
    """Sample z = mean + std * epsilon where epsilon ~ N(0, 1)"""
    std = exp(0.5 * logvar)
    eps = randn_like(std)
    return mean + std * eps
```

---

## Training Configuration

### Full Configuration (config.yaml)

```yaml
vae_train:
    # ===== Dataset =====
    data_source: streaming_caption    # "streaming_caption" for URL-based datasets
    dataset_name: hmu013/LAION-300k   # HuggingFace dataset name
    image_field: png                   # Field containing image data
    caption_field: json                # Field containing caption (unused for VAE)
    split: train

    # ===== Streaming Settings =====
    buffer_size: 5000                  # Shuffle buffer size
    skip_failures: true                # Skip failed downloads
    url_timeout: 10                    # Timeout in seconds
    max_retries: 3                     # Max download retries

    # ===== DataLoader =====
    num_workers: 8                     # Parallel data loading
    prefetch_factor: 2                 # Batches to prefetch per worker

    # ===== Image Settings =====
    image_size: 64                     # Input image size (64×64)

    # ===== Architecture =====
    latent_channels: 16                # SD3-style (16 channels)
    vae_ch: 64                         # Base channel count
    vae_ch_mult: [1, 2, 4, 4]          # Channel multipliers (f8 compression)

    # ===== Training Hyperparameters =====
    epochs: 100
    batch_size: 256                    # Large batch for streaming dataset
    learning_rate: 4.0e-4              # AdamW optimizer

    # ===== KL Annealing =====
    kl_weight: 0.000001                # Very low (reconstruction-focused)
    kl_annealing: cyclical             # "none", "linear", "cyclical"
    kl_n_cycles: 4                     # Number of cyclical annealing cycles
    kl_cycle_ratio: 0.5                # Proportion spent increasing

    # ===== Training Settings =====
    mixed_precision: false             # Enable for faster training on GPU
    device: auto                       # "auto", "cuda", "mps", "cpu"
    seed: 42

    # ===== Checkpointing =====
    checkpoint_path: checkpoints/vae.pt
    checkpoint_interval: 10            # Save every N epochs
    resume: false                      # Resume from checkpoint
```

### Key Configuration Decisions

#### 1. KL Weight (`kl_weight: 0.000001`)

| KL Weight | Behavior | Use Case |
|-----------|----------|----------|
| 1.0 | Strong regularization | Standard VAE |
| 0.01 - 0.1 | Balanced | β-VAE |
| **0.000001** | **Nearly zero** | **SD-style VAE** |

**Why so low?**
- Stable Diffusion VAE prioritizes **reconstruction quality** over latent structure
- Diffusion model learns the latent distribution, not VAE
- High KL weight causes blurry reconstructions

#### 2. Cyclical Annealing (`kl_annealing: cyclical`)

Prevents "KL vanishing" by periodically resetting KL weight:

```
KL Weight
    ^
β   |    ╱╲    ╱╲    ╱╲    ╱╲
    |   ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
    |  ╱    ╲╱    ╲╱    ╲╱    ╲
0   +-----------------------------------------> Epochs
       Cycle 1  Cycle 2  Cycle 3  Cycle 4
```

#### 3. Learning Rate Schedule

Cosine annealing with warmup:

```python
# Warmup: 5% of total steps
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = base_lr * 0.5 * (1 + cos(π * progress))
```

---

## Dataset

### Recommended Datasets

| Dataset | Size | Type | Best For |
|---------|------|------|----------|
| **hmu013/LAION-300k** | 300K | Streaming | VAE training (default) |
| reach-vb/pokemon-blip-captions | 833 | Direct | Quick testing |
| lambdalabs/pokemon-blip-captions | 833 | Direct | Quick testing |

### Using LAION-300k

```yaml
vae_train:
    data_source: streaming_caption
    dataset_name: hmu013/LAION-300k
    image_field: png        # Image URLs in 'png' field
    caption_field: json     # Caption in json: {"caption": "..."}
```

The streaming dataset downloads images on-the-fly from URLs, which:
- ✅ Doesn't require local storage
- ✅ Handles large datasets efficiently
- ⚠️ Requires internet connection
- ⚠️ Some images may fail to download

### Data Preprocessing

```python
# Applied transforms:
1. Resize to (image_size, image_size) with bicubic interpolation
2. Center crop if aspect ratio differs
3. Convert to tensor
4. Normalize to [-1, 1]: (x * 2) - 1
```

---

## Training Pipeline

### Command

```bash
# Basic training
uv run main.py --train-vae

# With custom parameters
uv run main.py --train-vae \
    --epochs 100 \
    --batch-size 256 \
    --learning-rate 4e-4

# Resume training
uv run main.py --train-vae --resume
```

### Training Loop

```python
for epoch in range(epochs):
    for batch in dataloader:
        images = batch["image"]  # (B, 3, 64, 64)

        # Forward pass
        mean, logvar = encoder(images)
        z = reparameterize(mean, logvar)
        reconstructed = decoder(z)

        # Compute loss
        recon_loss = MSE(reconstructed, images)
        kl_loss = KL_divergence(mean, logvar)
        total_loss = recon_loss + kl_weight * kl_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # Save checkpoint every N epochs
    if (epoch + 1) % checkpoint_interval == 0:
        save_checkpoint(model, "vae_epoch_{epoch+1}.pt")
```

### Output Files

```
checkpoints/
├── vae.pt                    # Latest checkpoint (full state)
├── vae_epoch_10.pt           # Epoch 10 weights only
├── vae_epoch_20.pt           # Epoch 20 weights only
└── ...

samples/
└── vae_epoch_N/
    ├── 00_original.png
    ├── 00_reconstruction.png
    ├── 01_original.png
    └── 01_reconstruction.png
```

---

## Loss Function

### Total Loss

```
L_total = L_recon + β × L_KL
```

Where:
- `L_recon`: Reconstruction loss (MSE)
- `L_KL`: KL divergence loss
- `β`: KL weight (typically 1e-6 for SD-style VAE)

### Reconstruction Loss

```python
L_recon = MSE(x, x_hat)
        = mean((x - x_hat)^2)
```

**Interpretation:**
- Lower is better
- Measures pixel-wise reconstruction quality
- Typical range: 0.001 - 0.05

### KL Divergence Loss

```python
L_KL = -0.5 * mean(1 + logvar - mean^2 - exp(logvar))
```

**Interpretation:**
- Regularizes latent space towards N(0, 1)
- Prevents posterior collapse
- With β=1e-6, has minimal effect on reconstruction

### Latent Space Statistics

During training, monitor:

| Metric | Target | Description |
|--------|--------|-------------|
| `μ` (latent mean) | ≈ 0 | Should be close to 0 |
| `σ` (latent std) | ≈ 1 | Should be close to 1 |

If μ drifts far from 0 or σ from 1, increase KL weight slightly.

---

## Evaluation

### Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **PSNR** | 10 × log₁₀(MAX²/MSE) | 20-50 dB | Higher = better |
| **SSIM** | Structural similarity | 0-1 | Higher = better |
| **MSE** | Mean squared error | 0-1 | Lower = better |

### Quality Reference

| PSNR (dB) | Quality |
|-----------|---------|
| > 40 | Excellent (near lossless) |
| 30-40 | Good |
| 20-30 | Fair |
| < 20 | Poor |

### Evaluation Commands

```bash
# Evaluate on local images
./scripts/evaluate-vae.sh --input-dir samples/original --checkpoint checkpoints/vae.pt

# Evaluate on HuggingFace dataset
./scripts/evaluate-vae.sh --dataset reach-vb/pokemon-blip-captions --max-samples 100

# Save results to JSON
./scripts/evaluate-vae.sh --input-dir samples/original --save results/eval.json
```

### Expected Results

| Epoch | PSNR (dB) | SSIM | MSE |
|-------|-----------|------|-----|
| 10 | ~32.5 | ~0.94 | ~0.0007 |
| 20 | ~33.0 | ~0.96 | ~0.0006 |
| 30 | ~37.0 | ~0.97 | ~0.0002 |
| 40 | ~38.0 | ~0.98 | ~0.0002 |

---

## Inference

### VAE Reconstruction

```bash
# Single image
uv run main.py --reconstruct-vae \
    --input image.png \
    --output reconstructed.png \
    --vae-checkpoint checkpoints/vae.pt

# Batch reconstruction
./scripts/inference-vae.sh --all
```

### Python API

```python
from src.models.vae import create_vae
from src.inference.vae_inference import reconstruct_image

# Load model
vae = create_vae()
vae.load_state_dict(torch.load("checkpoints/vae.pt")["model_state_dict"])
vae.eval()

# Reconstruct image
original, reconstructed = reconstruct_image(vae, "image.png")

# Encode to latent
latent = vae.encode_to_latent(image)  # For diffusion training
```

---

## Troubleshooting

### Common Issues

#### 1. Blurry Reconstructions

**Cause:** KL weight too high
**Solution:** Reduce `kl_weight` (try 1e-6 or lower)

#### 2. Posterior Collapse (latent = 0)

**Cause:** KL weight too low or training instability
**Solution:**
- Use cyclical annealing
- Increase `kl_weight` slightly (try 1e-5)
- Check latent statistics (μ ≈ 0, σ ≈ 1)

#### 3. Training Loss Not Decreasing

**Cause:** Learning rate too low
**Solution:** Increase to 4e-4

#### 4. Out of Memory

**Solution:**
- Reduce `batch_size`
- Enable `mixed_precision: true`
- Reduce `num_workers`

#### 5. Dataset Download Failures

**Solution:**
- Increase `url_timeout`
- Increase `max_retries`
- Check internet connection

---

## References

### Papers

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Original VAE paper
- [Cyclical Annealing Schedule](https://arxiv.org/abs/1903.10145) - KL annealing
- [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) - SD3 VAE design

### Code References

- [src/models/vae.py](../src/models/vae.py) - VAE implementation
- [src/training/vae_trainer.py](../src/training/vae_trainer.py) - Training loop

### Evaluation

- [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [SSIM](https://ieeexplore.ieee.org/document/1284395)
