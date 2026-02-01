# Inference & Generation Deep Dive

> This document provides a detailed explanation of the image generation process in tiny-stable-diffusion.

## Table of Contents

1. [Generation Pipeline Overview](#generation-pipeline-overview)
2. [Sampling Algorithms](#sampling-algorithms)
3. [Classifier-Free Guidance](#classifier-free-guidance)
4. [Step-by-Step Generation](#step-by-step-generation)
5. [Advanced Options](#advanced-options)
6. [Performance Optimization](#performance-optimization)

---

## Generation Pipeline Overview

> File location: `src/inference/generator.py`

### Complete Generation Process

**Input**: A text prompt such as "a cute cat sitting on a couch"

**Step 1: Text Encoding (CLIP)**
- The prompt is tokenized and passed through the CLIP Transformer to produce `text_embed` with shape (1, 512)
- An empty string is also encoded to produce `uncond_embed` with shape (1, 512) for classifier-free guidance

**Step 2: Initialize Random Noise**
- Create pure Gaussian noise in latent space: `z_T = randn(1, 16, 8, 8)`
- If a seed is provided, `torch.manual_seed(seed)` is called for reproducible generation

**Step 3: Iterative Denoising (Euler ODE)**

For each timestep t from 1.0 down to 0.0:
1. Compute conditional velocity: `v_cond = MMDiT(z_t, t, text_embed)`
2. Compute unconditional velocity: `v_uncond = MMDiT(z_t, t, uncond_embed)`
3. Apply Classifier-Free Guidance: `v_pred = v_uncond + scale × (v_cond - v_uncond)`
4. Perform Euler step: `z_{next} = z_t + v_pred * dt`

Output: `z_0` (1, 16, 8, 8) - Clean latent

**Step 4: VAE Decoding**
- Pass the clean latent through post_quant_conv and Decoder to produce image (1, 3, 64, 64)
- Normalize from [-1, 1] to [0, 1]: `image = (image + 1) / 2`
- Clamp values: `image = clamp(image, 0, 1)`

**Output**: 64×64 RGB image

---

## Sampling Algorithms

### Rectified Flow vs DDPM

| Feature | DDPM | Rectified Flow (SD3) |
|---------|------|----------------------|
| **Trajectory** | Curved, stochastic | **Straight, deterministic** |
| **Prediction** | Noise ($\epsilon$) | **Velocity ($v$)** |
| **Solver** | Custom (DDPM/DDIM) | **Standard ODE (Euler, RK4)** |
| **Steps** | 50-1000 | **10-50** |
| **Efficiency** | Slower | **Faster** |

### Euler Sampling (Rectified Flow)

```python
def euler_step(model, x_t, t_curr, t_next, text_embeds, use_cfg=True):
    """
    Euler ODE Step:
    x_{next} = x_{curr} + v(x_{curr}, t) * dt

    Process:
    1. Predict velocity: v = model(x_t, t, text)
    2. Calculate dt: dt = t_next - t_curr (negative value)
    3. Update: x_{next} = x_t + v * dt
    """
```

**Characteristics**:
- **First-order ODE solver**: Simple and fast.
- **Deterministic**: Same seed + same prompt = exact same image.
- **Linear interpolation**: Transitions smoothly from noise (t=1) to data (t=0).

---

## Classifier-Free Guidance

### CFG Concept

**Purpose**: Strengthen the influence of text conditions

**Method**:
1. Conditional prediction: `v_cond = model(x_t, t, text_embed)`
2. Unconditional prediction: `v_uncond = model(x_t, t, uncond_embed)`
3. Apply guidance: `v_pred = v_uncond + s × (v_cond - v_uncond)`

**Mathematical interpretation**: `v_pred = (1 - s) × v_uncond + s × v_cond`

**Guidance scale values**:
- s = 1.0: Same as conditional only (no guidance)
- s > 1.0: Stronger movement toward conditional direction
- s = 7.5: Recommended value (SD3 default)

### Guidance Scale Effects

| Scale | Effect |
|-------|--------|
| s = 1.0 | Uses conditional generation only; weak text reflection; high diversity; may have lower quality |
| s = 4.0 | Weak guidance; moderate text reflection; maintains diversity |
| s = 7.5 (recommended) | Good balance; text well reflected; good quality; SD3 default |
| s = 15.0+ | Very strong guidance; excessive text reflection; color saturation; possible artifacts |

---

## Step-by-Step Generation

### Detailed Process

```python
def sample(self, model, shape, text_embeds, num_steps=50, ...):
    """
    Complete sampling process
    """
    B, C, H, W = shape  # (1, 16, 8, 8)
    device = next(model.parameters()).device

    # 1. Set seed (reproducibility)
    if seed is not None:
        torch.manual_seed(seed)

    # 2. Start from pure noise
    x_t = torch.randn(shape, device=device)  # z_T

    # 3. Calculate timesteps (Linear)
    # [1.0, 0.98, ..., 0.02, 0.0] * 1000
    timesteps = torch.linspace(1000, 0, num_steps + 1)

    # 4. Iterative denoising
    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i+1]

        # Euler step
        x_t = euler_step(model, x_t, t_curr, t_next, text_embeds)

    # 5. VAE decoding
    if vae_decoder:
        x_t = vae_decoder.decode_from_latent(x_t)

    # 6. Normalize to [0, 1]
    x_t = (x_t + 1.0) / 2.0
    x_t = torch.clamp(x_t, 0.0, 1.0)

    return x_t
```

---

## Advanced Options

### CLI Options

```bash
uv run main.py --generate \
    --prompt "a cute cat sitting on a couch" \
    --checkpoint checkpoints/diffusion.pt \
    --vae-checkpoint checkpoints/vae.pt \
    --steps 50 \           # Number of sampling steps
    --guidance 7.5 \       # CFG scale
    --seed 42 \            # Reproducibility seed
    --num-samples 4 \      # Number of images to generate
    --output output.png    # Output file
```

### Parameter Descriptions

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--steps` | 50 | 10-100 | Number of sampling steps. Higher = better quality, slower |
| `--guidance` | 7.5 | 1.0-20.0 | CFG scale. Higher = stronger prompt reflection |
| `--seed` | None | int | Reproducibility seed. Same seed = same result |
| `--num-samples` | 1 | 1-16 | Number of images to generate |

### Steps vs Quality

| Steps | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| 10 | Very fast | Low (noisy residue) | Quick prototyping |
| 20 | Fast | Acceptable | Fast preview |
| 50 (recommended) | Moderate | Good | Default setting |
| 100 | Slow | Very good | High-quality generation |

---

## Performance Optimization

### Memory Optimization

```python
# 1. Disable gradients during inference
with torch.no_grad():
    image = generate(prompt)

# 2. Half precision (FP16/BF16)
model = model.bfloat16()
vae = vae.bfloat16()

# 3. VAE slicing (for large batches)
def decode_with_slicing(vae, latents, slice_size=4):
    images = []
    for i in range(0, len(latents), slice_size):
        batch = latents[i:i+slice_size]
        images.append(vae.decode(batch))
    return torch.cat(images)
```

---

## Interactive Demo

```bash
# Run interactive demo
uv run main.py --demo
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Noisy images | Not enough steps | Use `--steps 50` or higher |
| Prompt not reflected | Guidance too low | Use `--guidance 7.5` |
| Oversaturated colors | Guidance too high | Use `--guidance 4.0-5.0` |
| CUDA OOM | Out of memory | Reduce batch size, use BF16 |
| Same image every time | Seed is fixed | Remove `--seed` option |

---

## Code Examples

### Direct Python Usage

```python
from src.inference.generator import generate

# Basic usage
images = generate(
    prompts=["a cute cat", "a beautiful sunset"],
    checkpoint="checkpoints/diffusion.pt",
    vae_checkpoint="checkpoints/vae.pt",
)

for i, img in enumerate(images):
    img.save(f"output_{i}.png")
```

---

## References

- [Stable Diffusion 3 Paper](https://arxiv.org/abs/2403.03206) - Scaling Rectified Flow Transformers
- [CFG Paper](https://arxiv.org/abs/2207.12598) - Classifier-Free Diffusion Guidance
