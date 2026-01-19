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

**Step 3: Iterative Denoising (DDIM 50 steps)**

For each timestep t in [999, 979, 959, ..., 19, 0]:
1. Compute conditional prediction: `noise_cond = DiT(z_t, t, text_embed)`
2. Compute unconditional prediction: `noise_uncond = DiT(z_t, t, uncond_embed)`
3. Apply Classifier-Free Guidance: `noise_pred = uncond + scale × (cond - uncond)`
4. Perform DDIM step: `z_{t-1} = ddim_step(z_t, noise_pred, t)`

Output: `z_0` (1, 16, 8, 8) - Clean latent

**Step 4: VAE Decoding**
- Pass the clean latent through post_quant_conv and Decoder to produce image (1, 3, 64, 64)
- Normalize from [-1, 1] to [0, 1]: `image = (image + 1) / 2`
- Clamp values: `image = clamp(image, 0, 1)`

**Output**: 64×64 RGB image

---

## Sampling Algorithms

### DDPM vs DDIM Comparison

| Feature | DDPM | DDIM |
|---------|------|------|
| Sampling method | Stochastic | Deterministic |
| Required steps | 1000 | **50** (or fewer) |
| Speed | Slow | **Fast** |
| Reproducibility | Different each time | **Same seed = same result** |
| Quality | Good | Good (may differ slightly) |

### DDPM Sampling

```python
def p_sample(self, model, x_t, t, text_embeds, use_cfg=True):
    """
    DDPM Reverse Process: Stochastic reverse sampling

    Mathematical background:
    p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)

    Process:
    1. Predict noise: ε_θ = model(x_t, t, text)
    2. Predict x_0: x̂_0 = (x_t - √(1-ᾱ_t)ε_θ) / √(ᾱ_t)
    3. Compute posterior mean: μ = coef1 × x̂_0 + coef2 × x_t
    4. Add noise: x_{t-1} = μ + σ_t × z, z ~ N(0, I)
    """
```

**DDPM Sampling Characteristics**:
- Random noise added at each step (stochastic)
- Different results from same starting point
- Good diversity but difficult to reproduce

### DDIM Sampling

```python
def ddim_sample(self, model, x_t, t, text_embeds, eta=0.0, use_cfg=True):
    """
    DDIM Reverse Process: Deterministic reverse sampling

    Key difference: eta parameter controls stochasticity
    - eta = 0: Fully deterministic (recommended)
    - eta = 1: Same as DDPM

    Process:
    1. Predict noise: ε_θ = model(x_t, t, text)
    2. Predict x_0: x̂_0 = (x_t - √(1-α_t)ε_θ) / √(α_t)
    3. Compute σ: σ_t = η × √((1-α_{t-1})/(1-α_t)) × √(1-α_t/α_{t-1})
    4. Compute direction: dir = √(1 - α_{t-1} - σ²) × ε_θ
    5. Sample: x_{t-1} = √(α_{t-1}) × x̂_0 + dir + σ × z
    """
```

**DDIM Sampling Characteristics** (eta=0):
- No noise added when eta=0 (deterministic)
- Compresses 1000 steps to 50 steps
- Same seed always produces the same result
- Timestep interval: 1000/50 = 20 timesteps skipped
- Timesteps: [999, 979, 959, ..., 39, 19, 0]

### Timestep Selection

```python
# DDIM timestep calculation
if use_ddim:
    step_indices = torch.linspace(0, num_timesteps - 1, num_steps + 1)
    timesteps = torch.flip(step_indices.long(), dims=[0])[:-1]
    # Example: when num_steps=50
    # timesteps = [999, 979, 959, ..., 39, 19, 0] (50 values)
else:
    timesteps = torch.arange(num_timesteps - 1, -1, -1)
    # timesteps = [999, 998, 997, ..., 2, 1, 0] (1000 values)
```

---

## Classifier-Free Guidance

### CFG Concept

**Purpose**: Strengthen the influence of text conditions

**Method**:
1. Conditional prediction: `ε_cond = model(x_t, t, text_embed)`
2. Unconditional prediction: `ε_uncond = model(x_t, t, uncond_embed)`
3. Apply guidance: `ε̃ = ε_uncond + s × (ε_cond - ε_uncond)`

**Mathematical interpretation**: `ε̃ = (1 - s) × ε_uncond + s × ε_cond`

**Guidance scale values**:
- s = 1.0: Same as conditional only (no guidance)
- s > 1.0: Stronger movement toward conditional direction
- s = 7.5: Recommended value (SD3 default)

### Guidance Scale Effects

| Scale | Effect |
|-------|--------|
| s = 1.0 | Uses conditional generation only; weak text reflection; high diversity; may have lower quality |
| s = 3.0 | Weak guidance; moderate text reflection; maintains diversity |
| s = 7.5 (recommended) | Good balance; text well reflected; good quality; SD3 default |
| s = 15.0+ | Very strong guidance; excessive text reflection; color saturation; possible artifacts |

### CFG Implementation

```python
# Applying CFG during inference
def denoise_with_cfg(model, x_t, t, text_embed, uncond_embed, guidance_scale):
    # 1. Conditional prediction
    noise_cond = model(x_t, t, text_embed)

    # 2. Unconditional prediction
    noise_uncond = model(x_t, t, uncond_embed)

    # 3. Apply CFG
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    return noise_pred
```

**Computational cost**:
- CFG requires 2 model calls (cond + uncond)
- Generation time approximately doubled
- Quality improvement is significant

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

    # 3. Calculate timesteps
    timesteps = [999, 979, 959, ..., 19, 0]  # 50 values

    # 4. Iterative denoising
    for t in tqdm(timesteps):
        t_batch = torch.full((B,), t, device=device)

        # DDIM step
        x_t = ddim_sample(model, x_t, t_batch, text_embeds, eta=0.0)

    # 5. VAE decoding
    if vae_decoder:
        x_t = vae_decoder.decode_from_latent(x_t)

    # 6. Normalize to [0, 1]
    x_t = (x_t + 1.0) / 2.0
    x_t = torch.clamp(x_t, 0.0, 1.0)

    return x_t
```

### Generation Progression

The denoising process transforms random noise into a coherent image through multiple steps:

1. **Step 0 (t=999)**: Pure noise - SNR ≈ 0.001 (almost pure noise)
2. **Step 10 (t=779)**: Rough structure begins to emerge - SNR ≈ 0.1
3. **Step 25 (t=479)**: Clear shapes become visible - SNR ≈ 1.0
4. **Step 40 (t=199)**: Fine details appear - SNR ≈ 10
5. **Step 50 (t=0)**: Final image - SNR → ∞ (clean image)

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
| `--steps` | 50 | 10-1000 | Number of sampling steps. Higher = better quality, slower |
| `--guidance` | 7.5 | 1.0-20.0 | CFG scale. Higher = stronger prompt reflection |
| `--seed` | None | int | Reproducibility seed. Same seed = same result |
| `--num-samples` | 1 | 1-16 | Number of images to generate |

### Steps vs Quality

| Steps | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| 10 | Very fast | Low (noisy residue) | Quick prototyping |
| 25 | Fast | Acceptable | General generation |
| 50 (recommended) | Moderate | Good | Default setting |
| 100 | Slow | Very good | High-quality generation |
| 1000 (DDPM) | Very slow | Best | Research/comparison |

### Seed Usage

```python
# Reproducible generation
--seed 42

# Different results each time
--seed None  # (default)

# Generate multiple variations
for seed in [42, 43, 44, 45]:
    generate(prompt, seed=seed)
```

**How seed works**:
1. `torch.manual_seed(42)` is set
2. Initial noise z_T becomes fixed
3. With DDIM (eta=0), the entire process is deterministic
4. Same prompt + seed = same image

**Applications**:
- A/B testing: Compare different prompts with same seed
- Bug reproduction: Record seed when issues occur
- Gallery: Save good seeds for reuse

---

## Performance Optimization

### Memory Optimization

```python
# 1. Disable gradients during inference
with torch.no_grad():
    image = generate(prompt)

# 2. Half precision (FP16)
model = model.half()  # 50% memory savings
vae = vae.half()

# 3. VAE slicing (for large batches)
def decode_with_slicing(vae, latents, slice_size=4):
    images = []
    for i in range(0, len(latents), slice_size):
        batch = latents[i:i+slice_size]
        images.append(vae.decode(batch))
    return torch.cat(images)
```

### Speed Optimization

```python
# 1. Fewer steps
--steps 25  # instead of 50

# 2. torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 3. Batch generation
--num-samples 4  # Generate 4 at once (more efficient than sequential)

# 4. Use EMA model (better quality = fewer steps needed)
ema.apply_shadow()  # Apply EMA weights
```

### GPU Utilization

```python
# Automatic device detection
device = "cuda" if torch.cuda.is_available() else "cpu"

# MPS (Apple Silicon)
if torch.backends.mps.is_available():
    device = "mps"

# Multi-GPU (DataParallel)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

## Interactive Demo

```bash
# Run interactive demo
uv run main.py --demo
```

### Demo Usage

The interactive demo provides a simple interface:
1. Enter a prompt (or 'quit' to exit)
2. Watch the generation progress bar
3. Image is saved automatically
4. Enter another prompt to continue

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Noisy images | Not enough steps | Use `--steps 100` |
| Prompt not reflected | Guidance too low | Use `--guidance 10.0` |
| Oversaturated colors | Guidance too high | Use `--guidance 5.0` |
| CUDA OOM | Out of memory | Reduce batch size, use FP16 |
| Same image every time | Seed is fixed | Remove `--seed` option |

### Debugging Checklist

- [ ] Do checkpoint files exist? → Check `checkpoints/diffusion.pt`, `checkpoints/vae.pt`
- [ ] Is CLIP installed? → `pip install git+https://github.com/openai/CLIP.git`
- [ ] Is GPU memory sufficient? → Check with `nvidia-smi`, reduce batch size
- [ ] Is model size correct? → Compare checkpoint's model_config with current settings
- [ ] Is text encoding working? → Check `print(text_embeds.shape)` outputs (B, 512)

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

### Custom Pipeline

```python
import torch
from src.models.vae import create_vae
from src.models.factory import DiT
from src.models.diffusion import Diffusion
from src.text_encoder.clip_encoder import CLIPTextEncoder

# Load models
device = "cuda"

vae = create_vae()
vae.load_state_dict(torch.load("checkpoints/vae.pt")["model_state_dict"])
vae = vae.to(device).eval()

clip = CLIPTextEncoder().to(device)

dit = DiT(in_channels=16, image_size=8, patch_size=2, model_size="S")
dit.load_state_dict(torch.load("checkpoints/diffusion.pt")["model_state_dict"])
dit = dit.to(device).eval()

# Compute uncond embedding
uncond_embed = clip.encode([""])

diffusion = Diffusion(
    num_timesteps=1000,
    guidance_scale=7.5,
    uncond_embed=uncond_embed,
)

# Generate
prompt = "a robot playing guitar"
text_embed = clip.encode([prompt])

with torch.no_grad():
    latent = diffusion.sample(
        model=dit,
        shape=(1, 16, 8, 8),
        text_embeds=text_embed,
        num_steps=50,
        use_ddim=True,
        vae_decoder=vae,
        seed=42,
    )

# Save image
from PIL import Image
import numpy as np

img = latent[0].permute(1, 2, 0).cpu().numpy()
img = (img * 255).astype(np.uint8)
Image.fromarray(img).save("output.png")
```

---

## References

- [DDIM Paper](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models
- [CFG Paper](https://arxiv.org/abs/2207.12598) - Classifier-Free Diffusion Guidance
- [Progressive Distillation](https://arxiv.org/abs/2202.00512) - Faster sampling
- [Consistency Models](https://arxiv.org/abs/2303.01469) - 1-step generation
