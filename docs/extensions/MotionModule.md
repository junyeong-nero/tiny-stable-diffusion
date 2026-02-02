# Motion Module (GIF Generation)

> **Status**: ✅ **Implemented** - All core features are complete and tested.

## Overview

The Motion Module extends `tiny-stable-diffusion` to generate GIFs and short animations. It leverages the existing pre-trained **VAE** and **MMDiT/DiT** models by adding a trainable **Motion Module**, following the [AnimateDiff](https://arxiv.org/abs/2307.04725) approach.

### Key Features
- **Temporal Attention**: Self-attention across video frames for motion coherence
- **Zero Initialization**: Stable training with frozen base model
- **Gradient Checkpointing**: Memory optimization for longer sequences
- **Mixed Precision**: AMP support for faster training

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Architecture Overview                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Text Prompt                                                           │
│       │                                                                 │
│       ▼                                                                 │
│   ┌──────────┐                                                          │
│   │   CLIP   │ (Existing, Frozen)                                       │
│   └────┬─────┘                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │              Latent Diffusion (Time Extended)                │      │
│   │  ┌────────────────────────────────────────────────────────┐  │      │
│   │  │   z_t: (B, F, C, H, W)  ←  F = num_frames (16~32)      │  │      │
│   │  └────────────────────────────────────────────────────────┘  │      │
│   │                          │                                   │      │
│   │                          ▼                                   │      │
│   │  ┌──────────────────────────────────────────────────────┐    │      │
│   │  │  MMDiT/DiT (frozen) + Motion Module (trainable)      │    │      │
│   │  │  ┌─────────────────┐  ┌───────────────────────────┐  │    │      │
│   │  │  │ Spatial Attn    │→ │ Temporal Attn (NEW)       │  │    │      │
│   │  │  │ (Existing)      │  │ (Attention across frames) │  │    │      │
│   │  │  └─────────────────┘  └───────────────────────────┘  │    │      │
│   │  └──────────────────────────────────────────────────────┘    │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                          │                                              │
│                          ▼                                              │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │   VAE Decoder (Existing, Frozen)                             │      │
│   │   z → (B, F, 3, 64, 64) → GIF                                │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

Ensure you have trained:
1. **VAE**: `checkpoints/vae.pt`
2. **Diffusion Model**: `checkpoints/diffusion.pt`

### Train Motion Module

```bash
# Basic training with synthetic data
uv run main.py --train-motion --epochs 50

# With memory optimizations
uv run main.py --train-motion --epochs 50 --batch-size 2

# Resume from checkpoint
uv run main.py --train-motion --epochs 100 --resume

# With wandb logging
uv run main.py --train-motion --epochs 50 --wandb
```

### Generate GIFs

```bash
# Simple generation
uv run main.py --generate-gif --prompt "a cat walking"

# With options
uv run main.py --generate-gif \
    --prompt "a bouncing ball" \
    --num-frames 16 \
    --steps 50 \
    --guidance 7.5 \
    --fps 8 \
    --output animation.gif

# Interactive demo
uv run main.py --animation-demo

# Streamlit demo
uv run streamlit run src/demo/app.py
```

## Core Components

### 1. TemporalTransformerBlock
**Location**: `src/models/motion.py`

Performs self-attention along the temporal dimension (frames).

```python
class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_frames: int = 64,
    ):
        # Temporal position embedding (sinusoidal)
        # Self-attention across frames
        # Zero-initialized output for stable training
```

**Key Features**:
- **Input Reshape**: `(B*F, N, D)` → `(B*N, F, D)` (Spatial → Temporal)
- **Positional Embedding**: Sinusoidal embeddings for frame order
- **Zero Initialization**: Output layers initialized to zero

### 2. MotionModule
**Location**: `src/models/motion.py`

Stacks multiple temporal transformer blocks with gradient checkpointing support.

```python
from src.models.motion import MotionModule, create_motion_module

# Create motion module
motion = create_motion_module(
    hidden_size=384,       # Match base model
    num_layers=2,
    num_heads=8,
    max_frames=32,
    use_gradient_checkpointing=True,  # Memory optimization
)

# Forward pass
# Input: (B*F, N, D) spatial features
# Output: (B*F, N, D) with temporal modeling
output = motion(features, num_frames=16)
```

### 3. AnimatedMMDiT
**Location**: `src/models/animated_mmdit.py`

Wraps the frozen base MMDiT with trainable Motion Module.

```python
from src.models.animated_mmdit import AnimatedMMDiT, load_animated_mmdit

# Load from checkpoints
model = load_animated_mmdit(
    base_checkpoint_path="checkpoints/diffusion.pt",
    motion_checkpoint_path="checkpoints/motion.pt",  # Optional
    device="cuda",
    use_gradient_checkpointing=True,
)

# Forward: Handles 4D or 5D input
# (B, F, C, H, W) or (B*F, C, H, W)
velocity = model(noisy_latents, timestep, text_embeds)
```

### 4. AnimatedDiffusion
**Location**: `src/models/animated_diffusion.py`

Extends the Diffusion class for video with Rectified Flow.

```python
from src.models.animated_diffusion import AnimatedDiffusion

diffusion = AnimatedDiffusion(
    num_timesteps=1000,
    num_frames=16,
    guidance_scale=7.5,
    temporal_consistency_weight=0.0,
)

# Training loss
loss, loss_dict = diffusion.training_loss_video(
    model, video_latents, timesteps, text_embeds
)

# Sampling
videos = diffusion.sample_video(
    model, batch_size=1, num_frames=16,
    text_embeds=text_embeds,
    vae_decoder=vae,
)
```

### 5. AnimationGenerator
**Location**: `src/inference/animation_generator.py`

High-level API for GIF generation.

```python
from src.inference.animation_generator import AnimationGenerator

generator = AnimationGenerator(
    vae_checkpoint="checkpoints/vae.pt",
    diffusion_checkpoint="checkpoints/diffusion.pt",
    motion_checkpoint="checkpoints/motion.pt",
    device="cuda",
    num_frames=16,
)

# Generate frames
frames = generator.generate(
    prompt="a cat walking",
    num_frames=16,
    num_steps=50,
    guidance_scale=7.5,
    seed=42,
)

# Save as GIF
generator.save_gif(frames, "output.gif", fps=8)

# Or generate and save in one call
generator.generate_and_save(
    prompt="a bouncing ball",
    output_path="ball.gif",
)
```

## Configuration

All settings are in `config.yaml` under `motion_train`:

```yaml
motion_train:
  # Model checkpoints
  vae_checkpoint: checkpoints/vae.pt
  base_checkpoint: checkpoints/diffusion.pt

  # Motion module architecture
  motion_num_layers: 2
  motion_num_heads: 8
  num_frames: 16

  # Training
  epochs: 100
  batch_size: 4
  learning_rate: 1.0e-4
  gradient_accumulation: 4

  # Memory optimization
  mixed_precision: false
  gradient_checkpointing: false

  # Dataset
  use_synthetic_data: true
  synthetic_size: 1000
  synthetic_pattern: moving_circle
```

## Memory Optimization

### Gradient Checkpointing

Enable in config.yaml:
```yaml
gradient_checkpointing: true
```

Or programmatically:
```python
model.enable_gradient_checkpointing()
```

**Trade-off**: ~30% slower training, ~40% less memory.

### Mixed Precision (AMP)

Enable in config.yaml:
```yaml
mixed_precision: true
```

**Trade-off**: ~2x faster training, fp16 precision.

### Recommended Settings by GPU

| GPU VRAM | batch_size | num_frames | gradient_checkpointing |
|----------|------------|------------|------------------------|
| 8GB      | 1          | 8          | true                   |
| 12GB     | 2          | 16         | true                   |
| 16GB     | 4          | 16         | false                  |
| 24GB+    | 8          | 32         | false                  |

## Training Strategy

### Stage 3: Motion Module Training

Only the Motion Module parameters are trained. VAE, CLIP, and MMDiT are frozen.

1. **Load Checkpoints**: VAE and base diffusion model
2. **Create AnimatedMMDiT**: Frozen base + trainable motion module
3. **Train**: Minimize velocity prediction loss on video data
4. **Save**: Only motion module weights (~2-10MB)

### Synthetic Data for Testing

Use synthetic video patterns to test the pipeline without real video data:

```yaml
use_synthetic_data: true
synthetic_pattern: moving_circle  # or: gradient, noise
```

### Real Video Data

For production training:

```yaml
use_synthetic_data: false
dataset_name: sayakpaul/ucf101-subset
video_field: video
caption_field: label
```

## HuggingFace Hub Upload

Push trained motion module to HuggingFace:

```bash
uv run main.py --train-motion --epochs 50 \
    --push-to-hub \
    --hub-model-id username/motion-module
```

Or programmatically:

```python
from src.utils.hf_upload import push_to_hub

push_to_hub(
    checkpoint_path="checkpoints/motion.pt",
    repo_id="username/motion-module",
    model_type="motion",
)
```

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| TemporalTransformerBlock | ✅ Complete | `src/models/motion.py` |
| MotionModule | ✅ Complete | `src/models/motion.py` |
| AnimatedMMDiT | ✅ Complete | `src/models/animated_mmdit.py` |
| AnimatedDiffusion | ✅ Complete | `src/models/animated_diffusion.py` |
| VideoDataset | ✅ Complete | `src/data/video_dataset.py` |
| Video Transforms | ✅ Complete | `src/data/video_transforms.py` |
| Motion Trainer | ✅ Complete | `src/training/motion_trainer.py` |
| AnimationGenerator | ✅ Complete | `src/inference/animation_generator.py` |
| CLI Integration | ✅ Complete | `main.py` |
| Streamlit Demo | ✅ Complete | `src/demo/app.py` |
| Gradient Checkpointing | ✅ Complete | All motion modules |
| Mixed Precision | ✅ Complete | Training loop |
| HuggingFace Upload | ✅ Complete | `src/utils/hf_upload.py` |

## Tests

Run motion-related tests:

```bash
# All motion tests
uv run pytest tests/test_motion.py tests/test_video_dataset.py \
    tests/test_animated_diffusion.py tests/test_animation_generator.py -v

# Specific test file
uv run pytest tests/test_animation_generator.py -v
```

## References

- **AnimateDiff**: [arXiv:2307.04725](https://arxiv.org/abs/2307.04725)
- **Stable Video Diffusion**: [arXiv:2311.15127](https://arxiv.org/abs/2311.15127)
- **Rectified Flow**: [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
