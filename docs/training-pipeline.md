# Training Pipeline Deep Dive

> This document provides a detailed explanation of tiny-stable-diffusion's training pipeline.

## Table of Contents

1. [Two-Stage Training Overview](#two-stage-training-overview)
2. [Stage 1: VAE Training](#stage-1-vae-training)
3. [Stage 2: Diffusion Training](#stage-2-diffusion-training)
4. [Optimizer & Scheduler](#optimizer--scheduler)
5. [EMA (Exponential Moving Average)](#ema-exponential-moving-average)
6. [Checkpointing](#checkpointing)
7. [HuggingFace Hub Upload](#huggingface-hub-upload)

---

## Two-Stage Training Overview

tiny-stable-diffusion is trained in two stages:

**Stage 1: VAE Training**
- Goal: Learn to compress images into latent space and reconstruct them
- Flow: Image (64×64) → Encoder → Latent (8×8×16) → Decoder → Image
- Loss: MSE(input, reconstruction) + β × KL(posterior || prior)
- Output: `checkpoints/vae.pt`

**Stage 2: Diffusion Training**
- Goal: Learn to generate images matching text conditions in latent space
- Flow: Image → [Frozen VAE] → Latent → Add Noise → DiT → Predict Noise
- Text conditioning: Text → CLIP → text embedding
- Loss: MSE(predicted_noise, actual_noise) × Min-SNR weight
- Output: `checkpoints/diffusion.pt`

### Why Two-Stage?

| Approach | Pros | Cons |
|----------|------|------|
| End-to-End | Simple | Very slow, unstable |
| **Two-Stage** | Efficient, stable | Requires two training phases |

Two-Stage advantages:
1. **Well-defined latent space first** - Diffusion operates in a meaningful space
2. **Independent debugging** - Easier to identify issues at each stage
3. **VAE reusability** - Can be used with different diffusion models

---

## Stage 1: VAE Training

> File location: `src/training/vae_trainer.py`

### Training Loop Details

```python
def train_vae(config, use_wandb=False):
    """
    VAE Training Main Loop

    Steps:
    1. Load dataset
    2. Initialize model
    3. Set up Optimizer/Scheduler
    4. Train for each epoch
    5. Save checkpoint
    6. Generate samples (validation)
    """
```

### Step-by-Step Training

1. **Batch Loading**: `images = next(dataloader)` - Shape: (B, 3, 64, 64), range [-1, 1]

2. **Forward Pass**: `reconstruction, mean, logvar = vae(images)`
   - `h = encoder(images)` → (B, 32, 8, 8)
   - `mean, logvar = split(h)` → each (B, 16, 8, 8)
   - `z = mean + exp(0.5*logvar) * ε` (reparameterization)
   - `reconstruction = decoder(z)` → (B, 3, 64, 64)

3. **Loss Computation**:
   - `recon_loss = MSE(images, reconstruction) = mean((images - reconstruction)²)`
   - `kl_loss = -0.5 * mean(1 + logvar - mean² - exp(logvar))`
   - `total_loss = recon_loss + kl_weight * kl_loss` (default kl_weight: 1e-6)

4. **Backward & Update**:
   - `optimizer.zero_grad()`
   - `loss.backward()`
   - `optimizer.step()`
   - `scheduler.step()`

### VAE Loss Analysis

#### Reconstruction Loss

```python
# Mean Squared Error
recon_loss = F.mse_loss(reconstruction, x, reduction="mean")
```

**Purpose**: Minimize difference between input and reconstructed images

**Formula**: L_recon = (1/N) × Σ (x_i - x̂_i)²
- N = batch_size × 3 × 64 × 64
- x_i: original pixel
- x̂_i: reconstructed pixel

**Value range**: 0 ~ 4 (when input is in [-1, 1] range)
**Good value**: < 0.01 (nearly perfect reconstruction)

#### KL Divergence Loss

```python
# KL(q(z|x) || p(z)) where p(z) = N(0, I)
kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
```

**Purpose**: Regularize latent distribution to be close to standard normal

**Derivation**:
- q(z|x) = N(μ, σ²)
- p(z) = N(0, 1)
- KL(q||p) = 0.5 × (μ² + σ² - 1 - log(σ²)) = -0.5 × (1 + log(σ²) - μ² - σ²)

Note: logvar = log(σ²), so exp(logvar) = σ²

**Role**:
- μ → 0: Center at origin
- σ → 1: Unit variance
- Enables meaningful image generation when sampling z ~ N(0, I)

#### KL Weight (β-VAE)

```python
kl_weight = 1e-6  # β in β-VAE
```

**Trade-off**:
- **kl_weight ↑** (e.g., 1e-3): More regularized latent space, closer to N(0, I), but reconstruction quality suffers
- **kl_weight ↓** (e.g., 1e-8): More accurate reconstruction, weaker latent structure, risk of posterior collapse
- **Recommended**: 1e-6 - Good balance between reconstruction quality and latent structure

### Mixed Precision Training

```python
use_amp = config.get("mixed_precision", False) and device.type == "cuda"
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# Training step with AMP
with torch.cuda.amp.autocast(enabled=use_amp):
    reconstruction, mean, logvar = model(images)
    loss, loss_dict = model.training_loss(images, kl_weight)

if use_amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

**Mixed Precision (FP16/BF16)**:
- Forward pass: FP16 operations (2x memory/speed improvement)
- Backward pass: FP32 gradients (maintains accuracy)
- GradScaler: Prevents FP16 underflow

**Benefits**:
- ~50% memory reduction
- ~1.5-2x faster training
- Nearly identical quality

---

## Stage 2: Diffusion Training

> File location: `src/training/trainer.py`

### Training Loop Details

```python
def train_diffusion(config, use_wandb=False):
    """
    Diffusion Training Main Loop

    Requirements:
    - Pre-trained VAE checkpoint
    - VAE is frozen (not trained)

    Steps:
    1. Load & Freeze VAE
    2. Load CLIP text encoder
    3. Initialize DiT/MMDiT
    4. Initialize Diffusion process (Rectified Flow)
    5. Train for each epoch
    """
```

### Step-by-Step Training

1. **Batch Loading**:
   - `images, captions = next(dataloader)`
   - images: (B, 3, 64, 64), range [-1, 1]
   - captions: list of strings

2. **VAE Encoding** (frozen, no grad):
   ```python
   with torch.no_grad():
       latents = vae.encode_to_latent(images)  # (B, 16, 8, 8)
   ```

3. **Text Encoding** (frozen, no grad):
   ```python
   with torch.no_grad():
       text_embeds = clip.encode(captions)  # (B, 512)
   ```

4. **Timestep Sampling** (Logit-Normal):
   ```python
   # Sample t ~ LogitNormal(0, 1), mapped to [0, 1000]
   t = sigmoid(randn(B)) * 1000
   ```
   *SD3 uses Logit-Normal sampling to focus training on middle timesteps.*

5. **Add Noise** (Forward Process - Linear Interpolation):
   ```python
   noise = torch.randn_like(latents)
   # x_t = (1 - t) * x_0 + t * noise
   noisy_latents = (1 - t) * latents + t * noise
   ```
   *Rectified Flow uses a straight path from data to noise.*

6. **CFG Dropout** (During training):
   ```python
   for i in range(B):
       if random() < cfg_probability:  # e.g., 10%
           text_embeds[i] = uncond_embed  # empty string ""
   ```

7. **Velocity Prediction**:
   ```python
   # Target velocity v = noise - latents
   v_pred = model(noisy_latents, t, text_embeds)  # (B, 16, 8, 8)
   ```

8. **Loss Computation** (Min-SNR weighted MSE):
   ```python
   mse = mean((v_pred - v_target)², dim=[1,2,3])
   
   # SNR = (1-t)² / t²
   snr = ((1-t)**2) / (t**2)
   weight = min(snr, γ) / snr  # γ=5.0
   
   loss = mean(mse × weight)
   ```

9. **Backward & Update**:
   ```python
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   scheduler.step()
   ```

10. **EMA Update**:
    ```python
    if use_ema:
        ema.update()  # θ_ema = decay × θ_ema + (1-decay) × θ
    ```

### CFG Warmup

```python
# Gradually increase CFG probability
initial_cfg = 0.0   # Start: no unconditional dropout
final_cfg = 0.1     # Final: 10% dropout
cfg_warmup_epochs = 10

for epoch in range(epochs):
    if epoch < cfg_warmup_epochs:
        progress = epoch / cfg_warmup_epochs
        cfg_prob = initial_cfg + (final_cfg - initial_cfg) * progress
    else:
        cfg_prob = final_cfg
```

**CFG Warmup rationale**:
- **Early training (epoch 0-10)**: Model is still unstable, learning text-image connections; lower CFG dropout is more stable
- **Later training (epoch 10+)**: Model has stabilized; CFG dropout makes it more robust and improves inference CFG effectiveness

### Validation Sample Generation

```python
def generate_samples(model, diffusion, clip_encoder, vae_decoder, prompts, ...):
    """
    Generate validation samples during training

    Every N epochs:
    1. Use fixed validation prompts
    2. Generate images with Euler ODE sampling
    3. Save images (samples/epoch_N/)
    """
```

**Example validation prompts**:
- "a photo of a cat"
- "a rocket flying in space"
- "a robot with blue eyes"
- "a beautiful sunset over the ocean"
- "a red sports car"

**Generation process**:
1. Encode each prompt with CLIP
2. Start from random noise (B, 16, 8, 8)
3. Denoise with DDIM for 50 steps
4. Convert to image with VAE decoder
5. Save: `samples/epoch_N/00_a_photo_of_a_cat.png`

---

## Optimizer & Scheduler

### AdamW Optimizer

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],  # e.g., 1e-4
    betas=(0.9, 0.999),          # momentum coefficients
    eps=1e-8,                    # numerical stability
    weight_decay=0.0,            # no L2 regularization
)
```

**Why AdamW**:
- Adam + decoupled weight decay
- Standard for Transformer-based models
- Stable training

**Hyperparameters**:
- lr: 1e-4 (both VAE/Diffusion)
- betas: (0.9, 0.999) - standard values
- weight_decay: 0.0 - following DiT paper

### Cosine Annealing with Warmup

```python
# Calculate steps
num_steps_per_epoch = len(dataloader)
total_steps = epochs * num_steps_per_epoch
warmup_steps = total_steps // 20  # 5% warmup

def lr_lambda(step):
    if step < warmup_steps:
        # Linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Learning Rate Schedule**:
- **Phase 1: Linear Warmup (0 → 5%)**: lr: 0 → target_lr; Purpose: Stabilize early training
- **Phase 2: Cosine Annealing (5% → 100%)**: lr: target_lr → 0 (cosine curve); Purpose: Smooth learning rate decay

---

## EMA (Exponential Moving Average)

> File location: `src/training/ema.py`

### EMA Concept

```python
class EMA:
    """
    Exponential Moving Average of model parameters

    Formula:
        θ_ema = decay × θ_ema + (1 - decay) × θ

    Where:
        - θ: current model parameters
        - θ_ema: EMA parameters
        - decay: typically 0.9999
    """
```

### Why Use EMA?

**Training Parameters (θ)**:
- Updated every step
- Can be noisy
- Reflects latest gradients

**EMA Parameters (θ_ema)**:
- Average of many steps
- More stable
- Usually better performance

With decay=0.9999: θ_ema represents a weighted average of approximately 10,000 steps (1 / (1 - 0.9999) ≈ 10,000)

**For inference**: Use θ_ema for better quality

### EMA Implementation

```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize: copy current parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Call after each training step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # θ_ema = decay × θ_ema + (1 - decay) × θ
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_((1 - self.decay) * param.data)

    def apply_shadow(self):
        """Before inference: apply EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """After inference: restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
```

---

## Checkpointing

> File location: `src/training/checkpoint.py`

### Checkpoint Structure

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, config, ema=None):
    """
    Save checkpoint

    Contents:
    {
        "epoch": int,              # current epoch
        "loss": float,             # current loss
        "model_state_dict": dict,  # model weights
        "optimizer_state_dict": dict,  # optimizer state
        "scheduler_state_dict": dict,  # scheduler state
        "model_config": {          # model settings
            "model_size": "S",
            "patch_size": 2,
            "image_size": 32,
            "model_type": "dit",
            "qk_rmsnorm": True,
            "register_tokens": 0,
        },
        "ema_state_dict": dict,    # EMA weights (optional)
    }
    """
```

### Best Model Saving

```python
# After each epoch
if avg_loss < best_loss:
    best_loss = avg_loss
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        loss=avg_loss,
        path=checkpoint_path,  # e.g., checkpoints/diffusion.pt
        config=config,
        ema=ema,
    )
```

**Saving strategy**:
- Save based on best loss only
- Saves disk space
- Keeps only the best model

**Alternatives**:
- Save every N epochs
- Keep last K checkpoints
- Save to separate directories per epoch

### Checkpoint Loading

```python
def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Load checkpoint

    Use cases:
    1. Resume training
    2. Load model for inference
    3. Fine-tuning starting point
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]
```

---

## HuggingFace Hub Upload

> File location: `src/utils/hf_upload.py`

### Feature Overview

```python
def push_to_hub(checkpoint_path, repo_id, model_type, config=None, ...):
    """
    Upload trained model to HuggingFace Hub

    Upload contents:
    1. {model_type}.pt - checkpoint file
    2. config.json - training settings
    3. README.md - model card (auto-generated)

    Usage:
    uv run main.py --train-vae --push-to-hub --hub-model-id username/my-vae
    """
```

### CLI Arguments

```bash
# HuggingFace Hub upload options
--push-to-hub           # Enable upload
--hub-model-id ID       # Repository ID (required)
--hub-private           # Private repository
```

### Upload Process

1. **Training complete**: `checkpoints/vae.pt` created
2. **push_to_hub() called**:
   - HF API authentication (HF_TOKEN env var or huggingface-cli)
   - Create repository (if doesn't exist)
   - Upload files
3. **Uploaded file structure**:
   ```
   username/my-vae/
   ├── vae.pt          # checkpoint
   ├── config.json     # training settings
   └── README.md       # model card
   ```
4. **Result**: https://huggingface.co/username/my-vae

### Environment Setup

```bash
# Method 1: Environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxx

# Method 2: CLI login
huggingface-cli login

# Method 3: config.yaml
hub:
    push_to_hub: true
    model_id: username/model-name
    private: false
```

---

## Training Tips

### Recommended Hyperparameters

| Stage | Parameter | VAE | Diffusion |
|-------|-----------|-----|-----------|
| Epochs | `epochs` | 100 | 200 |
| Batch Size | `batch_size` | 32-128 | 32 |
| Learning Rate | `learning_rate` | 4e-4 | 1e-4 |
| KL Weight | `kl_weight` | 1e-6 | - |
| CFG Prob | `cfg_probability` | - | 0.1 |
| EMA Decay | `ema_decay` | - | 0.9999 |

### Memory Optimization

```python
# 1. Gradient Accumulation
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. Mixed Precision
config["mixed_precision"] = True

# 3. Smaller Batch Size
config["batch_size"] = 16  # if GPU memory is limited

# 4. Gradient Checkpointing (if implemented)
model.gradient_checkpointing_enable()
```

### Debugging Checklist

- [ ] Is VAE reconstruction working properly? → Check `samples/vae_epoch_N/` images
- [ ] Is KL loss not diverging? → kl_loss < 100 is normal
- [ ] Is Diffusion loss decreasing? → Initial ~1.0 → Final ~0.1
- [ ] Are generated images related to prompts? → Check `samples/epoch_N/`
- [ ] No NaN/Inf loss? → Lower learning rate if occurring

---

## References

- [β-VAE Paper](https://openreview.net/forum?id=Sy2fzU9gl) - Understanding disentangling in β-VAE
- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [AdamW Paper](https://arxiv.org/abs/1711.05101) - Decoupled Weight Decay Regularization
- [EMA in Diffusion](https://arxiv.org/abs/2102.09672) - Improved Denoising Diffusion Probabilistic Models
