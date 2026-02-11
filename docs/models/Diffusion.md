# Diffusion & Rectified Flow

> Mastering the art of iterative denoising.

---

## 🔬 Overview

`tiny-stable-diffusion` utilizes **Rectified Flow**, the same mathematical framework powering Stable Diffusion 3. Unlike traditional Gaussian diffusion (DDPM), which follows a curved stochastic path, Rectified Flow learns to map noise to data along a **straight, deterministic line**.

### Why Rectified Flow?
- **Efficiency**: Straight paths are easier for ordinary differential equation (ODE) solvers to follow, requiring fewer steps (e.g., 20–50 instead of 100+).
- **Simplicity**: The objective is to predict **Velocity** ($v = X_1 - X_0$), which is more intuitive than predicting noise ($\epsilon$).

---

## 🏗 The Training Process

### 1. Linear Interpolation
During training, we create a noisy latent $X_t$ by interpolating between the clean image latent $X_0$ and pure noise $X_1$:
$$X_t = (1-t)X_0 + tX_1$$
where $t \in [0, 1]$.

### 2. Velocity Prediction
The MMDiT model is trained to predict the velocity vector that points from $X_0$ to $X_1$:
$$\text{Target } v = X_1 - X_0$$
$$\text{Loss} = \| \text{MMDiT}(X_t, t, c) - v \|^2$$
where $c$ is the text condition.

### 3. Logit-Normal Timestep Sampling
Instead of sampling timesteps $t$ uniformly, we use a **Logit-Normal** distribution. This focuses the model's training on the middle timesteps where the denoising task is most critical for final image quality.

---

## 🚀 Generation (Inference)

### Euler ODE Solver
To generate an image, we start at $t=1.0$ (pure noise) and solve the ODE:
$$\frac{dX_t}{dt} = v(X_t, t, c)$$
using a simple Euler step:
$$X_{t-dt} = X_t - v \times dt$$

### Classifier-Free Guidance (CFG)
During inference, we extrapolate the velocity vector away from the "unconditional" (empty prompt) prediction to boost prompt adherence:
$$v_{guided} = v_{uncond} + s \times (v_{cond} - v_{uncond})$$

---

## ⚙️ Configuration

Key settings in `config.yaml`:
```yaml
diffusion_train:
    num_timesteps: 1000
    guidance_scale: 7.5
    use_logit_normal_sampling: true
    logit_mean: 0.0
    logit_std: 1.0
```

---

## 📊 Latest Evaluation (checkpoints/diffusion.pt)

From `results/evaluation/diffusion/eval_results.json`:

| Metric | Value |
|---|---:|
| FID | 343.1047 |
| CLIP-FID | 47.2341 |
| Inception Score (mean) | 1.0000 |
| Inception Score (std) | 0.0000000843 |
| CLIP Score (mean) | 25.3393 |
| CLIP Score (std) | 0.0000 |
| Generated samples | 10 |
| Real samples | 10 |

> Note: these metrics were computed on a very small sample size (`num_generated=10`, `num_real=10`), so they are useful for quick regression checks but not for stable quality benchmarking.

---

## 📚 Implementation Reference
- **Diffusion Process**: `src/models/diffusion.py`
- **Inference Generator**: `src/inference/generator.py`
- **Training Loop**: `src/training/trainer.py`
