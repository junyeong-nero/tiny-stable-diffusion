# 🌊 Diffusion & Rectified Flow

> Mathematical framework for straight-line noise transport.

---

## 🔬 Overview

`tiny-stable-diffusion` implements **Rectified Flow**, the state-of-the-art framework for training and sampling diffusion models. Unlike Gaussian diffusion (DDPM/DDIM), which follows a curved stochastic path, Rectified Flow maps noise to data along a **straight, deterministic line**.

### Why Rectified Flow?
- **Speed**: Straight paths are the easiest for ODE solvers (like Euler) to follow, requiring fewer steps.
- **Intuitiveness**: The model learns to predict **Velocity** ($v$), not complex noise residuals.

---

## 🏗 The Training Process

### 1. Linear Interpolation
We create a noisy latent $z_t$ by interpolating between the clean image $z_0$ and pure Gaussian noise $\epsilon$:
$$z_t = (1-t)z_0 + t\epsilon$$
where $t \in [0, 1]$. As $t$ increases, the image becomes noisier.

### 2. Velocity Prediction
The MMDiT model learns the velocity vector $v$ that points from $z_0$ toward $\epsilon$:
$$\text{Target } v = \epsilon - z_0$$
$$\text{Loss} = \| \text{MMDiT}(z_t, t, c) - v \|^2$$

### 3. Logit-Normal Sampling
Timesteps $t$ are sampled using a **Logit-Normal distribution**. This focuses training on the "mid-range" of the denoising process, where the most important structural features are formed.

---

## 🚀 Inference (Sampling)

### Euler ODE Solver
To generate an image, we start at $t=1.0$ (noise) and solve the flow equation backward to $t=0.0$:
$$\frac{dz_t}{dt} = v(z_t, t, c)$$
In discrete steps:
$$z_{t-dt} = z_t - v \times dt$$

### Classifier-Free Guidance (CFG)
We adjust the velocity vector to steer the generation toward the prompt:
$$v_{guided} = v_{uncond} + \text{scale} \times (v_{cond} - v_{uncond})$$

---

## ⚙️ Configuration Reference

```yaml
diffusion_train:
  num_timesteps: 1000
  use_logit_normal_sampling: true
  logit_mean: 0.0
  logit_std: 1.0
  guidance_scale: 7.5
```

---

## 📚 Reference Implementation
- **Mathematical Flow**: `src/models/diffusion.py`
- **Inference Logic**: `src/inference/generator.py`
- **Training Objectives**: `src/training/trainer.py`