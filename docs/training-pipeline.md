# 🏛 Training Pipeline Philosophy

> Why we train the way we do. A deep dive into the Latent Diffusion Model strategy.

---

## 🧩 The Two-Stage Approach

Training a diffusion model directly on pixels is computationally prohibitive. `tiny-stable-diffusion` follows the **Latent Diffusion** strategy to maximize efficiency.

| Stage | Focus | Learnable Parameters | Input/Output |
| :--- | :--- | :--- | :--- |
| **1: VAE** | Spatial Compression | 21M | $64 \times 64 \times 3 \rightleftarrows 8 \times 8 \times 16$ |
| **2: Diffusion** | Semantic Mapping | 87M - 559M | Text Embeddings $\rightarrow$ Latents |

### 💡 Why separate them?
1.  **Complexity Reduction**: The diffusion model works on a $64 \times$ smaller representation.
2.  **Stability**: We freeze the VAE during Stage 2, ensuring the "ground truth" for the diffusion model doesn't shift during training.

---

## 🏗 Stage 1: VAE Optimization

**Objective**: Perfect reconstruction while maintaining a Gaussian latent distribution.

- **Loss Composition**: $Loss = MSE + \beta \cdot KL\_Divergence$
- **Perceptual Tuning**: We use a very low $\beta$ ($10^{-6}$). This prioritizes pixel-perfect detail over extreme latent regularity, which is ideal for low-resolution $64 \times 64$ generation.
- **Bottleneck**: The 16-channel latent space is wide enough to prevent "information collapse" but narrow enough to force meaningful compression.

---

## 🔥 Stage 2: Diffusion & Rectified Flow

**Objective**: Learn the velocity field that maps noise to data.

### 1. Rectified Flow Training
Unlike traditional models that predict noise ($\epsilon$), we predict **Velocity** ($v$).
- **Linear Path**: During training, we interpolate: $z_t = (1-t)z_0 + t\epsilon$.
- **Straightness**: This linear interpolation forces the model to learn a straight path, making inference much faster and more accurate with simple ODE solvers.

### 2. CFG Dropout (Conditioning)
During training, we randomly drop the text prompt for **10% of samples**.
- **Reason**: This forces the model to learn both conditional and unconditional generation, which is the prerequisite for **Classifier-Free Guidance** during inference.

### 3. EMA (Exponential Moving Average)
We maintain a "Shadow Copy" of the model weights that updates slowly ($\text{decay} = 0.9999$).
- **Benefit**: EMA weights are significantly smoother and produce fewer artifacts. They represent a temporal consensus of the model's knowledge.

---

## 📈 Monitoring & Best Practices

1.  **Visual Over Metrics**: Loss curves can be misleading. Always judge progress by the samples generated in the `samples/` directory.
2.  **Latent Statistics**: In W&B, monitor the mean and standard deviation of the VAE latents. They should stay close to $0.0$ and $1.0$ respectively.
3.  **Warmup**: Use a small learning rate warmup (1,000 steps) to prevent early training instability.

---
*Reference: [Rectified Flow Foundations](https://arxiv.org/abs/2209.03003)*
