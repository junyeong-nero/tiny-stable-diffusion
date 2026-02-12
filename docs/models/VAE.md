# 🖼 VAE (Variational AutoEncoder)

> The high-fidelity bridge between pixel space and latent space.

---

## 🔬 Overview

The VAE is the critical first stage of the `tiny-stable-diffusion` pipeline. Its role is to compress $64 \times 64$ RGB images into a dense $8 \times 8 \times 16$ latent representation. This $64 \times$ spatial reduction allows the Diffusion model to operate on a computationally efficient manifold while preserving rich semantic detail.

### Key Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Input Resolution** | $64 \times 64 \times 3$ | RGB Pixels |
| **Latent Resolution** | $8 \times 8 \times 16$ | 16-channel bottleneck |
| **Compression (f8)** | $8 \times$ Spatial reduction | From $64 \rightarrow 8$ px width/height |
| **Model Size** | ~21M Params | Optimized for fast inference |

---

## 🏗 Architecture

Based on the **AutoencoderKL** design, our VAE consists of a symmetric Encoder-Decoder pair with a regularized bottleneck.

### 1. Encoder (Pixel $\rightarrow$ Latent)
- **Downsampling**: 3 stages of strided convolutions ($64 \rightarrow 32 \rightarrow 16 \rightarrow 8$).
- **ResNet Blocks**: Each stage utilizes residual blocks with GroupNorm and SiLU activations to maintain gradient flow.
- **Global Context**: A self-attention layer at the $8 \times 8$ bottleneck allows the encoder to capture long-range dependencies.
- **Stochasticity**: Outputs a mean ($\mu$) and log-variance ($\log \sigma^2$) to parameterize the Gaussian latent distribution.

### 2. Decoder (Latent $\rightarrow$ Pixel)
- **Upsampling**: Reverses the encoder path using nearest-neighbor upsampling followed by refined convolutions to eliminate checkerboard artifacts.
- **Reconstruction**: Restores the original $64 \times 64$ resolution with high fidelity.

---

## 📉 Loss Function & Regularization

The VAE is optimized to balance reconstruction accuracy with latent space smoothness.

$$Loss = \text{MSE}(x, \hat{x}) + \beta \cdot \text{KL}(q(z|x) \| p(z))$$

- **MSE (Mean Squared Error)**: Ensures pixel-level fidelity.
- **KL Divergence**: Regularizes the latent space toward a Standard Normal distribution $\mathcal{N}(0, I)$.
- **Beta ($\beta$)**: We set $\beta = 10^{-6}$. This "light" regularization prioritizes sharp reconstructions, which is essential for low-resolution generation.

---

## ⚙️ Configuration Reference

Key hyperparameters in `config.yaml`:
```yaml
vae_train:
  latent_channels: 16       # SD3 standard bottleneck
  vae_ch: 64                # Base hidden dimension
  vae_ch_mult: [1, 2, 4, 4] # Channel scaling per stage
  kl_weight: 0.000001       # Regularization strength
```

---

## 🛠 Direct CLI Usage

### Image Reconstruction
Verify the VAE's quality by encoding and then decoding an image:
```bash
uv run main.py --reconstruct-vae \
    --input "test.png" \
    --output "reconstructed.png"
```

### Batch Evaluation
```bash
bash scripts/evaluate-vae.sh --input-dir assets/samples
```

---
*Implementation: `src/models/vae.py`*