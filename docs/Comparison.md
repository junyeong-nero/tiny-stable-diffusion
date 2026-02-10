# Comparison: tiny-stable-diffusion vs. Others

> How `tiny-stable-diffusion` stacks up against industry standards like Stable Diffusion 3 and SD 1.5.

---

## ⚖️ Feature Comparison Matrix

| Feature | tiny-stable-diffusion | Stable Diffusion 3 (Medium) | Stable Diffusion 1.5 |
| :--- | :--- | :--- | :--- |
| **Architecture** | **MMDiT** (Transformer) | MMDiT (Transformer) | U-Net (CNN) |
| **Diffusion Type** | **Rectified Flow** (Linear) | Rectified Flow (Linear) | DDPM / DDIM (Gaussian) |
| **Resolution** | **64 × 64** | 1024 × 1024 | 512 × 512 |
| **Latent Size** | **8 × 8 × 16** | 128 × 128 × 16 | 64 × 64 × 4 |
| **Latent Channels** | **16** | 16 | 4 |
| **Parameters** | **~200M** (Backbone) | 2B | 860M |
| **Text Encoder** | **CLIP ViT-B/32** | CLIP-G + CLIP-L + T5-XXL | CLIP ViT-L/14 |
| **Hardware Req.** | **Consumer GPU (8GB)** | A100 / H100 Cluster | A100 Cluster |

---

## 🔍 Detailed Differences

### 1. Transformer vs. U-Net
*   **SD 1.5 (U-Net)**: Uses a convolutional U-Net with inductive biases for spatial locality. While effective, it is harder to scale compared to Transformers.
*   **tiny-sd (MMDiT)**: Adopts the **SD3 architecture**. It treats images as sequences of patches (tokens), processing them with Transformers. This allows for better scaling and more natural multi-modal (text/image) interaction via **Joint Attention**.

### 2. Rectified Flow vs. DDPM
*   **SD 1.5 (DDPM)**: Uses a curved stochastic path from noise to data, requiring complex noise schedules (linear, cosine).
*   **tiny-sd / SD3 (Rectified Flow)**: Uses a **straight-line deterministic path**.
    *   $X_t = (1-t)X_0 + tX_1$
    *   **Advantage**: Simpler to simulate using standard ODE solvers (like Euler), leading to faster convergence with fewer steps.

### 3. High-Dimensional Latents
*   **SD 1.5**: Uses a 4-channel latent space ($64 \times 64 \times 4$).
*   **tiny-sd / SD3**: Uses a **16-channel** latent space.
    *   Even at our $64 \times 64$ resolution, we maintain the 16-channel design. This allows the VAE to capture significantly more semantic information per spatial location, reducing the complexity required of the diffusion transformer.

---

## 🎯 Why "Tiny"?

This project prioritizes **education, accessibility, and hackability** over photorealism.

1.  **Consumer-Grade Training**: Train from scratch on a single gaming GPU or a free Colab instance in just a few hours.
2.  **Readable Implementation**: The code is designed to be read. You can understand the core of `MMDiT` by looking at a few hundred lines of clean PyTorch code.
3.  **The Perfect Sandbox**: Want to experiment with a new attention mechanism or loss function? It's much easier to iterate on a 200M parameter model than a 2B parameter one.
4.  **Full Logic, Small Scale**: We use the **exact same training dynamics** and architectural innovations as SD3, providing a faithful "miniature" version of the state-of-the-art.

---

## 📉 Expectations

*   **What it is NOT**: A replacement for Midjourney or SDXL. It won't generate high-fidelity faces or complex photorealistic textures.
*   **What it IS**: A powerful educational tool that generates coherent objects (animals, simple scenes, shapes) and demonstrates the mathematical beauty of modern diffusion models.