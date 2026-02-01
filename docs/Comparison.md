# Comparison: tiny-stable-diffusion vs. Others

> A detailed breakdown of how `tiny-stable-diffusion` compares to the original Stable Diffusion 3 and SD 1.5.

---

## Feature Comparison Matrix

| Feature | tiny-stable-diffusion | Stable Diffusion 3 (Medium) | Stable Diffusion 1.5 |
| :--- | :--- | :--- | :--- |
| **Resolution** | **64 × 64** | 1024 × 1024 | 512 × 512 |
| **Architecture** | **MMDiT** (Transformer) | MMDiT (Transformer) | U-Net (CNN) |
| **Diffusion Type** | **Rectified Flow** (Linear) | Rectified Flow (Linear) | DDPM / DDIM (Gaussian) |
| **Latent Size** | **8 × 8 × 16** | 128 × 128 × 16 | 64 × 64 × 4 |
| **Latent Channels** | **16** | 16 | 4 |
| **Compression** | **f8** (12x) | f8 (64x effectively) | f8 (48x effectively) |
| **Text Encoder** | **CLIP ViT-B/32** | CLIP-G/14 + T5-XXL | CLIP ViT-L/14 |
| **Parameters** | **~200M** | 2B | 860M |
| **Training Data** | **~300K - 3M** | Billions | Billions |
| **Hardware Req.** | **Consumer GPU (8GB)** | A100 / H100 Cluster | A100 Cluster |

---

## Detailed Differences

### 1. Architecture: Transformer vs. U-Net

*   **SD 1.5 (U-Net)**: Uses a convolutional U-Net with Downblocks and Upblocks. Inductive bias is strong for images but harder to scale.
*   **tiny-sd (MMDiT)**: Uses the **same architecture as SD3**. It splits the image into patches (like ViT) and processes them with Transformers. This scales better and handles multi-modal (text/image) data more naturally via Joint Attention.

### 2. Diffusion Process: Rectified Flow vs. DDPM

*   **SD 1.5 (DDPM)**: Traverses a curved path from noise to data. Requires complex noise schedules (linear, cosine).
*   **tiny-sd / SD3 (Rectified Flow)**: Traverses a **straight line** path.
    *   $X_t = (1-t)X_0 + tX_1$
    *   Easier to simulate (ODE solvers work faster).
    *   Predicts **velocity** ($v = X_1 - X_0$) instead of noise ($\epsilon$).

### 3. Latent Space: 16 Channels vs 4 Channels

*   **SD 1.5**: Uses a 4-channel latent space. This is compact but can lose some high-frequency detail.
*   **tiny-sd / SD3**: Uses a **16-channel** latent space.
    *   Even though our resolution is small (64x64), we keep the 16 channels to faithfully mimic the SD3 design philosophy.
    *   This allows the VAE to encode more semantic information, relieving the burden on the diffusion model.

### 4. Text Encoder

*   **SD3**: Uses a massive ensemble of CLIP-G, CLIP-L, and T5-XXL (Billions of params just for text!).
*   **tiny-sd**: Uses a single **CLIP ViT-B/32** (~123M params).
    *   Reason: To fit on consumer GPUs.
    *   Trade-off: Complex prompt understanding (e.g., "a red cube on top of a blue cylinder") is weaker.

---

## Why "Tiny"?

The goal of this project is **education and accessibility**, not state-of-the-art generation.

*   **Trainable in hours**: You can train this from scratch on Colab free tier or a single gaming GPU.
*   **Readable Code**: No abstraction hell. You can read `models/mmdit.py` and understand SD3 in 10 minutes.
*   **Hackable**: Want to try a new attention mechanism? A new loss function? It's much easier to modify a 200M model than a 2B one.

---

## Performance Expectations

*   **tiny-sd will NOT** generate photorealistic faces or complex scenes.
*   **tiny-sd WILL** generate coherent objects (cats, dogs, simple shapes) that follow the prompt.
*   **tiny-sd WILL** demonstrate the **exact training dynamics** of large foundation models (loss curves, CFG behavior, latent structure).
