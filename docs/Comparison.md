# ⚖️ Comparison: tiny vs. SOTA

> How `tiny-stable-diffusion` stacks up against industry standards like SD3 and SD 1.5.

---

## 🔍 Feature Comparison Matrix

| Feature | tiny-stable-diffusion | Stable Diffusion 3 (Med) | Stable Diffusion 1.5 |
| :--- | :--- | :--- | :--- |
| **Architecture** | **MMDiT** (Transformer) | MMDiT (Transformer) | U-Net (CNN) |
| **Framework** | **Rectified Flow** | Rectified Flow | DDPM / DDIM |
| **Resolution** | **64 × 64** | 1024 × 1024 | 512 × 512 |
| **Latent Space** | **8 × 8 × 16** | 128 × 128 × 16 | 64 × 64 × 4 |
| **Parameters** | **~200M** (Backbone) | 2,000M (2B) | 860M |
| **Hardware** | **Consumer GPU / Laptop** | A100 / H100 Clusters | Mid-range GPU |

---

## 💡 Key Differentiators

### 1. Transformer-Centric Design (MMDiT)
Unlike SD 1.5 which uses a convolutional U-Net, this project adopts the **MMDiT (Multi-Modal Diffusion Transformer)**. 
- **The Patch Advantage**: We treat images as sequences of patches (tokens). This allows the model to leverage the same scaling laws that made LLMs successful.
- **Joint Attention**: Both text and image tokens inhabit the same Transformer blocks, allowing for bidirectional information flow.

### 2. Rectified Flow vs. Gaussian Diffusion
We use **Rectified Flow**, which is the current state-of-the-art for diffusion efficiency.
- **Straight Paths**: While SD 1.5 follows a curved stochastic path, we learn a **straight-line deterministic path** between noise and data.
- **Velocity Prediction**: Instead of predicting "noise" to be subtracted, we predict the "velocity" required to reach the target image. This leads to faster convergence with fewer sampling steps.

### 3. High-Dimensional Latents (16 Channels)
Standard VAEs (like SD 1.5) use 4 latent channels. Following SD3, we use **16 channels**.
- **Information Density**: Even at $64 \times 64$, the 16-channel bottleneck allows the VAE to capture rich semantic details, reducing the workload on the Diffusion model.

---

## 🎯 Target Audience & Goals

This project is NOT intended to replace production-grade models. Instead, it serves as a **"Miniature Lab"**:

1.  **Educational Transparency**: The entire pipeline fits in your head. You can trace a single pixel from the prompt to the final output.
2.  **Rapid Prototyping**: Test new ideas (like custom attention or loss functions) in hours, not weeks.
3.  **Hardware Accessibility**: If you can run a modern web browser, you can likely train or run inference on this model.

---

## 📉 Expectations

*   **Capabilities**: Generates coherent animals, simple objects, and abstract scenes. Excellent at demonstrating prompt alignment.
*   **Limitations**: No photorealistic human faces or complex text rendering. It is a "coarse-to-fine" learning demonstration at low resolution.

---
*For a deep dive into the math, see [Diffusion Process](./models/Diffusion.md).*
