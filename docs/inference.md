# 🚀 Inference & Generation Deep Dive

> Technical walkthrough of the journey from a text prompt to a $64 \times 64$ RGB image.

---

## 🛠 The Generation Pipeline

The generation process is a deterministic mapping from a text string to a pixel grid, managed by `src/inference/generator.py`.

### 1. Semantic Encoding (CLIP)
The input prompt (e.g., *"a golden retriever in a field"*) is tokenized and passed through the **Frozen CLIP Text Encoder**.
- **Conditional Embed ($c$+)**: Represents the prompt's meaning.
- **Unconditional Embed ($c$-)**: Represents an empty prompt (`""`), used as a baseline for **Classifier-Free Guidance (CFG)**.

### 2. Latent Initialization
We begin in the $8 \times 8 \times 16$ latent domain.
- **Noise Sampling**: Sample $z_1 \sim \mathcal{N}(0, I)$.
- **Determinism**: By providing a fixed `--seed`, the initial noise state becomes reproducible, ensuring identical outputs for the same parameters.

### 3. Iterative Denoising (The ODE Solver)
We solve the Rectified Flow equation using an **Euler ODE Solver**. For $N$ steps (default 50), the model iterates from $t=1.0$ (noise) to $t=0.0$ (clean):

1.  **Velocity Estimation**:
    - $v_{pos} = \text{MMDiT}(z_t, t, c_+)$
    - $v_{neg} = \text{MMDiT}(z_t, t, c_-)$
2.  **CFG Application**:
    $v_{final} = v_{neg} + \text{scale} \times (v_{pos} - v_{neg})$
3.  **Step Update**:
    $z_{t-dt} = z_t - (v_{final} \times dt)$

### 4. VAE Reconstruction
The final latent $z_0$ is mapped back to pixel space.
- **Decoding**: The VAE Decoder expands the $8 \times 8 \times 16$ latent to a $64 \times 64 \times 3$ grid.
- **Post-processing**: Values are clamped and rescaled to $[0, 255]$ for standard image formats.

---

## ⌨️ CLI Parameters & Tuning

```bash
uv run main.py --generate \
    --prompt "a spaceship landing on mars" \
    --steps 50 \
    --guidance 7.5 \
    --seed 42
```

### 💡 Tuning Guide

| Parameter | Default | Range | Impact |
| :--- | :--- | :--- | :--- |
| **Steps** | 50 | 20 – 100 | **Quality vs. Speed**. 20-30 steps are often sufficient for Rectified Flow. |
| **Guidance** | 7.5 | 1.0 – 15.0 | **Prompt Adherence**. Higher values increase contrast and alignment but may introduce artifacts. |
| **Seed** | Random | Integer | **Variability**. Change the seed to get a different composition for the same prompt. |

---

## ⚡ Performance Optimization

- **Precision**: Running in `fp16` or `bf16` significantly reduces VRAM usage and speeds up inference on modern GPUs (RTX 30+ / Apple Silicon).
- **Device Support**: 
    - Linux/Windows: `--device cuda`
    - macOS: `--device mps`
- **Batching**: You can generate multiple images simultaneously to maximize throughput by increasing the batch size in the generator config.

---

## 🔍 Common Issues (FAQ)

**Q: The generated image looks like colorful static.**
- *Check your checkpoint paths. This usually happens when the VAE or Diffusion weights are not loaded correctly.*

**Q: The image is blurry or lacks detail.**
- *Increase the number of steps (e.g., to 100) or check if the VAE has been fully trained (Stage 1).*

**Q: The prompt is being ignored.**
- *Increase the `guidance_scale` (CFG). Values between 7.5 and 10.0 are typically the "sweet spot".*

---
*Reference Implementation: `src/inference/generator.py`*
