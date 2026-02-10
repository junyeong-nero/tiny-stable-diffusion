# Inference & Generation Deep Dive

> A technical walkthrough of how `tiny-stable-diffusion` transforms a text prompt into a pixel-perfect image.

---

## 🚀 The Generation Pipeline

Generating an image happens in four distinct stages. The core logic is implemented in `src/inference/generator.py`.

### Stage 1: Text Encoding (CLIP)
The input prompt (e.g., *"a cute cat"*) is tokenized and processed by a frozen **CLIP ViT-B/32** model.
- **Output**: A 512-dimensional embedding representing the prompt's semantics.
- **Unconditional Embedding**: We also encode an empty string (`""`) for **Classifier-Free Guidance (CFG)**.

### Stage 2: Latent Initialization
We start in the **latent space** ($8 \times 8 \times 16$).
- **Action**: Sample pure Gaussian noise $z_T \sim \mathcal{N}(0, I)$.
- **Reproducibility**: If a `--seed` is provided, we fix the random generator to ensure the exact same image can be recreated.

### Stage 3: Iterative Denoising (MMDiT)
This is the heart of the model. We use the **Euler ODE Solver** with **Rectified Flow**.

For each timestep $t$ from $1.0$ (pure noise) down to $0.0$ (clean latent):
1.  **Predict Conditional Velocity**: $v_{cond} = \text{MMDiT}(z_t, t, \text{text\_embed})$
2.  **Predict Unconditional Velocity**: $v_{uncond} = \text{MMDiT}(z_t, t, \text{uncond\_embed})$
3.  **Apply CFG**: $v_{pred} = v_{uncond} + s \times (v_{cond} - v_{uncond})$
4.  **Update Latent**: $z_{next} = z_t + v_{pred} \times dt$

### Stage 4: VAE Decoding
The clean latent $z_0$ is transformed back into the pixel domain.
- **Action**: Pass through the VAE Decoder.
- **Normalization**: Rescale from $[-1, 1]$ to $[0, 1]$ RGB values.
- **Output**: A $64 \times 64$ RGB image.

---

## 🛠 Sampling Details

### Rectified Flow vs. Traditional Diffusion
`tiny-stable-diffusion` (like SD3) uses **Rectified Flow**, which is mathematically simpler and more efficient than the Gaussian diffusion used in SD1.5/2.1.

| Feature | SD1.5 (DDPM/DDIM) | tiny-sd (Rectified Flow) |
| :--- | :--- | :--- |
| **Trajectory** | Curved / Stochastic | **Straight / Deterministic** |
| **Prediction Target** | Noise ($\epsilon$) | **Velocity ($v$)** |
| **Path** | $X_t = \sqrt{\bar{\alpha}_t}X_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ | $X_t = (1-t)X_0 + tX_1$ |

### Classifier-Free Guidance (CFG)
CFG is a technique to trade off diversity for prompt alignment.
- **Guidance Scale ($s$)**: A value of $1.0$ uses only the prompt. Higher values (e.g., $7.5$) "push" the generation harder toward the prompt's direction.
- **Impact**: Higher $s$ leads to more vivid colors and better alignment but can cause "oversaturation" or artifacts if set too high ($>15$).

---

## ⌨️ CLI Usage

```bash
uv run main.py --generate \
    --prompt "a fluffy orange cat on a sofa" \
    --steps 50 \
    --guidance 7.5 \
    --seed 42 \
    --output "my_cat.png"
```

### Parameter Tuning

| Parameter | Recommended | Range | Impact |
| :--- | :--- | :--- | :--- |
| **Steps** | 50 | 10–100 | Higher = more detail, slower generation. |
| **Guidance** | 7.5 | 1.0–15.0 | Higher = stronger adherence to the prompt. |
| **Seed** | - | Any Int | Fixes the random noise for reproducibility. |

---

## ⚡ Optimization Tips

- **Half Precision**: Use `fp16` or `bf16` to reduce VRAM usage by 50% on compatible GPUs.
- **Batching**: Generating multiple samples in a single batch is significantly faster than generating them sequentially.
- **MPS Support**: On macOS, use `--device mps` to leverage the Apple Silicon Neural Engine.

---

## 📚 Advanced Implementation
- **Euler Solver**: `src/inference/generator.py`
- **MMDiT Logic**: `src/models/mmdit.py`
- **VAE Decoding**: `src/models/vae.py`