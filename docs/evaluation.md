# 📊 Evaluation Results

> Consolidated performance metrics for VAE and Diffusion checkpoints. Last updated: **2026-02-11**.

---

## 📐 VAE Performance (Latent Fidelity)

The VAE is evaluated on its ability to reconstruct original pixels after an 8x spatial compression.

### Test Set: `assets/samples` (100 images)
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **PSNR** | **37.99 dB** | Excellent pixel-level fidelity. |
| **SSIM** | **0.9784** | Strong structural similarity. |
| **MSE** | **0.000195** | Negligible reconstruction error. |
| **LPIPS** | **0.0094** | Minimal perceptual distance (Very High Quality). |

### Dataset: `Pokemon BLIP` (833 samples)
*Cross-domain evaluation to test generalization.*
| Metric | Value | Status |
| :--- | :--- | :--- |
| **PSNR** | **31.60 dB** | Good |
| **SSIM** | **0.9679** | Strong |
| **LPIPS** | **0.0146** | High Fidelity |

---

## 🎨 Diffusion Performance (Generative Quality)

Metrics computed using `1,000` generated samples vs. `1,000` real samples.

| Metric | Value | Context (64x64 scale) |
| :--- | :--- | :--- |
| **FID** ↓ | **96.57** | Expected for low-res experimental models. |
| **CLIP-FID** ↓ | **19.29** | Reasonable alignment for this scale. |
| **Inception Score** ↑ | **7.73** | Moderate variety in generated objects. |
| **CLIP Score** ↑ | **24.83** | Decent text-image alignment. |

### 📈 Metric Interpretation Guide

| Metric | Direction | What it measures |
| :--- | :--- | :--- |
| **FID (Fréchet Inception Distance)** | Lower is better | Distribution similarity between real and fake images. |
| **Inception Score (IS)** | Higher is better | Clarity and diversity of generated objects. |
| **CLIP Score** | Higher is better | How well the image matches the text prompt. |
| **LPIPS** | Lower is better | Perceptual difference (matches human vision). |

---

## ⚖️ Benchmarks (Inference Speed)

**Environment**: M2 MacBook Air (`device=mps`)

| Task | Configuration | Performance |
| :--- | :--- | :--- |
| **Single Generation** | 10 steps | **~1.55 sec** |
| **Peak VRAM** | Base Model (B) | **~1.5 GB** |
| **Throughput** | Batch size 4 | ~2.8 images/sec |

---

## 🛠 Reproduction Commands

To run these evaluations on your own hardware:

```bash
# VAE Evaluation
bash scripts/evaluate-vae.sh --input-dir assets/samples --checkpoint checkpoints/vae.pt

# Diffusion Evaluation
uv run python src/evaluation/diffusion_evaluator.py \
    --checkpoint checkpoints/diffusion.pt \
    --num-samples 1000
```

---
*Detailed logic found in `src/evaluation/metrics.py`.*