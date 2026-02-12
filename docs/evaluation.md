# Evaluation Results

> Consolidated latest evaluation results for VAE and Diffusion checkpoints.

---

## 📐 VAE Evaluation (checkpoints/vae.pt)

### Latest Evaluation Run

```bash
./scripts/evaluate-vae.sh \
    --input-dir assets/samples \
    --checkpoint checkpoints/vae.pt \
    --max-samples 100
```

**Run Context**
- Date: `2026-02-11`
- Checkpoint: `checkpoints/vae.pt` (epoch 39)
- Input set: `assets/samples`
- Evaluated samples: `100`
- LPIPS: Computed (`lpips==0.1.4`)

**Measured Reconstruction Metrics**

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **PSNR** | **37.99 dB** | Strong pixel-level fidelity (high) |
| **SSIM** | **0.9784** | Strong structural similarity (high) |
| **MSE** | **0.000195** | Low reconstruction error |
| **LPIPS** | **0.0094** | Very low perceptual distance |

Overall, this checkpoint shows high-fidelity reconstruction quality for 64x64 inputs and is suitable as the latent encoder/decoder stage for diffusion training and inference.

### Additional Dataset Evaluation (`--max-samples 1000`)

```bash
./scripts/evaluate-vae.sh \
    --dataset reach-vb/pokemon-blip-captions \
    --checkpoint checkpoints/vae.pt \
    --max-samples 1000
```

**Run Context**
- Date: `2026-02-11`
- Checkpoint: `checkpoints/vae.pt` (epoch 39)
- Dataset: `reach-vb/pokemon-blip-captions` (`split=train`, `image_field=image`)
- Requested samples: `1000`
- Evaluated samples: `833` (dataset availability limit during run)

**Measured Reconstruction Metrics (833 samples)**

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **PSNR** | **31.60 dB** | Good pixel-level fidelity |
| **SSIM** | **0.9679** | Strong structural similarity |
| **MSE** | **0.000770** | Low reconstruction error |
| **LPIPS** | **0.0146** | Low perceptual distance |

---

## 🎨 Diffusion Evaluation (checkpoints/diffusion.pt)

From `results/evaluation/diffusion/eval_results.json`:

| Metric | Value |
|---|---:|
| FID | 96.5757 |
| CLIP-FID | 19.2907 |
| Inception Score (mean) | 7.7384 |
| Inception Score (std) | 0.5936 |
| CLIP Score (mean) | 24.8311 |
| CLIP Score (std) | 0.0000 |
| Generated samples | 1000 |
| Real samples | 1000 |

> Note: these metrics were computed with `num_generated=1000`, `num_real=1000`, which is more stable than tiny-sample checks but can still vary depending on prompt set, seed, and evaluation protocol.

## 📈 Diffusion Metric Interpretation

The values above are reasonable for an educational 64x64 model. They should not be directly compared to large-scale 512/1024-resolution production models without protocol matching.

### Typical Ranges (Rule of Thumb)

| Metric | Typical range in practice | Direction |
|---|---:|---|
| FID | SOTA text-to-image on COCO: ~7-12; small/low-res experimental models: ~50-200 | Lower is better |
| CLIP-FID | High-quality models often land in low double digits or below; small/low-res models commonly ~15-30+ | Lower is better |
| Inception Score | Highly protocol-dependent; small text-to-image experiments often ~5-15 | Higher is better |
| CLIP Score (`100 * cosine` scale) | ~23-26: decent alignment, ~27-31: strong alignment, ~32+: very strong | Higher is better |

### Current Checkpoint in Context

- `FID=96.5757`: within a typical low-resolution experimental band.
- `CLIP-FID=19.2907`: reasonable for this project scale and resolution.
- `IS=7.7384`: moderate and plausible for a compact 64x64 setup.
- `CLIP Score=24.8311`: prompt alignment is present, roughly in a "base model" quality zone.

### Comparison Caveats (Important)

- **Metric pipeline differences matter**: this project upsamples to 299x299 for Inception-based metrics and uses CLIP ViT-B/32 for CLIP-based metrics.
- **CLIP-FID is less standardized** than vanilla FID across papers/repositories.
- **Only compare scores under matched protocol**: same prompt set, sample count, preprocessing/resizing, metric implementation, and backbone.

Reference points used for interpretation include Imagen/GLIDE/DALL-E2 reports, clean-fid guidance, and TorchMetrics metric definitions.

---

## 📚 Reference Implementation

- **Metrics Logic**: `src/evaluation/metrics.py`
- **VAE Evaluator**: `src/evaluation/vae_evaluator.py`
- **Diffusion Evaluator**: `src/evaluation/diffusion_evaluator.py`
- **Benchmark Suite**: `src/evaluation/benchmark.py`
