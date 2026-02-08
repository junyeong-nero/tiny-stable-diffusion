# Evaluation & Metrics

> This document explains how to quantitatively evaluate the performance of `tiny-stable-diffusion` and describes the metrics used for assessment.

## Table of Contents

1. [VAE Evaluation (Reconstruction)](#vae-evaluation-reconstruction)
2. [Diffusion Evaluation (Generation)](#diffusion-evaluation-generation)
3. [Benchmarking (Speed & Memory)](#benchmarking-speed--memory)
4. [Available Datasets](#available-datasets)

---

## VAE Evaluation (Reconstruction)

VAE evaluation measures how accurately the model compresses an input image into the latent space and reconstructs it back to pixels.

### Key Metrics

| Metric | Description | Goal |
|------|------|------|
| **PSNR** | Peak Signal-to-Noise Ratio. Measures pixel-level reconstruction accuracy. | Higher is better (typically > 30dB) |
| **SSIM** | Structural Similarity Index. Measures structural similarity of images. | Higher is better (range 0-1) |
| **MSE** | Mean Squared Error. Average squared difference between original and reconstructed pixels. | Lower is better |
| **LPIPS** | Learned Perceptual Image Patch Similarity. Measures human-like perceptual similarity. | Lower is better |

### Usage

```bash
# Evaluate using a local image directory
./scripts/evaluate-vae.sh --input-dir samples/original

# Evaluate on a HuggingFace dataset
./scripts/evaluate-vae.sh --dataset reach-vb/pokemon-blip-captions --max-samples 200

# Use a specific checkpoint and save results
./scripts/evaluate-vae.sh --input-dir samples/original \
    --checkpoint checkpoints/vae.pt \
    --save results/vae_eval.json
```

---

## Diffusion Evaluation (Generation)

Diffusion model evaluation measures the quality, diversity, and text-alignment of the generated images.

### Key Metrics

| Metric | Description | Feature |
|------|------|------|
| **FID** | Fréchet Inception Distance. Distance between real and generated image distributions. | Lower is better. Based on InceptionV3. |
| **CLIP-FID** | FID calculated using CLIP feature vectors. | More reliable for 64x64 low-resolution images. |
| **IS** | Inception Score. Measures clarity and diversity of generated images. | Higher is better. |
| **CLIPScore** | Semantic similarity between text prompts and generated images. | Higher is better. Key for measuring Alignment. |

### Usage

```bash
# Full evaluation (requires real image directory for FID)
./scripts/evaluate-diffusion.sh --real-images-dir data/real_images --num-samples 1000

# Evaluate CLIPScore only (no real images required)
./scripts/evaluate-diffusion.sh --metrics clip_score --num-samples 100

# Use dataset captions as prompts to calculate FID
./scripts/evaluate-diffusion.sh --dataset visual-layer/oxford-iiit-pet-vl-enriched --num-samples 500
```

---

## Benchmarking (Speed & Memory)

Measures the inference speed and hardware resource consumption of the models.

### Metrics

*   **Throughput**: Number of images generated per second (Images/sec).
*   **Latency**: Time taken for each stage (Text Encoding, Sampling, VAE Decoding).
*   **Peak VRAM**: Maximum GPU memory usage (MB).

### Usage

```bash
# Run basic benchmark
uv run main.py --benchmark

# Run sweep test for various batch sizes and step counts
uv run main.py --benchmark --benchmark-steps "10,25,50" --benchmark-batch "1,4,8"
```

---

## Available Datasets

Commonly used datasets for evaluation.

| Dataset ID | Purpose | Description |
|-------------|------|------|
| `hmu013/LAION-300k` | VAE | Ideal for evaluating general image reconstruction. |
| `visual-layer/oxford-iiit-pet-vl-enriched` | Diffusion | Evaluating generation and text alignment in the pet domain. |
| `reach-vb/pokemon-blip-captions` | Debugging | Small dataset for quick verification. |

---

## Reference Code

*   **Metrics Implementation**: `src/evaluation/metrics.py`
*   **VAE Evaluator**: `src/evaluation/vae_evaluator.py`
*   **Diffusion Evaluator**: `src/evaluation/diffusion_evaluator.py`
*   **Benchmark Tool**: `src/evaluation/benchmark.py`