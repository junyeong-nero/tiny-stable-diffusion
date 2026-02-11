# Evaluation & Metrics

> How to quantitatively assess the performance of your VAE and Diffusion models.

---

## 📐 VAE Evaluation (Reconstruction)

VAE evaluation focuses on **fidelity**: how well can the model reconstruct an image after compressing it to the latent space?

### Key Metrics

| Metric | Full Name | Target | Description |
| :--- | :--- | :--- | :--- |
| **PSNR** | Peak Signal-to-Noise Ratio | Higher (>30dB) | Measures pixel-level accuracy. |
| **SSIM** | Structural Similarity Index | Higher (>0.9) | Measures structural/perceptual similarity. |
| **MSE** | Mean Squared Error | Lower | Average squared difference between pixels. |
| **LPIPS** | Learned Perceptual Similarity | Lower | Measures "human-like" perceptual distance. |

### Running VAE Eval
```bash
# Evaluate on a local folder of images
./scripts/evaluate-vae.sh --input-dir assets/samples --checkpoint checkpoints/vae.pt

# Evaluate on a HuggingFace dataset
./scripts/evaluate-vae.sh --dataset reach-vb/pokemon-blip-captions --max-samples 200
```

---

## 🎨 Diffusion Evaluation (Generation)

Diffusion evaluation focuses on **Quality**, **Diversity**, and **Alignment**.

### Key Metrics

| Metric | Full Name | Target | Description |
| :--- | :--- | :--- | :--- |
| **FID** | Fréchet Inception Distance | Lower | Distance between real and generated distributions. |
| **CLIP-FID** | CLIP-based FID | Lower | More sensitive to semantic quality at low resolutions. |
| **CLIPScore** | CLIP Text-Image Score | Higher | Measures how well the image matches the prompt. |
| **IS** | Inception Score | Higher | Measures both clarity and diversity. |

### Running Diffusion Eval
```bash
# Calculate FID (requires a folder of real images)
./scripts/evaluate-diffusion.sh --real-images-dir data/real_images --num-samples 1000

# Calculate CLIPScore only
./scripts/evaluate-diffusion.sh --metrics clip_score --num-samples 100
```

---

## ⚡ Benchmarking (Speed & Memory)

Performance metrics are critical for deployment on consumer hardware.

*   **Throughput**: Images generated per second.
*   **Latency**: Time taken for each stage (Text Encode, Denoise, VAE Decode).
*   **Peak VRAM**: Maximum GPU memory consumed during a single inference pass.

### Running Benchmarks
```bash
# Run a standard benchmark
uv run main.py --benchmark

# Sweep across batch sizes and steps
uv run main.py --benchmark --benchmark-steps "10,25,50" --benchmark-batch "1,4"
```

---

## 📊 Reference Implementation

- **Metrics Logic**: `src/evaluation/metrics.py`
- **VAE Evaluator**: `src/evaluation/vae_evaluator.py`
- **Diffusion Evaluator**: `src/evaluation/diffusion_evaluator.py`
- **Benchmark Suite**: `src/evaluation/benchmark.py`
