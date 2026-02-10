# Training Pipeline Deep Dive

> A comprehensive look at the "how" and "why" behind our training methodology.

---

## 🏛 The Two-Stage Philosophy

`tiny-stable-diffusion` follows the standard **Latent Diffusion Model (LDM)** strategy, splitting training into two independent stages.

| Stage | Goal | Active Component | Weights |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Learn Image Compression | VAE | Trainable |
| **Stage 2** | Learn Text-to-Image Mapping | MMDiT | Trainable (VAE Frozen) |

### Why separate them?
1.  **Efficiency**: Diffusion in pixel space is extremely slow. By compressing images first, we reduce the computational workload by over 12x.
2.  **Stability**: Training a VAE to reconstruct pixels is a well-understood task. Once the latent space is "stable," training the Diffusion model becomes much more predictable.

---

## 🏗 Stage 1: VAE Training

**Core Objective**: Minimize Reconstruction Error while keeping Latent Space regularized.

### The Loss Function
$$L = L_{pixel} + \beta L_{KL}$$
- **$L_{pixel}$ (MSE)**: Forces the model to preserve fine details.
- **$L_{KL}$**: Prevents the latent space from becoming too sparse, allowing for smoother generation. We use a very low $\beta$ ($10^{-6}$) to prioritize detail.

### Optimization Strategy
- **AdamW Optimizer**: Standard choice for high-stability training.
- **Cosine Annealing**: Gradually reduces the learning rate to settle into a local minimum for high-quality reconstruction.

---

## 🔥 Stage 2: Diffusion Training

**Core Objective**: Learn to predict the velocity vector that reverses noise into a clean latent.

### Key Innovations
1.  **Rectified Flow**: We train the model on a linear path between noise and data.
2.  **Min-SNR Weighting**: We weigh the loss based on the Signal-to-Noise Ratio (SNR) of each timestep. This ensures the model focuses on the most "difficult" parts of the denoising process.
3.  **CFG Dropout**: During training, we randomly drop the text prompt (10% of the time). This forces the model to learn an "unconditional" generation path, which is essential for Classifier-Free Guidance during inference.

### Exponential Moving Average (EMA)
We maintain a "shadow" copy of the MMDiT weights that updates very slowly (decay rate of $0.9999$).
- **Benefit**: The EMA weights represent a temporal average of the model's parameters, which significantly reduces noise and artifacts in the generated images. Always use EMA weights for final inference.

---

## 📦 Checkpoint & Hub Integration

The pipeline is designed to be fully automated.
- **Best Model Saving**: We automatically track the validation loss and save the `best` weights to `checkpoints/`.
- **Hugging Face Hub**: Use the `--push-to-hub` flag to automatically version your models on the HF Hub, making them easily shareable.

---

## 🧪 Best Practices for Fine-Tuning

1.  **Start Small**: Use the `S` (Small) model to quickly verify your dataset and hyperparameters.
2.  **Monitor Latent Stats**: If your VAE's latent mean drifts too far from $0$, or variance from $1$, your Diffusion model will struggle.
3.  **Check Validation Samples**: Don't just trust the loss curve. Visual inspection of the `samples/` folder is the most reliable way to judge generation progress.