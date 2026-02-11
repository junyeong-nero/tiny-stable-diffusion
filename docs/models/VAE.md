# VAE (Variational AutoEncoder)

> The bridge between pixel space and latent space.

---

## 🔬 Overview

The VAE is the first stage of the `tiny-stable-diffusion` pipeline. It is responsible for compressing $64 \times 64$ RGB images into a compact $8 \times 8 \times 16$ latent representation. This compression is critical because it allows the Diffusion model to operate on a much lower-dimensional space, saving massive amounts of compute.

### Key Specs

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Input Resolution** | $64 \times 64 \times 3$ | RGB Pixels |
| **Latent Resolution** | $8 \times 8 \times 16$ | 16-channel Latents |
| **Compression Factor** | **f8** | 8x spatial reduction |
| **Parameters** | ~21M | Optimized for consumer GPUs |

---

## 🏗 Architecture

The VAE follows the **AutoencoderKL** design used in SD3.

### 1. Encoder
- **Downsampling**: Uses 3 stages of strided convolutions to reduce resolution from $64 \rightarrow 32 \rightarrow 16 \rightarrow 8$.
- **Residual Blocks**: Each stage uses ResNet-style blocks with GroupNorm and SiLU activation.
- **Attention**: The bottleneck stage ($8 \times 8$) uses self-attention to capture global image context.
- **Output**: Produces a mean ($\mu$) and log-variance ($\log \sigma^2$) for the latent distribution.

### 2. Reparameterization
To make the sampling process differentiable, we use the reparameterization trick:
$$z = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)$$

### 3. Decoder
- **Upsampling**: Reverses the encoder's path using nearest-neighbor upsampling followed by convolutions.
- **Goal**: Reconstruct the original $64 \times 64$ pixels from the sampled latent $z$.

---

## 📉 Loss Function

The VAE is trained using two combined loss terms:

1.  **Reconstruction Loss (MSE)**: Minimizes the pixel-wise difference between the input image and the reconstructed output.
2.  **KL Divergence Loss**: Regularizes the latent space to be close to a standard normal distribution.
    - **Note**: We use a very small KL weight ($10^{-6}$), prioritizing reconstruction quality for the diffusion process.

---

## ⚙️ Configuration

Key settings in `config.yaml`:
```yaml
vae_train:
    latent_channels: 16
    vae_ch: 64                # Base channel count
    vae_ch_mult: [1, 2, 4, 4] # Channel multipliers for f8 compression
    kl_weight: 0.000001
```

---

## 🛠 Usage

### Reconstructing an Image
```bash
uv run main.py --reconstruct-vae \
    --input "input.png" \
    --output "reconstruction.png" \
    --vae-checkpoint "checkpoints/vae.pt"
```

### Evaluating Performance
```bash
./scripts/evaluate-vae.sh --input-dir assets/samples
```

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

## 📚 Implementation Reference
- **Model Definition**: `src/models/vae.py`
- **Trainer**: `src/training/vae_trainer.py`
- **Inference**: `src/inference/vae_inference.py`
