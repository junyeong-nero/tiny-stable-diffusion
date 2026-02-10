# Training Quick Start Guide

> From zero to generative model in a few steps.

---

## 🚀 Environment Setup

We recommend using `uv` for lightning-fast dependency management.

```bash
# 1. Install dependencies
uv sync

# 2. Run the automated setup script
bash scripts/setup.sh
```

---

## 🏗 Stage 1: VAE Training

The VAE must be trained first to establish the latent space.

```bash
# Train VAE for 100 epochs
uv run main.py --train-vae --epochs 100 --batch-size 128
```

- **Logging**: Add `--wandb` to monitor reconstruction loss and latent stats.
- **Verification**: Check `samples/vae_epoch_N/` to see how reconstruction improves over time.

---

## 🔥 Stage 2: Diffusion Training

Once you have a pre-trained VAE, you can train the Diffusion model.

```bash
# Train Diffusion using the base MMDiT model
uv run main.py --train-diffusion --epochs 200 --batch-size 32
```

- **Requirements**: Ensure `checkpoints/vae.pt` exists or specify it with `--vae-checkpoint`.
- **Validation**: The model will generate sample images every 10 epochs in `samples/epoch_N/`.

---

## 🖥 Hardware Requirements

| Stage | Model Size | Recommended VRAM | Est. Training Time |
| :--- | :--- | :--- | :--- |
| **VAE** | - | 8GB+ | 1–2 Hours |
| **Diffusion** | Small (S) | 8GB+ | 2–4 Hours |
| **Diffusion** | Base (B) | 12GB+ | 6–10 Hours |

*Estimates based on a single modern gaming GPU (e.g., RTX 3080/4080).*

---

## 🛠 Common Training Flags

| Flag | Description |
| :--- | :--- |
| `--resume` | Resume training from the latest checkpoint. |
| `--mixed-precision` | Use `fp16` to speed up training and save VRAM. |
| `--wandb` | Enable Weights & Biases logging. |
| `--push-to-hub` | Automatically upload checkpoints to Hugging Face. |

---

## 📝 Troubleshooting

- **Out of Memory (OOM)**: Reduce `--batch-size` or use a smaller `model_size` (S instead of B).
- **Blurry Samples**: Usually means the VAE hasn't trained long enough. Aim for at least 50 epochs on a diverse dataset like LAION.
- **Prompt Ignored**: Increase the `guidance_scale` during inference or check if `cfg_warmup_epochs` has completed during training.