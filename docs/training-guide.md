# 🛠 Training Quick Start

> A step-by-step guide to training your own `tiny-stable-diffusion` model from scratch.

---

## 🏗 Phase 0: Environment Setup

Ensure you have `uv` installed and the environment initialized.

```bash
# 1. Initialize environment
bash scripts/setup.sh

# 2. Verify installation
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}')"
```

---

## 🎨 Phase 1: VAE Training (Stage 1)

The VAE must be trained first to create the latent space bottleneck ($64 \rightarrow 8$).

```bash
# Basic training run
uv run main.py --train-vae --epochs 50 --batch-size 128

# Advanced: With logging and mixed precision
uv run main.py --train-vae --wandb --mixed-precision fp16
```

- **Goal**: Low MSE and LPIPS scores.
- **Check**: Look at `samples/vae_epoch_N/` to see reconstruction progress.

---

## 🔥 Phase 2: Diffusion Training (Stage 2)

Once `checkpoints/vae.pt` is ready, you can train the generative MMDiT model.

```bash
# Basic training run (Base model)
uv run main.py --train-diffusion --epochs 100 --batch-size 32

# Advanced: Resume from a checkpoint and push to hub
uv run main.py --train-diffusion --resume --push-to-hub
```

- **Requirements**: A pre-trained VAE checkpoint.
- **Check**: Generation samples appear every 10 epochs in `samples/epoch_N/`.

---

## 🖥 Hardware Guidelines

Estimates for reaching "converged" quality on a single GPU.

| Setup | VRAM | Time (VAE) | Time (Diffusion) |
| :--- | :--- | :--- | :--- |
| **RTX 3060 (12GB)** | Good | ~1 Hour | ~6-8 Hours |
| **RTX 4090 (24GB)** | Excellent | ~20 Mins | ~2-3 Hours |
| **Apple M2 Max** | Good | ~40 Mins | ~5-6 Hours |
| **Colab (T4)** | Minimal | ~1.5 Hours | ~10-12 Hours |

---

## 🛠 Useful CLI Flags

| Flag | Purpose | Recommended For |
| :--- | :--- | :--- |
| `--resume` | Restart from last saved checkpoint. | Long runs. |
| `--wandb` | Real-time loss/sample tracking. | All serious training. |
| `--mixed-precision` | Faster training, less VRAM. | NVIDIA RTX GPUs (`fp16`). |
| `--model-size S` | Use the smaller 87M parameter model. | Low VRAM / Fast testing. |

---

## 🆘 Troubleshooting

- **Out of Memory (OOM)**: Reduce `--batch-size` (try 16 or 8) or use `--model-size S`.
- **Loss is NaN**: Decrease learning rate in `config.yaml` or ensure you're using `qk_rmsnorm: true`.
- **Samples are Gray**: This usually means the model hasn't learned the "Straight Line" of Rectified Flow yet. Train for at least 10-20 epochs.

---
*Deep Dive: [Training Pipeline Philosophy](./training-pipeline.md)*
