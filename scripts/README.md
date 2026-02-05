# Scripts

Utility scripts for training, inference, evaluation, and model distribution.

## Training

- `scripts/train-vae.sh`: Train VAE (`--train-vae`)
- `scripts/train-diffusion.sh`: Train diffusion model (`--train-diffusion`)
- `scripts/train-motion.sh`: Train motion module (`--train-motion`)

All training scripts forward extra CLI arguments to `main.py`.

## Inference / Evaluation

- `scripts/inference-vae.sh`: Reconstruct image(s) with VAE and save reconstruction metrics (PSNR/SSIM/MSE/LPIPS)
- `scripts/inference-diffusion.sh`: Generate image(s) from text prompt
- `scripts/evaluate-vae.sh`: Evaluate VAE reconstruction quality

Default outputs are saved under `results/`:
- VAE: `results/vae/`
- Diffusion: `results/diffusion/`

## HuggingFace Hub

- `scripts/upload_to_hub.py`: Upload checkpoints to Hub
- `scripts/download_from_hub.py`: Download checkpoints from Hub
- `scripts/hf.sh`: Thin wrapper around upload/download scripts

## Environment / Remote Sync

- `scripts/setup.sh`: Install tooling and project dependencies
- `scripts/download.sh`: Pull/push checkpoints and samples via `scp`

## Notes

- Most shell scripts use environment variable overrides, e.g.:
  - `CHECKPOINT=checkpoints/vae.pt ./scripts/train-vae.sh`
  - `NUM_SAMPLES=4 ./scripts/inference-diffusion.sh "a cat"`
- For multiple prompts in `main.py`, use `||` separator:
  - `--prompt "a cat||a dog"`
