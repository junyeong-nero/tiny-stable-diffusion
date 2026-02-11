# Scripts

Utility scripts for training, inference, evaluation, and model distribution.

## Training

- `scripts/train-vae.sh`: Train VAE (`--train-vae`)
- `scripts/train-diffusion.sh`: Train diffusion model (`--train-diffusion`)

All training scripts forward extra CLI arguments to `main.py`.

## Inference / Evaluation

- `scripts/inference-vae.sh`: Reconstruct image(s) with VAE and save reconstruction metrics (PSNR/SSIM/MSE/LPIPS)
- `scripts/inference-diffusion.sh`: Generate image(s) from text prompt
- `scripts/measure-inference.sh`: Measure diffusion inference latency, accelerator memory, and RAM (repeated runs)
- `scripts/evaluate-vae.sh`: Evaluate VAE reconstruction quality

Default outputs are saved under `results/`:
- VAE: `results/inference/vae/`
- Diffusion: `results/inference/diffusion/`

## Inference Profiling

Use `scripts/measure-inference.sh` to measure end-to-end inference latency, accelerator memory (CUDA/MPS), and process RAM.

Examples:

```bash
# Default profile
./scripts/measure-inference.sh

# Custom run
./scripts/measure-inference.sh \
  --checkpoint checkpoints/diffusion.pt \
  --vae-checkpoint checkpoints/vae.pt \
  --prompt "a photo of a cat" \
  --steps 25 \
  --batch-size 1 \
  --repeats 3 \
  --warmup-runs 2
```

Saved JSON:
- Default path: `results/benchmarks/inference_profile.json`
- Includes: `latency_mean_ms`, `latency_p50_ms`, `latency_p95_ms`, `sec_per_image_mean`, `sec_per_image_p50`, `sec_per_image_p95`, `peak_vram_max_mb`, `peak_ram_max_mb`, `ram_delta_max_mb`, and per-run breakdown in `runs`.

Latest measured samples (prompt=`a photo of a cat`):

| Device | Steps | Batch | Repeats | Latency Mean (ms) | Speed (sec/img) | Peak Accel Mem (primary, MB) | Peak Reserved (MB) | Peak RAM (MB) | RAM Delta (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MPS | 10 | 1 | 2 | 1553.15 | 1.553 | 1504.99 | 3715.97 | 119.36 | 81.38 |
| CPU | 5 | 1 | 1 | 981.36 | 0.981 | 0.00 | 0.00 | 1098.06 | 1083.66 |

Measured JSON files:
- `results/benchmarks/inference_profile_mps.json`
- `results/benchmarks/inference_profile_cpu.json`

Notes:
- `Peak Accel Mem`/`Peak Reserved` map to `peak_vram_max_mb`/`peak_reserved_max_mb` in JSON.
- On CPU, accelerator memory is `0.0` by design; use `peak_ram_max_mb` and `ram_delta_max_mb` for memory analysis.
- On MPS/CUDA, use `peak_vram_max_mb` (and optionally `peak_reserved_max_mb`) as the primary inference memory indicator. `ram_delta_max_mb` is only the additional process RAM during the measured section.

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
