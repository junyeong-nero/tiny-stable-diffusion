# VAE (Variational Autoencoder) Evaluation Report

This document summarizes the reconstruction quality of the VAE model at different training epochs.

## Model Architecture

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| Type | AutoencoderKL | SD3-style VAE |
| Latent Channels | 16 | SD3 uses 16 channels |
| Base Channels | 64 | Lightweight for 64x64 |
| Channel Multipliers | [1, 2, 4, 4] | f8 compression |
| Input Size | 64x64 RGB | - |
| Latent Size | 8x8x16 | 1,024 values |

## Training Configuration

```yaml
# Dataset
data_source: streaming_caption
dataset_name: hmu013/LAION-300k
image_size: 64

# Architecture
latent_channels: 16
vae_ch: 64
vae_ch_mult: [1, 2, 4, 4]

# Training
epochs: 100
batch_size: 256
learning_rate: 4.0e-4

# KL Weight (Very Important!)
kl_weight: 0.000001  # Extremely low for reconstruction-focused training

# KL Annealing
kl_annealing: cyclical
kl_n_cycles: 4
kl_cycle_ratio: 0.5
```

### Why Low KL Weight?

Stable Diffusion VAE는 **reconstruction 성능에 초점**을 맞추기 위해 매우 낮은 KL weight를 사용합니다.

| KL Weight | 특성 | 용도 |
| --------- | ---- | ---- |
| 1.0 | 강한 정규화, latent space 구조화 | 일반 VAE, 생성 모델 |
| 0.01 ~ 0.1 | 균형 잡힌 trade-off | beta-VAE |
| **0.000001** | **거의 무시, reconstruction 최적화** | **SD-style VAE** |

**이유:**
- Diffusion model이 latent space에서 노이즈를 학습하므로, VAE의 역할은 **고품질 reconstruction**
- KL regularization이 너무 강하면 reconstruction이 blurry해짐
- SD3는 KL loss를 거의 사용하지 않고 reconstruction loss (L1 + LPIPS)에 집중

## Evaluation Metrics

| Metric | Description | Interpretation |
| ------ | ----------- | -------------- |
| **PSNR** | Peak Signal-to-Noise Ratio | Higher is better (typical: 20-40 dB) |
| **SSIM** | Structural Similarity Index | 0-1, higher is better |
| **MSE** | Mean Squared Error | Lower is better |

## Results Summary

Evaluated on 100 images from `samples/original/` directory.

| Epoch | PSNR (dB) | SSIM | MSE |
| ----- | --------- | ---- | --- |
| 10 | 32.59 | 0.9490 | 0.000655 |
| 20 | 32.98 | 0.9611 | 0.000585 |
| 30 | 37.22 | 0.9760 | 0.000229 |
| **40** | **37.99** | **0.9784** | **0.000195** |

## Training Progress

```text
PSNR (dB)
40 |                              ████
   |                        ████ ████
35 |                        ████ ████
   |                        ████ ████
   |  ████  ████            ████ ████
30 |  ████  ████            ████ ████
   |  ████  ████            ████ ████
   +----------------------------------
      E10   E20   E30   E40

SSIM
1.0 |
    |                        ████ ████
0.98|                        ████ ████
    |                        ████ ████
0.96|        ████            ████ ████
    |  ████  ████            ████ ████
0.94|  ████  ████            ████ ████
    +----------------------------------
       E10   E20   E30   E40
```

## Comparison with Stable Diffusion VAE

### Official SD VAE Benchmarks (512x512)

| Model | PSNR (dB) | SSIM | rFID |
| ----- | --------- | ---- | ---- |
| Original kl-f8 VAE | 23.4 | 0.69 | 4.99 |
| sd-vae-ft-ema | 23.8 | 0.69 | 4.42 |
| SDXL-VAE | 24.7 | 0.73 | 4.42 |

### Our VAE (64x64)

| Model | PSNR (dB) | SSIM |
| ----- | --------- | ---- |
| **vae_e40** | **37.99** | **0.978** |

### Why Direct Comparison is Difficult

| Factor | SD VAE | Our VAE |
| ------ | ------ | ------- |
| Input Resolution | 512x512 | 64x64 |
| Latent Size | 64x64x4 | 8x8x16 |
| Pixel Count | 262,144 | 4,096 |
| Compression Ratio | ~48:1 | ~12:1 |

- Our VAE has **1/64 resolution** → compression is much easier
- Our VAE has **1/4 compression ratio** → reconstruction is simpler

### Quality Assessment

Despite the differences, our VAE shows **excellent reconstruction quality** for its resolution:

| PSNR Range | Rating | Our Status |
| ---------- | ------ | ---------- |
| > 40 dB | Excellent (near lossless) | - |
| 30-40 dB | Good (minor differences) | **E10-E40** |
| 20-30 dB | Fair (noticeable differences) | - |
| < 20 dB | Poor | - |

**Conclusion:** PSNR 38 dB and SSIM 0.978 are **more than sufficient** for diffusion training. Even SD's VAE works well at PSNR 23-25 dB.

## Analysis

### Key Observations

1. **Significant improvement at Epoch 30**: PSNR jumped from ~33 dB to ~37 dB, indicating the model learned meaningful representations around this point.

2. **SSIM consistently high**: All checkpoints maintain SSIM > 0.94, showing good structural preservation even in early training.

3. **Diminishing returns after Epoch 30**: Improvements from E30 to E40 are incremental (~0.8 dB PSNR, ~0.002 SSIM).

**Recommendation**: Use `vae_e40.pt` for best reconstruction quality. `vae_e30.pt` is also acceptable with minimal quality loss.

## Evaluation Commands

```bash
# Evaluate on local images
./scripts/evaluate-vae.sh --input-dir samples/original --checkpoint checkpoints/vae_e40.pt

# Evaluate on HuggingFace dataset
./scripts/evaluate-vae.sh --dataset reach-vb/pokemon-blip-captions --max-samples 100

# Save results to JSON
./scripts/evaluate-vae.sh --input-dir samples/original --save results/eval.json
```

## Detailed Results

Full per-sample metrics are saved in `results/vae_e{epoch}_eval.json` files.

Example JSON structure:

```json
{
  "checkpoint": "checkpoints/vae_e40.pt",
  "epoch": 39,
  "metrics": {
    "psnr": 37.99,
    "ssim": 0.9784,
    "mse": 0.000195
  },
  "per_sample": [
    {"file": "sample_000.png", "psnr": 35.07, "ssim": 0.977, "mse": 0.00031}
  ]
}
```

## References

- [Understanding PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [SSIM Paper](https://ieeexplore.ieee.org/document/1284395)
- [Stable Diffusion 3 VAE](https://stability.ai/news/stable-diffusion-3)
- [SDXL-VAE Benchmark](https://www.aimodels.fyi/models/huggingFace/sdxl-vae-stabilityai)
- [sd-vae-ft-ema](https://dataloop.ai/library/model/stabilityai_sd-vae-ft-ema/)
