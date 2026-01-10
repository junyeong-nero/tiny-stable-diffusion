# tiny-stable-diffusion

> **Stable Diffusion 3 from Scratch** - A minimal educational implementation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview

**tiny-stable-diffusion**은 교육 목적으로 Stable Diffusion 3 파이프라인을 처음부터 구현한 프로젝트입니다. 실제 SD3와 동일한 구조를 따르면서 64x64 해상도로 경량화하여 일반 GPU에서도 학습할 수 있습니다.

### 핵심 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Stable Diffusion 3 Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. VAE Training (Stage 1)                                              │
│     Image → Encoder → Latent Space → Decoder → Reconstructed Image      │
│                                                                         │
│  2. Diffusion Training (Stage 2)                                        │
│     Image → [Frozen VAE] → Latent → DiT + Text → Noise Prediction       │
│                                                                         │
│  3. Generation (Inference)                                              │
│     Noise → DiT Denoise → Clean Latent → [VAE Decoder] → Image          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 왜 Latent Space Diffusion인가?

| Pixel Space | Latent Space |
|-------------|--------------|
| 64×64×3 = 12,288 차원 | 8×8×16 = 1,024 차원 |
| 계산량 많음 | **12배 효율적** |
| 메모리 사용량 높음 | **메모리 절약** |
| 고해상도 어려움 | **고해상도 가능** |

VAE로 이미지를 압축한 후 latent space에서 diffusion을 수행하면 훨씬 효율적으로 학습할 수 있습니다.

---

## Architecture

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    tiny-stable-diffusion                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: VAE Training                                           │
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │  Image  │  ->  │ Encoder │  ->  │ Latent  │                │
│   │ (64×64) │      │         │      │(16,8,8) │                │
│   └─────────┘      └─────────┘      └────┬────┘                │
│                                          │                      │
│                                          v                      │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │  Recon  │  <-  │ Decoder │  <-  │ Sample  │                │
│   │  Image  │      │         │      │  z~N    │                │
│   └─────────┘      └─────────┘      └─────────┘                │
│                                                                 │
│   Loss = MSE(Image, Recon) + β × KL(q(z|x) || p(z))            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Diffusion Training (Latent Space)                      │
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │  Image  │  ->  │   VAE   │  ->  │ Latent  │                │
│   │ (64×64) │      │ Encoder │      │(16,8,8) │                │
│   └─────────┘      └─────────┘      └────┬────┘                │
│                     (frozen)              │                     │
│                                          v                      │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │Predicted│  <-  │   DiT   │  <-  │ Noisy   │                │
│   │  Noise  │      │ + Text  │      │ Latent  │                │
│   └─────────┘      └─────────┘      └─────────┘                │
│                                                                 │
│   Loss = MSE(Predicted Noise, Actual Noise) × SNR Weight        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Generation                                             │
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌───────┐ │
│   │ Random  │  ->  │   DiT   │  ->  │  Clean  │  ->  │ Image │ │
│   │  Noise  │      │ Denoise │      │ Latent  │      │(64×64)│ │
│   └─────────┘      └─────────┘      └─────────┘      └───────┘ │
│                    (DDIM 50步)            │                     │
│                         ^                 v                     │
│                    ┌─────────┐      ┌─────────┐                │
│                    │  Text   │      │   VAE   │                │
│                    │  CLIP   │      │ Decoder │                │
│                    └─────────┘      └─────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### VAE (Variational AutoEncoder)

SD3 스타일의 AutoencoderKL 구현:

```
Encoder:
  Conv3x3(3→64) → ResBlock×2 → Downsample
                → ResBlock×2 → Downsample
                → ResBlock×2 → Downsample
                → ResBlock×2 → SelfAttention → ResBlock
                → Conv3x3(512→32) → [mean, logvar]

Decoder:
  Conv3x3(16→512) → ResBlock → SelfAttention → ResBlock
                  → ResBlock×3 → Upsample
                  → ResBlock×3 → Upsample
                  → ResBlock×3 → Upsample
                  → Conv3x3(64→3)
```

| 설정 | 값 |
|------|-----|
| Input | 64×64×3 RGB |
| Latent | 8×8×16 |
| Compression | f8 (8배 압축) |
| Base channels | 64 |
| Channel multipliers | [1, 2, 4, 4] |
| Parameters | ~21M |

### Diffusion Transformer (DiT / MMDiT)

두 가지 아키텍처를 지원합니다:

#### DiT (Vanilla) - Cross-Attention 방식
```
Image Tokens → Self-Attention → Cross-Attention(with Text) → MLP → Output
```

| Size | Layers | Hidden | Heads | Params | 용도 |
|------|--------|--------|-------|--------|------|
| **S** | 12 | 384 | 6 | **39.9M** | 기본값, 빠른 실험 |
| B | 12 | 768 | 12 | **158.8M** | 중간 규모 |
| L | 24 | 1024 | 16 | **559.0M** | 고품질 |
| XL | 28 | 1152 | 16 | **824.2M** | 최고 품질 |

#### MMDiT (SD3 스타일) - Joint Attention 방식
```
[Text Tokens, Image Tokens] → Joint Self-Attention → Separate MLPs → Output
```

| Size | Layers | Hidden | Heads | Params | 용도 |
|------|--------|--------|-------|--------|------|
| **S** | 12 | 384 | 6 | **87.0M** | 기본값, 권장 |
| B | 12 | 768 | 12 | **186.9M** | 중간 규모 |
| L | 24 | 1024 | 16 | **558.9M** | 고품질 |
| XL | 28 | 1152 | 16 | **780.1M** | 최고 품질 |

**DiT vs MMDiT 비교:**
| 특징 | DiT | MMDiT |
|------|-----|-------|
| Text 처리 | Cross-Attention | Joint Attention |
| 구조 | 분리된 attention | 통합 attention |
| 학습 안정성 | 좋음 | 더 좋음 (QK-RMSNorm) |
| 실제 SD3 | ❌ | ✅ |

**DiT Block 구조:**
```
Input → LayerNorm → Self-Attention → + → LayerNorm → Cross-Attention → + → LayerNorm → MLP → + → Output
         ↑                           |                                  |               |
         └── AdaLN-Zero (timestep) ──┴──────────────────────────────────┴───────────────┘
                                     (text conditioning)
```

**MMDiT Block 구조:**
```
[Text, Image] → Joint LayerNorm → Joint Self-Attention → Split → Separate MLPs → Output
                     ↑                                              |
                     └──────── Time Conditioning ───────────────────┘
```

### 실제 SD3와의 비교

| Component | Stable Diffusion 3 | tiny-stable-diffusion |
|-----------|-------------------|----------------------|
| Image Size | 1024×1024 | **64×64** |
| VAE Latent Channels | 16 | **16** |
| VAE Compression | f8 | **f8** |
| Diffusion Architecture | MMDiT | **DiT / MMDiT** |
| Text Encoder | T5-XXL + CLIP-G + CLIP-L | **CLIP ViT-B/32** |
| Total Parameters | 2B+ | **~60M** |
| Training Time | 수천 GPU-hours | **수 시간** |

---

## Quick Start

### 설치

```bash
# uv 패키지 매니저 설치 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 또는 pip 사용
pip install -e .
```

### 전체 학습 파이프라인

```bash
# Step 1: VAE 학습 (이미지 압축 학습)
uv run main.py --train-vae --epochs 100 --batch-size 32

# Step 2: Diffusion 학습 (latent space에서 노이즈 제거 학습)
uv run main.py --train-diffusion --epochs 200 --batch-size 32

# Step 3: 이미지 생성
uv run main.py --generate --prompt "a cute cat sitting on a couch"
```

### 빠른 테스트

```bash
# 작은 데이터셋으로 빠르게 테스트
uv run main.py --train-vae --epochs 10 --dataset reach-vb/pokemon-blip-captions
uv run main.py --train-diffusion --epochs 20 --dataset reach-vb/pokemon-blip-captions
```

---

## Training Guide

### Stage 1: VAE Training

VAE는 이미지를 latent space로 압축하고 다시 복원하는 방법을 학습합니다.

```bash
uv run main.py --train-vae \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4
```

**주요 설정 (config.yaml):**
```yaml
vae_train:
    image_size: 64           # 입력 이미지 크기
    latent_channels: 16      # latent 채널 수 (SD3 스타일)
    vae_ch: 64               # VAE 기본 채널
    vae_ch_mult: [1, 2, 4, 4]  # 채널 증가 비율
    kl_weight: 1.0e-6        # KL divergence 가중치
    epochs: 100
    batch_size: 32
    learning_rate: 1.0e-4
    checkpoint_path: checkpoints/vae.pt
```

**학습 팁:**
- `kl_weight`가 너무 크면 reconstruction quality가 떨어집니다
- `kl_weight`가 너무 작으면 posterior collapse가 발생할 수 있습니다
- 권장 시작값: `1e-6`

**출력:**
- `checkpoints/vae.pt`: 학습된 VAE 체크포인트
- `samples/vae_epoch_N/`: reconstruction 샘플 이미지

### Stage 2: Diffusion Training

사전 학습된 VAE를 사용하여 latent space에서 diffusion 모델을 학습합니다.

```bash
uv run main.py --train-diffusion \
    --vae-checkpoint checkpoints/vae.pt \
    --epochs 200 \
    --batch-size 32
```

**주요 설정 (config.yaml):**
```yaml
diffusion_train:
    image_size: 64           # 원본 이미지 크기
    latent_size: 8           # latent 공간 크기 (64/8)
    in_channels: 16          # latent 채널 (VAE와 일치)

    # CFG (Classifier-Free Guidance) 설정
    initial_cfg_prob: 0.0    # 초기 unconditional dropout 확률
    final_cfg_prob: 0.1      # 최종 unconditional dropout 확률
    cfg_warmup_epochs: 10    # CFG warmup 기간

    # VAE
    vae_checkpoint: checkpoints/vae.pt

    epochs: 200
    batch_size: 32
    learning_rate: 1.0e-4
    checkpoint_path: checkpoints/diffusion.pt
```

**학습 과정:**
1. 이미지를 frozen VAE encoder로 latent로 변환
2. Latent에 노이즈 추가
3. DiT가 노이즈 예측
4. MSE loss로 학습

**출력:**
- `checkpoints/diffusion.pt`: 학습된 diffusion 체크포인트
- `samples/epoch_N/`: 생성된 샘플 이미지

---

## Generation

### 기본 생성

```bash
# 단일 프롬프트
uv run main.py --generate --prompt "a photo of a cat"

# 여러 프롬프트
uv run main.py --generate --prompt "cat,dog,sunset,mountain"

# 프롬프트당 여러 샘플
uv run main.py --generate --prompt "a robot" --num-samples 4
```

### 고급 옵션

```bash
uv run main.py --generate \
    --prompt "a beautiful landscape with mountains" \
    --checkpoint checkpoints/diffusion.pt \
    --vae-checkpoint checkpoints/vae.pt \
    --steps 100 \           # diffusion steps (default: 50)
    --guidance 7.5 \        # CFG scale (default: 7.5)
    --seed 42 \             # 재현성을 위한 시드
    --output my_image.png
```

### Interactive Demo

```bash
uv run main.py --demo
```

프롬프트를 입력하면 실시간으로 이미지를 생성합니다.

---

## Configuration

모든 설정은 `config.yaml`에서 관리됩니다:

```yaml
# tiny-stable-diffusion Configuration

# 현재 학습 단계: "vae_train" 또는 "diffusion_train"
training_stage: vae_train

# Diffusion 모델 타입: "dit" 또는 "mmdit"
model_type: mmdit

# ═══════════════════════════════════════════════════════════════
# Stage 1: VAE Training
# ═══════════════════════════════════════════════════════════════
vae_train:
    data_source: caption
    dataset_name: jxie/flickr8k
    image_size: 64
    latent_channels: 16
    vae_ch: 64
    vae_ch_mult: [1, 2, 4, 4]
    kl_weight: 1.0e-6
    epochs: 100
    batch_size: 32
    learning_rate: 1.0e-4
    checkpoint_path: checkpoints/vae.pt

# ═══════════════════════════════════════════════════════════════
# Stage 2: Diffusion Training
# ═══════════════════════════════════════════════════════════════
diffusion_train:
    data_source: caption
    dataset_name: jxie/flickr8k
    image_size: 64
    latent_size: 8
    in_channels: 16
    initial_cfg_prob: 0.0
    final_cfg_prob: 0.1
    cfg_warmup_epochs: 10
    vae_checkpoint: checkpoints/vae.pt
    epochs: 200
    batch_size: 32
    learning_rate: 1.0e-4
    checkpoint_path: checkpoints/diffusion.pt

# ═══════════════════════════════════════════════════════════════
# Common Settings
# ═══════════════════════════════════════════════════════════════
common:
    model_size: S           # S, B, L, XL
    patch_size: 2
    num_timesteps: 1000
    beta_schedule: cosine
    guidance_scale: 7.5
    use_ema: true
    ema_decay: 0.9999
    device: auto
    seed: 42
    validation_prompts:
        - a photo of a cat
        - a rocket flying in space
        - a robot with blue eyes
    validation_interval: 10
```

---

## Dataset

### 권장 데이터셋

| 데이터셋 | 크기 | 특징 | 용도 |
|---------|------|------|------|
| **jxie/flickr8k** | 8K images | 이미지당 5개 캡션, 고품질 | 권장 |
| reach-vb/pokemon-blip-captions | 833 images | 픽셀아트 스타일 | 빠른 테스트 |

### 데이터셋 변경

```bash
# CLI로 변경
uv run main.py --train-vae --dataset jxie/flickr8k

# 또는 config.yaml 수정
vae_train:
    dataset_name: jxie/flickr8k
```

---

## Project Structure

```
tiny-stable-diffusion/
├── main.py                         # CLI 진입점
├── config.yaml                     # 설정 파일
├── pyproject.toml                  # 프로젝트 메타데이터
├── README.md                       # 이 문서
│
├── src/
│   ├── models/
│   │   ├── vae.py                  # VAE (AutoencoderKL)
│   │   │   ├── Encoder             # 이미지 → latent
│   │   │   ├── Decoder             # latent → 이미지
│   │   │   └── training_loss()     # VAE 손실 함수
│   │   │
│   │   ├── diffusion.py            # DDPM/DDIM 프로세스
│   │   │   ├── q_sample()          # forward diffusion
│   │   │   ├── p_sample()          # reverse (DDPM)
│   │   │   ├── ddim_sample()       # reverse (DDIM)
│   │   │   └── sample()            # 전체 생성 루프
│   │   │
│   │   ├── factory.py              # DiT 모델 팩토리
│   │   ├── vanilla_dit.py          # 표준 DiT 구현
│   │   ├── mmdit.py                # Multi-Modal DiT (SD3)
│   │   └── layers.py               # 공통 레이어
│   │
│   ├── training/
│   │   ├── vae_trainer.py          # VAE 학습 루프
│   │   ├── trainer.py              # Diffusion 학습 루프
│   │   ├── ema.py                  # Exponential Moving Average
│   │   └── checkpoint.py           # 체크포인트 관리
│   │
│   ├── inference/
│   │   └── generator.py            # 이미지 생성
│   │
│   ├── text_encoder/
│   │   └── clip_encoder.py         # CLIP 텍스트 인코더
│   │
│   ├── data/
│   │   ├── dataset.py              # 데이터셋 로더
│   │   └── loader.py               # DataLoader 유틸리티
│   │
│   ├── config/
│   │   ├── loader.py               # config.yaml 로더
│   │   └── dataclasses.py          # 설정 데이터클래스
│   │
│   └── utils/
│       └── common.py               # 공통 유틸리티
│
├── checkpoints/                    # 저장된 모델
│   ├── vae.pt                      # VAE 체크포인트
│   └── diffusion.pt                # Diffusion 체크포인트
│
├── samples/                        # 생성된 샘플
│   ├── vae_epoch_N/                # VAE reconstruction
│   └── epoch_N/                    # Diffusion 생성 결과
│
└── tests/                          # 테스트 코드
```

---

## CLI Reference

```
usage: main.py [-h] [--train-vae] [--train-diffusion] [--train]
               [--generate] [--demo] [options]

tiny-stable-diffusion - Stable Diffusion 3 from Scratch

Training:
  --train-vae           Stage 1: VAE 학습
  --train-diffusion     Stage 2: Diffusion 학습 (VAE 필요)
  --train               config.yaml의 training_stage 사용

  --epochs N            에포크 수
  --batch-size N        배치 크기
  --learning-rate F     학습률
  --dataset NAME        데이터셋 이름
  --vae-checkpoint P    VAE 체크포인트 경로

Generation:
  --generate            이미지 생성
  --demo                인터랙티브 데모

  --prompt TEXT         프롬프트 (쉼표로 구분)
  --num-samples N       프롬프트당 샘플 수
  --steps N             diffusion 스텝 (default: 50)
  --guidance F          CFG 스케일 (default: 7.5)
  --seed N              랜덤 시드
  --checkpoint P        Diffusion 체크포인트
  --output PATH         출력 파일 경로

Logging:
  --wandb               Wandb 로깅 활성화
  --wandb-project NAME  Wandb 프로젝트 이름
  --wandb-run-name NAME Wandb 런 이름
```

---

## Technical Details

### Diffusion Process

**Forward Process (노이즈 추가):**
```
x_t = √(α̅_t) × x_0 + √(1 - α̅_t) × ε
```

**Reverse Process (DDIM):**
```
x_{t-1} = √(α̅_{t-1}) × pred_x_0 + √(1 - α̅_{t-1} - σ²) × ε_θ + σ × z
```

**Min-SNR Weighting:**
```python
snr = α̅_t / (1 - α̅_t)
weight = min(snr, γ) / snr  # γ = 5.0
loss = weight × MSE(ε_θ, ε)
```

### Classifier-Free Guidance (CFG)

```python
# 학습 시: 10% 확률로 텍스트 조건 드롭
if random() < 0.1:
    text_embed = uncond_embed  # 빈 문자열 임베딩

# 추론 시: conditional과 unconditional 예측 결합
noise_pred = uncond_pred + guidance_scale × (cond_pred - uncond_pred)
```

---

## References

- **Stable Diffusion 3**: [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)
- **DiT**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **DDIM**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- **VAE**: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Min-SNR**: [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

이슈와 PR을 환영합니다. 교육 목적의 프로젝트이므로 코드의 명확성과 이해하기 쉬운 구현을 중시합니다.
