# Training Quick Start Guide

> tiny-stable-diffusion 훈련을 위한 빠른 시작 가이드입니다.
> 더 자세한 내용은 [training-pipeline.md](./training-pipeline.md)를 참조하세요.

## 목차

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Hyperparameters](#hyperparameters)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. 환경 설정

```bash
# uv 설치 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 또는 pip 사용
pip install -e .
```

### 2. Stage 1: VAE 훈련

```bash
# 기본 VAE 훈련
uv run main.py --train-vae --epochs 100 --batch-size 32

# Wandb 로깅 활성화
uv run main.py --train-vae --epochs 100 --batch-size 32 --wandb

# 커스텀 데이터셋 사용
uv run main.py --train-vae --epochs 100 --dataset reach-vb/pokemon-blip-captions
```

### 3. Stage 2: Diffusion 훈련

```bash
# 기본 Diffusion 훈련 (VAE 필요)
uv run main.py --train-diffusion --epochs 200 --batch-size 32

# VAE 체크포인트 지정
uv run main.py --train-diffusion --vae-checkpoint checkpoints/vae.pt --epochs 200
```

### 4. 이미지 생성

```bash
# 단일 이미지 생성
uv run main.py --generate --prompt "a cute cat"

# 여러 이미지 생성
uv run main.py --generate --prompt "a robot,a sunset,a mountain" --num-samples 4

# 설정 옵션
uv run main.py --generate \
    --prompt "a beautiful landscape" \
    --steps 50 \
    --guidance 7.5 \
    --seed 42
```

### 5. HuggingFace Hub에 업로드

```bash
# VAE 훈련 후 업로드
uv run main.py --train-vae --push-to-hub --hub-model-id username/my-vae

# Diffusion 훈련 후 업로드
uv run main.py --train-diffusion --push-to-hub --hub-model-id username/my-diffusion
```

---

## Configuration

### config.yaml 구조

```yaml
# 현재 훈련 단계
training_stage: vae_train  # 또는 diffusion_train

# VAE 훈련 설정
vae_train:
    data_source: streaming_caption
    dataset_name: hmu013/LAION-300k
    image_size: 64
    latent_channels: 16
    epochs: 100
    batch_size: 128
    learning_rate: 4.0e-4
    kl_weight: 1.0e-6
    checkpoint_path: checkpoints/vae.pt

# Diffusion 훈련 설정
diffusion_train:
    model_type: mmdit  # dit 또는 mmdit
    model_size: S      # S, B, L, XL
    epochs: 200
    batch_size: 32
    learning_rate: 1.0e-4
    guidance_scale: 7.5
    use_ema: true
    ema_decay: 0.9999
    vae_checkpoint: checkpoints/vae.pt
    checkpoint_path: checkpoints/diffusion.pt
```

### CLI 우선순위

CLI 인자가 config.yaml 값을 덮어씁니다:

```bash
# config.yaml에서 epochs=100이어도 CLI가 우선
uv run main.py --train-vae --epochs 50
```

---

## Hyperparameters

### VAE 훈련

| 파라미터 | 기본값 | 권장 범위 | 설명 |
|----------|--------|-----------|------|
| `epochs` | 100 | 50-200 | 훈련 에폭 수 |
| `batch_size` | 128 | 32-256 | 배치 크기 |
| `learning_rate` | 4e-4 | 1e-4 ~ 1e-3 | 학습률 |
| `kl_weight` | 1e-6 | 1e-7 ~ 1e-5 | KL 손실 가중치 |

### Diffusion 훈련

| 파라미터 | 기본값 | 권장 범위 | 설명 |
|----------|--------|-----------|------|
| `epochs` | 200 | 100-500 | 훈련 에폭 수 |
| `batch_size` | 32 | 16-64 | 배치 크기 |
| `learning_rate` | 1e-4 | 5e-5 ~ 3e-4 | 학습률 |
| `guidance_scale` | 7.5 | 3.0-15.0 | CFG 스케일 |
| `cfg_probability` | 0.1 | 0.05-0.2 | CFG 드롭아웃 확률 |
| `ema_decay` | 0.9999 | 0.999-0.9999 | EMA 감쇠율 |

### 모델 크기

| Size | Layers | Hidden | Heads | Params | VRAM |
|------|--------|--------|-------|--------|------|
| **S** | 12 | 384 | 6 | ~40M | ~4GB |
| B | 12 | 768 | 12 | ~160M | ~8GB |
| L | 24 | 1024 | 16 | ~560M | ~16GB |
| XL | 28 | 1152 | 16 | ~820M | ~24GB |

---

## Hardware Requirements

### GPU 메모리

| 단계 | Model Size | Batch Size | VRAM |
|------|------------|------------|------|
| VAE | - | 32 | ~4GB |
| VAE | - | 128 | ~8GB |
| Diffusion | S | 32 | ~6GB |
| Diffusion | B | 32 | ~12GB |
| Diffusion | L | 16 | ~20GB |

### 권장 사양

**최소:**
- GPU: RTX 3060 12GB 이상
- RAM: 16GB
- Storage: 20GB SSD

**권장:**
- GPU: RTX 3090 24GB 이상
- RAM: 32GB
- Storage: 50GB SSD

### Apple Silicon (MPS)

```bash
# MPS 자동 감지
uv run main.py --train-vae --batch-size 32

# 또는 config.yaml에서 설정
# device: mps
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# 해결책 1: 배치 크기 줄이기
--batch-size 16

# 해결책 2: Mixed precision 활성화
# config.yaml에서
mixed_precision: true

# 해결책 3: 모델 크기 줄이기
model_size: S
```

### Loss가 감소하지 않음

1. **학습률 확인**: 너무 높으면 불안정, 너무 낮으면 느림
2. **KL weight 확인**: VAE에서 1e-6 권장
3. **데이터셋 확인**: 이미지가 제대로 로딩되는지

### NaN Loss 발생

```bash
# 해결책: 학습률 낮추기
--learning-rate 5e-5

# 또는 gradient clipping 추가 (코드 수정 필요)
```

### 생성 품질이 낮음

1. **더 많은 에폭** 훈련
2. **Guidance scale** 조정: 7.5-10.0
3. **Steps** 늘리기: 50-100
4. **EMA weights** 사용 확인

### CLIP 설치 오류

```bash
# OpenAI CLIP 설치
pip install git+https://github.com/openai/CLIP.git
```

---

## Best Practices

### 1. 점진적 훈련

```bash
# Step 1: 작은 데이터셋으로 테스트
uv run main.py --train-vae --epochs 10 --dataset reach-vb/pokemon-blip-captions

# Step 2: 큰 데이터셋으로 본 훈련
uv run main.py --train-vae --epochs 100 --dataset hmu013/LAION-300k
```

### 2. 체크포인트 관리

```bash
# 훈련 중 자동 저장: best loss 기준
# 위치: checkpoints/vae.pt, checkpoints/diffusion.pt

# HuggingFace Hub에 백업
--push-to-hub --hub-model-id username/model-name
```

### 3. 모니터링

```bash
# Wandb로 훈련 모니터링
--wandb --wandb-project tiny-stable-diffusion

# 샘플 확인
# samples/vae_epoch_N/: VAE reconstruction
# samples/epoch_N/: Diffusion generation
```

### 4. 재현성

```bash
# 시드 고정
# config.yaml에서
seed: 42

# 또는 generation 시
--seed 42
```

---

## 추가 문서

- [Architecture Deep Dive](./architecture.md) - 모델 아키텍처 상세
- [Training Pipeline Deep Dive](./training-pipeline.md) - 훈련 과정 상세
- [Inference Deep Dive](./inference.md) - 이미지 생성 상세

---

## 참고 자료

- [Stable Diffusion 3 Paper](https://arxiv.org/abs/2403.03206)
- [DiT Paper](https://arxiv.org/abs/2212.09748)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [VAE Paper](https://arxiv.org/abs/1312.6114)