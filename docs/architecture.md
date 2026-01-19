# Architecture Overview

> 이 문서는 tiny-stable-diffusion의 전체 아키텍처 개요를 설명합니다. 각 모델의 상세 구현은 다음 문서를 참조하세요:
> - VAE 상세: [models/VAE.md](./models/VAE.md)
> - MMDiT 상세: [models/MMDiT.md](./models/MMDiT.md)

## 목차

1. [전체 시스템 개요](#전체-시스템-개요)
2. [데이터 흐름](#데이터-흐름)
3. [핵심 컴포넌트](#핵심-컴포넌트)

---

## 전체 시스템 개요

tiny-stable-diffusion은 **Stable Diffusion 3**의 아키텍처를 따르는 교육용 구현체입니다. 두 단계로 구성된 파이프라인으로, 효율적인 latent space diffusion을 실현합니다.

### 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         tiny-stable-diffusion Architecture                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │    Image    │────▶│     VAE     │────▶│  Diffusion  │────▶│   Output    │   │
│  │   (64×64)   │     │   Encoder   │     │  Transformer│     │   Image     │   │
│  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘   │
│         │                   │                   ▲                   ▲           │
│         │                   ▼                   │                   │           │
│         │            ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│         │            │   Latent    │     │    Text     │     │     VAE     │   │
│         │            │  (8×8×16)   │     │  Encoder    │     │   Decoder   │   │
│         │            └─────────────┘     │   (CLIP)    │     └─────────────┘   │
│         │                                └─────────────┘                       │
│         │                                       ▲                               │
│         │                                       │                               │
│         │                                ┌─────────────┐                       │
│         └───────────────────────────────▶│   Prompt    │                       │
│                                          │  "a cat..."  │                       │
│                                          └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 핵심 특징

| 구성요소 | 역할 | 관련 문서 |
|----------|------|-----------|
| **VAE** | 이미지를 latent space로 압축/복원 | [models/VAE.md](./models/VAE.md) |
| **Diffusion Transformer** | 노이즈 제거를 통한 이미지 생성 | `src/models/vanilla_dit.py` |
| **MMDiT** | SD3 스타일의 Joint Attention 기반 생성 | [models/MMDiT.md](./models/MMDiT.md) |
| **CLIP Encoder** | 텍스트 프롬프트를 임베딩으로 변환 | `src/text_encoder/clip_encoder.py` |

---

## 데이터 흐름

| 단계 | 입력 | 출력 | 차원 변화 |
|------|------|------|-----------|
| 1. Image Input | RGB Image | - | `(B, 3, 64, 64)` |
| 2. VAE Encode | Image | Latent | `(B, 3, 64, 64)` → `(B, 16, 8, 8)` |
| 3. Add Noise | Clean Latent | Noisy Latent | `(B, 16, 8, 8)` → `(B, 16, 8, 8)` |
| 4. DiT Predict | Noisy Latent + Text | Predicted Noise | `(B, 16, 8, 8)` → `(B, 16, 8, 8)` |
| 5. Denoise | Predicted Noise | Clean Latent | `(B, 16, 8, 8)` → `(B, 16, 8, 8)` |
| 6. VAE Decode | Clean Latent | Output Image | `(B, 16, 8, 8)` → `(B, 3, 64, 64)` |

---

## 핵심 컴포넌트

### 1. VAE (Variational AutoEncoder)

이미지를 저차원 잠재 공간(latent space)으로 압축하고 복원합니다.

- **압축률**: 64×64×3 = 12,288 → 8×8×16 = **1,024** (12배 효율적)
- **구조**: Encoder + Decoder (ResNet + Self-Attention)
- **손실함수**: Reconstruction Loss + KL Divergence
- **자세한 내용**: [models/VAE.md](./models/VAE.md)

### 2. Diffusion Process

노이즈 추가(Forward)와 제거(Reverse) 과정을 정의합니다.

| 요소 | 설명 |
|------|------|
| **Beta Schedule** | Cosine schedule (1000 timesteps) |
| **Sampling** | DDPM (확률적) / DDIM (결정론적) |
| **Conditioning** | Classifier-Free Guidance (CFG) |
| **Loss Weighting** | Min-SNR Weighting |

자세한 내용은 `src/models/diffusion.py`을 참조하세요.

### 3. Diffusion Transformer (DiT / MMDiT)

Latent representation에서 노이즈를 예측하는 트랜스포머 모델입니다.

| 모델 | 특징 | 파일 위치 |
|------|------|-----------|
| **Vanilla DiT** | Cross-Attention 기반 text conditioning | `src/models/vanilla_dit.py` |
| **MMDiT** | Joint Attention, 양방향 text-image 상호작용 | [models/MMDiT.md](./models/MMDiT.md) |

**모델 크기별 파라미터:**
- S: ~40M / B: ~160M / L: ~560M / XL: ~820M

### 4. Text Encoder (CLIP)

텍스트 프롬프트를 embedding으로 변환합니다.

- **모델**: CLIP ViT-B/32
- **임베딩 차원**: 512
- **토큰 길이**: 77 tokens (BPE)
- **파일 위치**: `src/text_encoder/clip_encoder.py`

---

## 파라미터 요약

| 모델 조합 | Total Parameters |
|-----------|-----------------|
| VAE + DiT-S | ~61M |
| VAE + MMDiT-S | ~108M |
| VAE + DiT-B | ~181M |
| VAE + MMDiT-B | ~208M |

---

## 참고 자료

- [VAE Paper](https://arxiv.org/abs/1312.6114) - Auto-Encoding Variational Bayes
- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [DDIM Paper](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models
- [DiT Paper](https://arxiv.org/abs/2212.09748) - Scalable Diffusion Models with Transformers
- [SD3 Paper](https://arxiv.org/abs/2403.03206) - Scaling Rectified Flow Transformers
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models
