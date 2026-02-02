# Architecture Overview

> 이 문서는 tiny-stable-diffusion의 전체 아키텍처 개요를 설명합니다. 각 모델의 상세 구현은 다음 문서를 참조하세요:
> - VAE 상세: [models/VAE.md](./models/VAE.md)
> - MMDiT 상세: [models/MMDiT.md](./models/MMDiT.md)
> - Diffusion 상세: [models/Diffusion.md](./models/Diffusion.md)

## 목차

1. [전체 시스템 개요](#전체-시스템-개요)
2. [데이터 흐름](#데이터-흐름)
3. [핵심 컴포넌트](#핵심-컴포넌트)

---

## 전체 시스템 개요

tiny-stable-diffusion은 **Stable Diffusion 3**의 아키텍처를 따르는 교육용 구현체입니다. 세 단계로 구성된 파이프라인으로, 이미지 생성부터 애니메이션(GIF) 생성까지 지원합니다.

### 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         tiny-stable-diffusion Architecture                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │    Image/   │────▶│     VAE     │────▶│  Diffusion  │────▶│   Output    │   │
│  │    Video    │     │   Encoder   │     │  Transformer│     │  Image/GIF  │   │
│  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘   │
│         │                   │                   ▲                   ▲           │
│         │                   ▼                   │                   │           │
│         │            ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│         │            │   Latent    │     │    Text     │     │     VAE     │   │
│         │            │  (F×16×8×8) │     │  Encoder    │     │   Decoder   │   │
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
| **VAE** | 이미지/프레임을 latent space로 압축/복원 | [models/VAE.md](./models/VAE.md) |
| **Diffusion** | Rectified Flow 기반 노이즈 제거 | [models/Diffusion.md](./models/Diffusion.md) |
| **MMDiT** | SD3 스타일의 Joint Attention 기반 Transformer | [models/MMDiT.md](./models/MMDiT.md) |
| **Motion Module** | 애니메이션 생성을 위한 Temporal Attention 레이어 | [extensions/MotionModule.md](./extensions/MotionModule.md) |
| **CLIP Encoder** | 텍스트 프롬프트를 임베딩으로 변환 | `src/text_encoder/clip_encoder.py` |

---

## 데이터 흐름

### 1. 이미지 생성 (Image Generation)
- `(B, 3, 64, 64)` → VAE → `(B, 16, 8, 8)` → MMDiT → Denoise → VAE → `(B, 3, 64, 64)`

### 2. 애니메이션 생성 (GIF Generation)
- `(B, F, 3, 64, 64)` → VAE (frame-wise) → `(B, F, 16, 8, 8)` → Animated MMDiT (Temporal Attention) → Denoise → VAE → `(B, F, 3, 64, 64)`

---

## 핵심 컴포넌트

### 1. VAE (Variational AutoEncoder)

이미지를 저차원 잠재 공간(latent space)으로 압축하고 복원합니다. GIF 생성 시 각 프레임을 독립적으로 처리합니다.

- **압축률**: 64×64×3 = 12,288 → 8×8×16 = **1,024** (12배 효율적)
- **구조**: Encoder + Decoder (ResNet + Self-Attention)
- **자세한 내용**: [models/VAE.md](./models/VAE.md)

### 2. Diffusion Process (Rectified Flow)

SD3에서 도입된 Rectified Flow 방식을 사용합니다. 애니메이션 생성 시 시간 축(Frame)을 포함한 Latent Tensor에 대해 작동합니다.

| 요소 | 설명 |
|------|------|
| **Method** | Rectified Flow |
| **Noise Schedule** | Linear Schedule (Timestep 0 -> 1) |
| **Prediction** | Velocity (v) Prediction |
| **Sampling** | Euler ODE Solver |

### 3. Diffusion Transformer (DiT / MMDiT)

Latent representation에서 노이즈(Velocity)를 예측하는 트랜스포머 모델입니다.

| 모델 | 특징 | 파일 위치 |
|------|------|-----------|
| **MMDiT** | Joint Attention, 양방향 text-image 상호작용 | [models/MMDiT.md](./models/MMDiT.md) |
| **Animated MMDiT** | MMDiT + Temporal Attention Layers | `src/models/animated_mmdit.py` |

### 4. Motion Module (GIF Generation)

기존 이미지 생성 모델을 확장하여 시간적 일관성을 가진 애니메이션을 생성하는 모듈입니다.

```
┌──────────────────────────────────────────────────────┐
│  MMDiT/DiT (frozen) + Motion Module (trainable)      │
│  ┌─────────────────┐  ┌───────────────────────────┐  │
│  │ Spatial Attn    │→ │ Temporal Attn (NEW)       │  │
│  │ (기존, frozen)   │  │ (frames 간 attention)     │  │
│  └─────────────────┘  └───────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

- **구조**: Temporal Attention Layer를 기존 Spatial Attention 뒤에 삽입하여 프레임 간의 움직임을 학습합니다.
- **학습**: 기존 MMDiT 가중치는 고정(freeze)하고 Motion Module만 비디오 데이터로 학습합니다.
- **상세 내용**: [extensions/MotionModule.md](./extensions/MotionModule.md)

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
- [DiT Paper](https://arxiv.org/abs/2212.09748) - Scalable Diffusion Models with Transformers
- [SD3 Paper](https://arxiv.org/abs/2403.03206) - Scaling Rectified Flow Transformers
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models
