# Inference & Generation Deep Dive

> 이 문서는 tiny-stable-diffusion의 이미지 생성 과정을 상세하게 설명합니다.

## 목차

1. [Generation Pipeline Overview](#generation-pipeline-overview)
2. [Sampling Algorithms](#sampling-algorithms)
3. [Classifier-Free Guidance](#classifier-free-guidance)
4. [Step-by-Step Generation](#step-by-step-generation)
5. [Advanced Options](#advanced-options)
6. [Performance Optimization](#performance-optimization)

---

## Generation Pipeline Overview

> 파일 위치: `src/inference/generator.py`

### 전체 생성 과정

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Image Generation Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Input: "a cute cat sitting on a couch"                                        │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 1: Text Encoding (CLIP)                                              │ │
│  │                                                                            │ │
│  │ "a cute cat..." ──▶ Tokenize ──▶ CLIP Transformer ──▶ text_embed (1, 512)│ │
│  │                                                                            │ │
│  │ "empty string"  ──▶ Tokenize ──▶ CLIP Transformer ──▶ uncond_embed (1,512)│ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 2: Initialize Random Noise                                           │ │
│  │                                                                            │ │
│  │ z_T = randn(1, 16, 8, 8)  # Pure Gaussian noise in latent space          │ │
│  │                                                                            │ │
│  │ if seed provided:                                                          │ │
│  │     torch.manual_seed(seed)  # Reproducible generation                    │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 3: Iterative Denoising (DDIM 50 steps)                               │ │
│  │                                                                            │ │
│  │ for t in [999, 979, 959, ..., 19, 0]:                                     │ │
│  │     ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │     │ # Conditional prediction                                         │   │ │
│  │     │ noise_cond = DiT(z_t, t, text_embed)                            │   │ │
│  │     │                                                                  │   │ │
│  │     │ # Unconditional prediction                                       │   │ │
│  │     │ noise_uncond = DiT(z_t, t, uncond_embed)                        │   │ │
│  │     │                                                                  │   │ │
│  │     │ # Classifier-Free Guidance                                       │   │ │
│  │     │ noise_pred = uncond + scale × (cond - uncond)                   │   │ │
│  │     │                                                                  │   │ │
│  │     │ # DDIM step                                                      │   │ │
│  │     │ z_{t-1} = ddim_step(z_t, noise_pred, t)                         │   │ │
│  │     └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                            │ │
│  │ Output: z_0 (1, 16, 8, 8) - Clean latent                                  │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 4: VAE Decoding                                                       │ │
│  │                                                                            │ │
│  │ z_0 ──▶ post_quant_conv ──▶ Decoder ──▶ image (1, 3, 64, 64)             │ │
│  │                                                                            │ │
│  │ image = (image + 1) / 2  # [-1, 1] → [0, 1]                               │ │
│  │ image = clamp(image, 0, 1)                                                │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Output: 64×64 RGB image                                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Sampling Algorithms

### DDPM vs DDIM 비교

| 특성 | DDPM | DDIM |
|------|------|------|
| 샘플링 방식 | 확률적 (Stochastic) | 결정론적 (Deterministic) |
| 필요 스텝 | 1000 | **50** (또는 더 적게) |
| 속도 | 느림 | **빠름** |
| 재현성 | 매 생성마다 다름 | **같은 시드 = 같은 결과** |
| 품질 | 좋음 | 좋음 (약간 다를 수 있음) |

### DDPM Sampling

```python
def p_sample(self, model, x_t, t, text_embeds, use_cfg=True):
    """
    DDPM Reverse Process: 확률적 역방향 샘플링
    
    수학적 배경:
    p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)
    
    과정:
    1. 노이즈 예측: ε_θ = model(x_t, t, text)
    2. x_0 예측: x̂_0 = (x_t - √(1-ᾱ_t)ε_θ) / √(ᾱ_t)
    3. 사후분포 평균: μ = coef1 × x̂_0 + coef2 × x_t
    4. 노이즈 추가: x_{t-1} = μ + σ_t × z, z ~ N(0, I)
    """
```

```
DDPM 샘플링 시각화:

x_T (noise) ──▶ x_{T-1} ──▶ x_{T-2} ──▶ ... ──▶ x_1 ──▶ x_0 (clean)
    │              │           │                 │         │
    ▼              ▼           ▼                 ▼         ▼
  + noise        + noise    + noise           + noise   (no noise)

특징:
- 매 스텝에서 랜덤 노이즈 추가 (확률적)
- 같은 시작점에서도 다른 결과
- 다양성은 좋지만 재현 어려움
```

### DDIM Sampling

```python
def ddim_sample(self, model, x_t, t, text_embeds, eta=0.0, use_cfg=True):
    """
    DDIM Reverse Process: 결정론적 역방향 샘플링
    
    핵심 차이: eta 파라미터로 확률성 제어
    - eta = 0: 완전 결정론적 (권장)
    - eta = 1: DDPM과 동일
    
    과정:
    1. 노이즈 예측: ε_θ = model(x_t, t, text)
    2. x_0 예측: x̂_0 = (x_t - √(1-α_t)ε_θ) / √(α_t)
    3. σ 계산: σ_t = η × √((1-α_{t-1})/(1-α_t)) × √(1-α_t/α_{t-1})
    4. 방향 계산: dir = √(1 - α_{t-1} - σ²) × ε_θ
    5. 샘플링: x_{t-1} = √(α_{t-1}) × x̂_0 + dir + σ × z
    """
```

```
DDIM 샘플링 시각화 (eta=0):

x_T (noise) ──▶ x_{t_49} ──▶ x_{t_48} ──▶ ... ──▶ x_{t_1} ──▶ x_0 (clean)
    │              │           │                    │           │
    ▼              ▼           ▼                    ▼           ▼
  (skip)       (skip)      (skip)              (skip)      (no skip)

스텝 간격: 1000/50 = 20 timesteps씩 건너뜀
timesteps: [999, 979, 959, ..., 39, 19, 0]

특징:
- eta=0이면 노이즈 추가 없음 (결정론적)
- 1000스텝을 50스텝으로 압축
- 같은 시드면 항상 같은 결과
```

### Timestep 선택

```python
# DDIM timestep 계산
if use_ddim:
    step_indices = torch.linspace(0, num_timesteps - 1, num_steps + 1)
    timesteps = torch.flip(step_indices.long(), dims=[0])[:-1]
    # 예: num_steps=50일 때
    # timesteps = [999, 979, 959, ..., 39, 19, 0] (50개)
else:
    timesteps = torch.arange(num_timesteps - 1, -1, -1)
    # timesteps = [999, 998, 997, ..., 2, 1, 0] (1000개)
```

---

## Classifier-Free Guidance

### CFG 개념

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Classifier-Free Guidance (CFG)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  목적: text 조건을 더 강하게 반영                                              │
│                                                                                 │
│  방법:                                                                          │
│  1. Conditional 예측: ε_cond = model(x_t, t, text_embed)                       │
│  2. Unconditional 예측: ε_uncond = model(x_t, t, uncond_embed)                 │
│  3. Guidance 적용: ε̃ = ε_uncond + s × (ε_cond - ε_uncond)                      │
│                                                                                 │
│  수학적 해석:                                                                   │
│  ε̃ = (1 - s) × ε_uncond + s × ε_cond                                          │
│                                                                                 │
│  s = 1.0: conditional과 동일 (guidance 없음)                                   │
│  s > 1.0: conditional 방향으로 더 강하게 이동                                  │
│  s = 7.5: 권장값 (SD3 기본값)                                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Guidance Scale 효과

```
Guidance Scale (s) 효과:

s = 1.0:
┌─────────────────────────────────────────────────────────────────┐
│ - 조건부 생성만 사용                                            │
│ - text 반영 약함                                                │
│ - 다양성 높음                                                   │
│ - 품질 낮을 수 있음                                             │
└─────────────────────────────────────────────────────────────────┘

s = 3.0:
┌─────────────────────────────────────────────────────────────────┐
│ - 약한 guidance                                                 │
│ - text 어느정도 반영                                            │
│ - 다양성 유지                                                   │
└─────────────────────────────────────────────────────────────────┘

s = 7.5 (권장):
┌─────────────────────────────────────────────────────────────────┐
│ - 적절한 balance                                                │
│ - text 잘 반영                                                  │
│ - 좋은 품질                                                     │
│ - SD3 기본값                                                    │
└─────────────────────────────────────────────────────────────────┘

s = 15.0+:
┌─────────────────────────────────────────────────────────────────┐
│ - 매우 강한 guidance                                            │
│ - text 과하게 반영                                              │
│ - 색상 saturation                                               │
│ - 아티팩트 발생 가능                                            │
└─────────────────────────────────────────────────────────────────┘
```

### CFG 구현

```python
# Inference 시 CFG 적용
def denoise_with_cfg(model, x_t, t, text_embed, uncond_embed, guidance_scale):
    # 1. Conditional 예측
    noise_cond = model(x_t, t, text_embed)
    
    # 2. Unconditional 예측
    noise_uncond = model(x_t, t, uncond_embed)
    
    # 3. CFG 적용
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    
    return noise_pred

"""
계산 비용:
- CFG 사용 시 모델 2번 호출 필요 (cond + uncond)
- 생성 시간 ~2배
- 하지만 품질 향상 효과가 큼
"""
```

---

## Step-by-Step Generation

### 상세 과정

```python
def sample(self, model, shape, text_embeds, num_steps=50, ...):
    """
    전체 샘플링 과정
    """
    B, C, H, W = shape  # (1, 16, 8, 8)
    device = next(model.parameters()).device
    
    # 1. 시드 설정 (재현성)
    if seed is not None:
        torch.manual_seed(seed)
    
    # 2. 순수 노이즈에서 시작
    x_t = torch.randn(shape, device=device)  # z_T
    
    # 3. Timestep 계산
    timesteps = [999, 979, 959, ..., 19, 0]  # 50개
    
    # 4. 반복적 디노이징
    for t in tqdm(timesteps):
        t_batch = torch.full((B,), t, device=device)
        
        # DDIM step
        x_t = ddim_sample(model, x_t, t_batch, text_embeds, eta=0.0)
    
    # 5. VAE 디코딩
    if vae_decoder:
        x_t = vae_decoder.decode_from_latent(x_t)
    
    # 6. 정규화 [0, 1]
    x_t = (x_t + 1.0) / 2.0
    x_t = torch.clamp(x_t, 0.0, 1.0)
    
    return x_t
```

### 각 스텝 시각화

```
Step 0 (t=999): Pure Noise
┌───────────────────────┐
│ ░░▓▓░░▓▓░░▓▓░░▓▓░░   │  SNR ≈ 0.001
│ ▓▓░░▓▓░░▓▓░░▓▓░░▓▓   │  (거의 순수 노이즈)
│ ░░▓▓░░▓▓░░▓▓░░▓▓░░   │
│ ▓▓░░▓▓░░▓▓░░▓▓░░▓▓   │
└───────────────────────┘

Step 10 (t=779): Rough Structure
┌───────────────────────┐
│   ▓▓▓▓▓▓              │  SNR ≈ 0.1
│  ▓▓▓▓▓▓▓▓             │  (대략적 형태 보임)
│ ▓▓▓▓▓▓▓▓▓▓            │
│  ▓▓▓▓▓▓▓▓             │
└───────────────────────┘

Step 25 (t=479): Clear Shape
┌───────────────────────┐
│    ╭───╮              │  SNR ≈ 1.0
│   (• •)               │  (명확한 형태)
│    ╰─╯  /\  /\        │
│     ╰──╯  ╰╯          │
└───────────────────────┘

Step 40 (t=199): Fine Details
┌───────────────────────┐
│    ╭───╮              │  SNR ≈ 10
│   (◉ ◉)               │  (세부 디테일)
│    ╰▽╯  /│\ /│\       │
│  ~~╰──╯~~╰╯~~         │
└───────────────────────┘

Step 50 (t=0): Final Image
┌───────────────────────┐
│   ╭─────╮             │  SNR → ∞
│  (◉   ◉)              │  (완성된 이미지)
│   ╰──▽──╯ /│\ /│\     │
│ ~~~~╰──╯~~~╰╯~~~~     │
│      couch            │
└───────────────────────┘
```

---

## Advanced Options

### CLI 옵션

```bash
uv run main.py --generate \
    --prompt "a cute cat sitting on a couch" \
    --checkpoint checkpoints/diffusion.pt \
    --vae-checkpoint checkpoints/vae.pt \
    --steps 50 \           # 샘플링 스텝 수
    --guidance 7.5 \       # CFG scale
    --seed 42 \            # 재현성 시드
    --num-samples 4 \      # 생성할 이미지 수
    --output output.png    # 출력 파일
```

### 파라미터 설명

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `--steps` | 50 | 10-1000 | 샘플링 스텝 수. 높을수록 품질↑, 속도↓ |
| `--guidance` | 7.5 | 1.0-20.0 | CFG scale. 높을수록 prompt 반영↑ |
| `--seed` | None | int | 재현성 시드. 같은 시드 = 같은 결과 |
| `--num-samples` | 1 | 1-16 | 생성할 이미지 수 |

### Steps vs Quality

```
Steps와 품질의 관계:

Steps = 10:
├── 속도: 매우 빠름
├── 품질: 낮음 (노이즈 잔여)
└── 용도: 빠른 프로토타이핑

Steps = 25:
├── 속도: 빠름
├── 품질: 괜찮음
└── 용도: 일반적 생성

Steps = 50 (권장):
├── 속도: 적당
├── 품질: 좋음
└── 용도: 기본 설정

Steps = 100:
├── 속도: 느림
├── 품질: 매우 좋음
└── 용도: 고품질 생성

Steps = 1000 (DDPM):
├── 속도: 매우 느림
├── 품질: 최고
└── 용도: 연구/비교
```

### Seed 사용법

```python
# 재현 가능한 생성
--seed 42

# 매번 다른 결과
--seed None  # (기본값)

# 여러 변형 생성
for seed in [42, 43, 44, 45]:
    generate(prompt, seed=seed)
```

```
Seed의 역할:

seed=42:
┌─────────────────────────────────────────────────────────────────┐
│ 1. torch.manual_seed(42) 설정                                  │
│ 2. 초기 노이즈 z_T가 고정됨                                    │
│ 3. DDIM (eta=0)이면 전체 과정 결정론적                         │
│ 4. 같은 prompt + seed = 같은 이미지                            │
└─────────────────────────────────────────────────────────────────┘

응용:
- A/B 테스트: 같은 seed로 다른 prompt 비교
- 버그 재현: 문제 발생 시 seed 기록
- 갤러리: 좋은 seed 저장해두기
```

---

## Performance Optimization

### 메모리 최적화

```python
# 1. Inference 시 gradient 비활성화
with torch.no_grad():
    image = generate(prompt)

# 2. Half precision (FP16)
model = model.half()  # 메모리 50% 절약
vae = vae.half()

# 3. VAE slicing (큰 배치)
def decode_with_slicing(vae, latents, slice_size=4):
    images = []
    for i in range(0, len(latents), slice_size):
        batch = latents[i:i+slice_size]
        images.append(vae.decode(batch))
    return torch.cat(images)
```

### 속도 최적화

```python
# 1. 적은 스텝 수
--steps 25  # 대신 50

# 2. torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 3. 배치 생성
--num-samples 4  # 한 번에 4개 생성 (순차 생성보다 효율적)

# 4. EMA 모델 사용 (더 좋은 품질 = 적은 스텝 가능)
ema.apply_shadow()  # EMA weights 적용
```

### GPU 활용

```python
# 자동 디바이스 감지
device = "cuda" if torch.cuda.is_available() else "cpu"

# MPS (Apple Silicon)
if torch.backends.mps.is_available():
    device = "mps"

# 다중 GPU (DataParallel)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

## Interactive Demo

```bash
# 대화형 데모 실행
uv run main.py --demo
```

### Demo 사용법

```
┌─────────────────────────────────────────────────────────────────┐
│                 tiny-stable-diffusion Demo                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Enter prompt (or 'quit' to exit): a beautiful sunset          │
│                                                                 │
│  Generating...                                                  │
│  [████████████████████████████████████████] 50/50               │
│                                                                 │
│  Image saved to: output_0.png                                   │
│                                                                 │
│  Enter prompt (or 'quit' to exit): _                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### 일반적인 문제

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 노이즈가 많은 이미지 | steps 부족 | `--steps 100` 으로 증가 |
| prompt 반영 안 됨 | guidance 낮음 | `--guidance 10.0` 으로 증가 |
| 과포화된 색상 | guidance 높음 | `--guidance 5.0` 으로 감소 |
| CUDA OOM | 메모리 부족 | batch size 줄이기, FP16 사용 |
| 같은 이미지만 나옴 | seed 고정됨 | `--seed` 제거 |

### 디버깅 체크리스트

```
[ ] Checkpoint 파일이 존재하는가?
    → checkpoints/diffusion.pt, checkpoints/vae.pt 확인

[ ] CLIP이 설치되어 있는가?
    → pip install git+https://github.com/openai/CLIP.git

[ ] GPU 메모리가 충분한가?
    → nvidia-smi로 확인, batch size 줄이기

[ ] 모델 크기가 올바른가?
    → checkpoint의 model_config와 현재 설정 비교

[ ] Text encoding이 제대로 되는가?
    → print(text_embeds.shape)로 확인 (B, 512)
```

---

## 코드 예제

### Python에서 직접 사용

```python
from src.inference.generator import generate

# 기본 사용
images = generate(
    prompts=["a cute cat", "a beautiful sunset"],
    checkpoint="checkpoints/diffusion.pt",
    vae_checkpoint="checkpoints/vae.pt",
)

for i, img in enumerate(images):
    img.save(f"output_{i}.png")
```

### 커스텀 파이프라인

```python
import torch
from src.models.vae import create_vae
from src.models.factory import DiT
from src.models.diffusion import Diffusion
from src.text_encoder.clip_encoder import CLIPTextEncoder

# 모델 로딩
device = "cuda"

vae = create_vae()
vae.load_state_dict(torch.load("checkpoints/vae.pt")["model_state_dict"])
vae = vae.to(device).eval()

clip = CLIPTextEncoder().to(device)

dit = DiT(in_channels=16, image_size=8, patch_size=2, model_size="S")
dit.load_state_dict(torch.load("checkpoints/diffusion.pt")["model_state_dict"])
dit = dit.to(device).eval()

# Uncond embedding 계산
uncond_embed = clip.encode([""])

diffusion = Diffusion(
    num_timesteps=1000,
    guidance_scale=7.5,
    uncond_embed=uncond_embed,
)

# 생성
prompt = "a robot playing guitar"
text_embed = clip.encode([prompt])

with torch.no_grad():
    latent = diffusion.sample(
        model=dit,
        shape=(1, 16, 8, 8),
        text_embeds=text_embed,
        num_steps=50,
        use_ddim=True,
        vae_decoder=vae,
        seed=42,
    )

# 이미지 저장
from PIL import Image
import numpy as np

img = latent[0].permute(1, 2, 0).cpu().numpy()
img = (img * 255).astype(np.uint8)
Image.fromarray(img).save("output.png")
```

---

## 참고 자료

- [DDIM Paper](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models
- [CFG Paper](https://arxiv.org/abs/2207.12598) - Classifier-Free Diffusion Guidance
- [Progressive Distillation](https://arxiv.org/abs/2202.00512) - 더 빠른 샘플링
- [Consistency Models](https://arxiv.org/abs/2303.01469) - 1-step 생성
