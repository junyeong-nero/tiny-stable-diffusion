# Training Pipeline Deep Dive

> 이 문서는 tiny-stable-diffusion의 훈련 파이프라인을 상세하게 설명합니다.

## 목차

1. [Two-Stage Training Overview](#two-stage-training-overview)
2. [Stage 1: VAE Training](#stage-1-vae-training)
3. [Stage 2: Diffusion Training](#stage-2-diffusion-training)
4. [Optimizer & Scheduler](#optimizer--scheduler)
5. [EMA (Exponential Moving Average)](#ema-exponential-moving-average)
6. [Checkpointing](#checkpointing)
7. [HuggingFace Hub Upload](#huggingface-hub-upload)

---

## Two-Stage Training Overview

tiny-stable-diffusion은 두 단계로 훈련됩니다.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Two-Stage Training Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ Stage 1: VAE Training                                                      │ │
│  │                                                                            │ │
│  │ 목표: 이미지를 latent space로 압축하고 복원하는 법 학습                    │ │
│  │                                                                            │ │
│  │ Image (64×64) ──▶ Encoder ──▶ Latent (8×8×16) ──▶ Decoder ──▶ Image       │ │
│  │                                                                            │ │
│  │ Loss = MSE(input, reconstruction) + β × KL(posterior || prior)            │ │
│  │                                                                            │ │
│  │ 출력: checkpoints/vae.pt                                                   │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                            │
│                                    ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ Stage 2: Diffusion Training                                               │ │
│  │                                                                            │ │
│  │ 목표: latent space에서 text 조건에 맞는 이미지 생성법 학습                 │ │
│  │                                                                            │ │
│  │ Image ──▶ [Frozen VAE] ──▶ Latent ──▶ Add Noise ──▶ DiT ──▶ Predict Noise │ │
│  │                                           ▲                                │ │
│  │                               Text ──▶ CLIP ──┘                           │ │
│  │                                                                            │ │
│  │ Loss = MSE(predicted_noise, actual_noise) × Min-SNR weight                │ │
│  │                                                                            │ │
│  │ 출력: checkpoints/diffusion.pt                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 왜 Two-Stage인가?

| 접근 방식 | 장점 | 단점 |
|-----------|------|------|
| End-to-End | 단순함 | 매우 느림, 불안정 |
| **Two-Stage** | 효율적, 안정적 | 두 번 훈련 필요 |

Two-Stage 장점:
1. **Latent space가 먼저 잘 정의됨** - Diffusion이 의미있는 공간에서 작동
2. **각 단계 독립적 디버깅 가능** - 문제 발생 시 원인 파악 용이
3. **VAE 재사용 가능** - 다른 diffusion 모델에도 활용

---

## Stage 1: VAE Training

> 파일 위치: `src/training/vae_trainer.py`

### Training Loop 상세

```python
def train_vae(config, use_wandb=False):
    """
    VAE Training Main Loop
    
    단계:
    1. 데이터셋 로딩
    2. 모델 초기화
    3. Optimizer/Scheduler 설정
    4. 에폭별 훈련
    5. 체크포인트 저장
    6. 샘플 생성 (검증)
    """
```

#### Step-by-Step Training

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         VAE Training Step                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. 배치 로딩                                                                   │
│     images = next(dataloader)  # (B, 3, 64, 64), range [-1, 1]                 │
│                                                                                 │
│  2. Forward Pass                                                                │
│     reconstruction, mean, logvar = vae(images)                                 │
│                                                                                 │
│     내부 과정:                                                                  │
│     ┌──────────────────────────────────────────────────────────┐               │
│     │ h = encoder(images)           # (B, 32, 8, 8)            │               │
│     │ mean, logvar = split(h)       # 각각 (B, 16, 8, 8)       │               │
│     │ z = mean + exp(0.5*logvar) * ε  # reparameterization    │               │
│     │ reconstruction = decoder(z)    # (B, 3, 64, 64)          │               │
│     └──────────────────────────────────────────────────────────┘               │
│                                                                                 │
│  3. Loss 계산                                                                   │
│     ┌──────────────────────────────────────────────────────────┐               │
│     │ recon_loss = MSE(images, reconstruction)                 │               │
│     │            = mean((images - reconstruction)²)            │               │
│     │                                                          │               │
│     │ kl_loss = -0.5 * mean(1 + logvar - mean² - exp(logvar)) │               │
│     │                                                          │               │
│     │ total_loss = recon_loss + kl_weight * kl_loss           │               │
│     │            = recon_loss + 1e-6 * kl_loss                │               │
│     └──────────────────────────────────────────────────────────┘               │
│                                                                                 │
│  4. Backward & Update                                                           │
│     optimizer.zero_grad()                                                       │
│     loss.backward()                                                             │
│     optimizer.step()                                                            │
│     scheduler.step()                                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### VAE Loss 분석

#### Reconstruction Loss

```python
# Mean Squared Error
recon_loss = F.mse_loss(reconstruction, x, reduction="mean")

"""
목적: 입력 이미지와 복원 이미지가 최대한 유사하도록

계산:
L_recon = (1/N) × Σ (x_i - x̂_i)²

여기서:
- N = batch_size × 3 × 64 × 64
- x_i: 원본 픽셀
- x̂_i: 복원 픽셀

값 범위: 0 ~ 4 (입력이 [-1, 1] 범위일 때)
좋은 값: < 0.01 (거의 완벽한 복원)
"""
```

#### KL Divergence Loss

```python
# KL(q(z|x) || p(z)) where p(z) = N(0, I)
kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

"""
목적: latent distribution이 표준 정규분포에 가깝도록 regularization

유도:
q(z|x) = N(μ, σ²)
p(z) = N(0, 1)

KL(q||p) = ∫ q(z) log(q(z)/p(z)) dz
         = 0.5 × (μ² + σ² - 1 - log(σ²))
         = -0.5 × (1 + log(σ²) - μ² - σ²)

참고: logvar = log(σ²) 이므로 exp(logvar) = σ²

역할:
- μ → 0으로: 중심이 원점에
- σ → 1로: 분산이 단위 분산에
- 이를 통해 sampling 시 z ~ N(0, I)가 의미있는 이미지 생성
"""
```

#### KL Weight (β-VAE)

```python
kl_weight = 1e-6  # β in β-VAE

"""
Trade-off:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  kl_weight ↑ (e.g., 1e-3)                                      │
│  ├── latent space 더 regularized                               │
│  ├── z ~ N(0, I)에 더 가까워짐                                 │
│  └── BUT: reconstruction 품질 저하                             │
│                                                                 │
│  kl_weight ↓ (e.g., 1e-8)                                      │
│  ├── reconstruction 더 정확                                     │
│  ├── latent space 구조 약함                                     │
│  └── BUT: posterior collapse 위험 (decoder가 z 무시)           │
│                                                                 │
│  권장값: 1e-6                                                   │
│  ├── 적절한 reconstruction 품질                                 │
│  └── 적절한 latent structure                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
```

### Mixed Precision Training

```python
use_amp = config.get("mixed_precision", False) and device.type == "cuda"
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# Training step with AMP
with torch.cuda.amp.autocast(enabled=use_amp):
    reconstruction, mean, logvar = model(images)
    loss, loss_dict = model.training_loss(images, kl_weight)

if use_amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()

"""
Mixed Precision (FP16/BF16):
- Forward pass: FP16 연산 (메모리/속도 2배 향상)
- Backward pass: FP32 gradients (정확도 유지)
- GradScaler: FP16 underflow 방지

효과:
- 메모리 사용량 ~50% 감소
- 학습 속도 ~1.5-2x 향상
- 품질 거의 동일
"""
```

---

## Stage 2: Diffusion Training

> 파일 위치: `src/training/trainer.py`

### Training Loop 상세

```python
def train_diffusion(config, use_wandb=False):
    """
    Diffusion Training Main Loop
    
    필수 조건:
    - 사전 훈련된 VAE checkpoint 필요
    - VAE는 frozen (학습 안 함)
    
    단계:
    1. VAE 로딩 & Freezing
    2. CLIP 텍스트 인코더 로딩
    3. DiT/MMDiT 초기화
    4. Diffusion 프로세스 초기화
    5. 에폭별 훈련
    """
```

#### Step-by-Step Training

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Diffusion Training Step                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. 배치 로딩                                                                   │
│     images, captions = next(dataloader)                                        │
│     # images: (B, 3, 64, 64), range [-1, 1]                                   │
│     # captions: list of strings                                                │
│                                                                                 │
│  2. VAE Encoding (frozen, no grad)                                             │
│     with torch.no_grad():                                                       │
│         latents = vae.encode_to_latent(images)  # (B, 16, 8, 8)               │
│                                                                                 │
│  3. Text Encoding (frozen, no grad)                                            │
│     with torch.no_grad():                                                       │
│         text_embeds = clip.encode(captions)  # (B, 512)                       │
│                                                                                 │
│  4. Random Timestep Sampling                                                    │
│     t = torch.randint(0, 1000, (B,))  # uniform random                        │
│                                                                                 │
│  5. Add Noise (Forward Diffusion)                                              │
│     ┌──────────────────────────────────────────────────────────┐               │
│     │ noise = torch.randn_like(latents)                        │               │
│     │                                                          │               │
│     │ noisy_latents = √(ᾱ_t) × latents + √(1-ᾱ_t) × noise    │               │
│     │                                                          │               │
│     │ # t가 크면: noise 비중 ↑                                 │               │
│     │ # t가 작으면: latent 비중 ↑                              │               │
│     └──────────────────────────────────────────────────────────┘               │
│                                                                                 │
│  6. CFG Dropout (Training 시)                                                  │
│     ┌──────────────────────────────────────────────────────────┐               │
│     │ for i in range(B):                                       │               │
│     │     if random() < cfg_probability:  # e.g., 10%          │               │
│     │         text_embeds[i] = uncond_embed  # empty string ""│               │
│     │                                                          │               │
│     │ # 이렇게 하면 모델이 unconditional 생성도 학습           │               │
│     │ # Inference 시 CFG 적용 가능해짐                         │               │
│     └──────────────────────────────────────────────────────────┘               │
│                                                                                 │
│  7. Noise Prediction                                                            │
│     predicted_noise = dit(noisy_latents, t, text_embeds)  # (B, 16, 8, 8)     │
│                                                                                 │
│  8. Loss 계산 (Min-SNR weighted MSE)                                           │
│     ┌──────────────────────────────────────────────────────────┐               │
│     │ mse = mean((predicted_noise - noise)², dim=[1,2,3])     │               │
│     │      # Per-sample MSE: (B,)                              │               │
│     │                                                          │               │
│     │ # Min-SNR weighting                                      │               │
│     │ snr = ᾱ_t / (1 - ᾱ_t)                                   │               │
│     │ weight = min(snr, γ) / snr  # γ=5.0                     │               │
│     │                                                          │               │
│     │ loss = mean(mse × weight)                               │               │
│     └──────────────────────────────────────────────────────────┘               │
│                                                                                 │
│  9. Backward & Update                                                           │
│     optimizer.zero_grad()                                                       │
│     loss.backward()                                                             │
│     optimizer.step()                                                            │
│     scheduler.step()                                                            │
│                                                                                 │
│  10. EMA Update                                                                 │
│      if use_ema:                                                                │
│          ema.update()  # θ_ema = decay × θ_ema + (1-decay) × θ                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### CFG Warmup

```python
# CFG probability를 점진적으로 증가
initial_cfg = 0.0   # 시작: unconditional dropout 없음
final_cfg = 0.1     # 최종: 10% dropout
cfg_warmup_epochs = 10

for epoch in range(epochs):
    if epoch < cfg_warmup_epochs:
        progress = epoch / cfg_warmup_epochs
        cfg_prob = initial_cfg + (final_cfg - initial_cfg) * progress
    else:
        cfg_prob = final_cfg

"""
CFG Warmup 이유:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  초기 (epoch 0-10):                                            │
│  ├── 모델이 아직 불안정                                        │
│  ├── text-image 연결 학습 중                                   │
│  └── CFG dropout 적으면 더 안정적                              │
│                                                                 │
│  후기 (epoch 10+):                                             │
│  ├── 모델이 안정화됨                                           │
│  ├── CFG dropout으로 robust해짐                                │
│  └── Inference 시 CFG 효과 향상                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
```

### Validation Sample Generation

```python
def generate_samples(model, diffusion, clip_encoder, vae_decoder, prompts, ...):
    """
    훈련 중 검증 샘플 생성
    
    매 N 에폭마다:
    1. 고정된 validation prompts 사용
    2. DDIM 샘플링으로 이미지 생성
    3. 이미지 저장 (samples/epoch_N/)
    
    목적:
    - 학습 진행 시각적 확인
    - text-image alignment 체크
    - Overfitting/mode collapse 감지
    """
```

```
Validation Prompts 예시:
┌─────────────────────────────────────────────────────────────────┐
│ validation_prompts:                                             │
│   - "a photo of a cat"                                         │
│   - "a rocket flying in space"                                 │
│   - "a robot with blue eyes"                                   │
│   - "a beautiful sunset over the ocean"                        │
│   - "a red sports car"                                         │
└─────────────────────────────────────────────────────────────────┘

생성 과정:
1. 각 prompt를 CLIP으로 인코딩
2. Random noise (B, 16, 8, 8) 시작
3. DDIM 50 steps로 denoising
4. VAE decoder로 이미지 변환
5. 저장: samples/epoch_N/00_a_photo_of_a_cat.png
```

---

## Optimizer & Scheduler

### AdamW Optimizer

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],  # e.g., 1e-4
    betas=(0.9, 0.999),          # momentum coefficients
    eps=1e-8,                    # numerical stability
    weight_decay=0.0,            # no L2 regularization
)

"""
AdamW 선택 이유:
- Adam + decoupled weight decay
- Transformer 계열 모델에 표준
- 안정적인 학습

하이퍼파라미터:
- lr: 1e-4 (VAE/Diffusion 모두)
- betas: (0.9, 0.999) - 표준값
- weight_decay: 0.0 - DiT 논문 따름
"""
```

### Cosine Annealing with Warmup

```python
# Steps 계산
num_steps_per_epoch = len(dataloader)
total_steps = epochs * num_steps_per_epoch
warmup_steps = total_steps // 20  # 5% warmup

def lr_lambda(step):
    if step < warmup_steps:
        # Linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

```
Learning Rate Schedule:

lr
│
│     ╱─────────────────────────────╮
│    ╱                               ╲
│   ╱   Cosine Annealing              ╲
│  ╱                                   ╲
│ ╱                                     ╲
├╱                                       ╲────▶ step
│◄──────►│◄─────────────────────────────────────►
   warmup              decay
   (5%)                (95%)

Phase 1: Linear Warmup (0 → 5%)
- lr: 0 → target_lr
- 목적: 학습 초기 안정화

Phase 2: Cosine Annealing (5% → 100%)
- lr: target_lr → 0 (cosine curve)
- 목적: 부드러운 learning rate decay
```

---

## EMA (Exponential Moving Average)

> 파일 위치: `src/training/ema.py`

### EMA 개념

```python
class EMA:
    """
    Exponential Moving Average of model parameters
    
    수식:
        θ_ema = decay × θ_ema + (1 - decay) × θ
        
    여기서:
        - θ: 현재 모델 파라미터
        - θ_ema: EMA 파라미터
        - decay: 보통 0.9999
    """
```

### 왜 EMA를 사용하는가?

```
┌─────────────────────────────────────────────────────────────────┐
│                      EMA 효과                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training Parameters (θ):                                       │
│  ├── 매 step마다 업데이트                                       │
│  ├── 노이즈가 있을 수 있음                                      │
│  └── 최신 gradient 반영                                         │
│                                                                 │
│  EMA Parameters (θ_ema):                                        │
│  ├── 여러 step의 평균                                           │
│  ├── 더 안정적                                                  │
│  └── 보통 더 좋은 성능                                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ θ_ema = 0.9999 × θ_ema + 0.0001 × θ                    │   │
│  │                                                         │   │
│  │ → θ_ema는 약 10,000 step의 weighted average            │   │
│  │ → 1 / (1 - 0.9999) ≈ 10,000                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Inference 시: θ_ema 사용 → 더 좋은 품질                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### EMA 구현

```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 초기화: 현재 파라미터 복사
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """매 training step 후 호출"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # θ_ema = decay × θ_ema + (1 - decay) × θ
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_((1 - self.decay) * param.data)
    
    def apply_shadow(self):
        """Inference 전: EMA 파라미터 적용"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Inference 후: 원래 파라미터 복원"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
```

---

## Checkpointing

> 파일 위치: `src/training/checkpoint.py`

### Checkpoint 구조

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, config, ema=None):
    """
    체크포인트 저장
    
    저장 내용:
    {
        "epoch": int,              # 현재 에폭
        "loss": float,             # 현재 loss
        "model_state_dict": dict,  # 모델 weights
        "optimizer_state_dict": dict,  # optimizer state
        "scheduler_state_dict": dict,  # scheduler state
        "model_config": {          # 모델 설정
            "model_size": "S",
            "patch_size": 2,
            "image_size": 32,
            "model_type": "dit",
            "qk_rmsnorm": True,
            "register_tokens": 0,
        },
        "ema_state_dict": dict,    # EMA weights (optional)
    }
    """
```

### Best Model 저장

```python
# 매 에폭 후
if avg_loss < best_loss:
    best_loss = avg_loss
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        loss=avg_loss,
        path=checkpoint_path,  # e.g., checkpoints/diffusion.pt
        config=config,
        ema=ema,
    )

"""
저장 전략:
- Best loss 기준으로만 저장
- 디스크 공간 절약
- 가장 좋은 모델만 유지

대안:
- 매 N 에폭마다 저장
- 마지막 K개 유지
- 별도 디렉토리에 epoch별 저장
"""
```

### Checkpoint 로딩

```python
def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    체크포인트 로딩
    
    사용 사례:
    1. 훈련 재개
    2. Inference용 모델 로딩
    3. Fine-tuning 시작점
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint["epoch"], checkpoint["loss"]
```

---

## HuggingFace Hub Upload

> 파일 위치: `src/utils/hf_upload.py`

### 기능 개요

```python
def push_to_hub(checkpoint_path, repo_id, model_type, config=None, ...):
    """
    훈련된 모델을 HuggingFace Hub에 업로드
    
    업로드 내용:
    1. {model_type}.pt - 체크포인트 파일
    2. config.json - 훈련 설정
    3. README.md - 모델 카드 (자동 생성)
    
    사용법:
    uv run main.py --train-vae --push-to-hub --hub-model-id username/my-vae
    """
```

### CLI 인자

```bash
# HuggingFace Hub 업로드 옵션
--push-to-hub           # 업로드 활성화
--hub-model-id ID       # 레포지토리 ID (필수)
--hub-private           # 비공개 레포지토리
```

### 업로드 과정

```
┌─────────────────────────────────────────────────────────────────┐
│                   HuggingFace Hub Upload                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 훈련 완료                                                   │
│     checkpoints/vae.pt 생성됨                                   │
│                                                                 │
│  2. push_to_hub() 호출                                          │
│     ├── HF API 인증 (HF_TOKEN 환경변수 또는 huggingface-cli)   │
│     ├── 레포지토리 생성 (없으면)                                │
│     └── 파일 업로드                                             │
│                                                                 │
│  3. 업로드 파일 구조                                            │
│     username/my-vae/                                            │
│     ├── vae.pt          # 체크포인트                            │
│     ├── config.json     # 훈련 설정                             │
│     └── README.md       # 모델 카드                             │
│                                                                 │
│  4. 결과                                                        │
│     https://huggingface.co/username/my-vae                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 환경 설정

```bash
# 방법 1: 환경변수
export HF_TOKEN=hf_xxxxxxxxxxxxx

# 방법 2: CLI 로그인
huggingface-cli login

# 방법 3: config.yaml
hub:
    push_to_hub: true
    model_id: username/model-name
    private: false
```

---

## 훈련 팁

### 권장 하이퍼파라미터

| 단계 | 파라미터 | VAE | Diffusion |
|------|----------|-----|-----------|
| Epochs | `epochs` | 100 | 200 |
| Batch Size | `batch_size` | 32-128 | 32 |
| Learning Rate | `learning_rate` | 4e-4 | 1e-4 |
| KL Weight | `kl_weight` | 1e-6 | - |
| CFG Prob | `cfg_probability` | - | 0.1 |
| EMA Decay | `ema_decay` | - | 0.9999 |

### 메모리 최적화

```python
# 1. Gradient Accumulation
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. Mixed Precision
config["mixed_precision"] = True

# 3. Smaller Batch Size
config["batch_size"] = 16  # GPU 메모리 부족 시

# 4. Gradient Checkpointing (구현 시)
model.gradient_checkpointing_enable()
```

### 디버깅 체크리스트

```
[ ] VAE reconstruction이 제대로 되는가?
    → samples/vae_epoch_N/ 이미지 확인
    
[ ] KL loss가 발산하지 않는가?
    → kl_loss < 100 정도가 정상
    
[ ] Diffusion loss가 감소하는가?
    → 초기 ~1.0 → 최종 ~0.1 정도
    
[ ] 생성 이미지가 prompt와 관련있는가?
    → samples/epoch_N/ 확인
    
[ ] NaN/Inf loss 발생하지 않는가?
    → learning rate 낮추기
```

---

## 참고 자료

- [β-VAE Paper](https://openreview.net/forum?id=Sy2fzU9gl) - Understanding disentangling in β-VAE
- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [AdamW Paper](https://arxiv.org/abs/1711.05101) - Decoupled Weight Decay Regularization
- [EMA in Diffusion](https://arxiv.org/abs/2102.09672) - Improved Denoising Diffusion Probabilistic Models
