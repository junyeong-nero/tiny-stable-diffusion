# Architecture Deep Dive

> 이 문서는 tiny-stable-diffusion의 모델 아키텍처를 상세하게 설명합니다.

## 목차

1. [전체 시스템 개요](#전체-시스템-개요)
2. [VAE (Variational AutoEncoder)](#vae-variational-autoencoder)
3. [Diffusion Process](#diffusion-process)
4. [DiT (Diffusion Transformer)](#dit-diffusion-transformer)
5. [MMDiT (Multi-Modal DiT)](#mmdit-multi-modal-dit)
6. [Text Encoder (CLIP)](#text-encoder-clip)

---

## 전체 시스템 개요

tiny-stable-diffusion은 Stable Diffusion 3의 아키텍처를 따르는 교육용 구현체입니다.

### 핵심 구성 요소

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

### 데이터 흐름

| 단계 | 입력 | 출력 | 차원 변화 |
|------|------|------|-----------|
| 1. Image Input | RGB Image | - | `(B, 3, 64, 64)` |
| 2. VAE Encode | Image | Latent | `(B, 3, 64, 64)` → `(B, 16, 8, 8)` |
| 3. Add Noise | Clean Latent | Noisy Latent | `(B, 16, 8, 8)` → `(B, 16, 8, 8)` |
| 4. DiT Predict | Noisy Latent + Text | Predicted Noise | `(B, 16, 8, 8)` → `(B, 16, 8, 8)` |
| 5. Denoise | Predicted Noise | Clean Latent | `(B, 16, 8, 8)` → `(B, 16, 8, 8)` |
| 6. VAE Decode | Clean Latent | Output Image | `(B, 16, 8, 8)` → `(B, 3, 64, 64)` |

---

## VAE (Variational AutoEncoder)

> 파일 위치: `src/models/vae.py`

VAE는 이미지를 저차원 잠재 공간(latent space)으로 압축하고 복원하는 역할을 합니다.

### 왜 Latent Space Diffusion인가?

| 항목 | Pixel Space | Latent Space |
|------|-------------|--------------|
| 차원 | 64×64×3 = **12,288** | 8×8×16 = **1,024** |
| 연산량 | 매우 높음 | **12배 효율적** |
| 메모리 | 높음 | **낮음** |
| 학습 속도 | 느림 | **빠름** |

### AutoencoderKL 구조

```python
class AutoencoderKL:
    """
    SD3 스타일 Variational AutoEncoder
    
    핵심 특징:
    - z_channels: 16 (SD3 표준)
    - f8 compression: 64×64 → 8×8
    - KL divergence regularization
    """
```

#### Encoder 상세 구조

```
Input Image (B, 3, 64, 64)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Conv3×3(3→64)                                                   │
│    Output: (B, 64, 64, 64)                                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 0: ResBlock×2(64→64) + Downsample                        │
│    ResBlock: GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3 │
│    Downsample: Conv3×3(stride=2)                               │
│    Output: (B, 64, 32, 32)                                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: ResBlock×2(64→128) + Downsample                       │
│    Output: (B, 128, 16, 16)                                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 2: ResBlock×2(128→256) + Downsample                      │
│    Output: (B, 256, 8, 8)                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 3: ResBlock×2(256→256) (NO Downsample - bottleneck)      │
│    Output: (B, 256, 8, 8)                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Middle Block:                                                   │
│    ResBlock(256→256) → AttnBlock(256) → ResBlock(256→256)      │
│                                                                 │
│    AttnBlock (Self-Attention):                                  │
│    - Query, Key, Value projections (1×1 Conv)                  │
│    - Dot-product attention with softmax                        │
│    - Output projection + residual connection                   │
│    Output: (B, 256, 8, 8)                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Output Layer:                                                   │
│    GroupNorm → SiLU → Conv3×3(256→32)                          │
│    Output: (B, 32, 8, 8) = [mean, logvar] 각각 16채널          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    mean (B, 16, 8, 8), logvar (B, 16, 8, 8)
```

#### Decoder 상세 구조

```
Latent z (B, 16, 8, 8)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Input Conv: Conv3×3(16→256)                                    │
│    Output: (B, 256, 8, 8)                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Middle Block:                                                   │
│    ResBlock(256→256) → AttnBlock(256) → ResBlock(256→256)      │
│    Output: (B, 256, 8, 8)                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 3: ResBlock×3(256→256)                                   │
│    Output: (B, 256, 8, 8)                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 2: ResBlock×3(256→256) + Upsample                        │
│    Upsample: Nearest Interpolation ×2 → Conv3×3                │
│    Output: (B, 256, 16, 16)                                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: ResBlock×3(256→128) + Upsample                        │
│    Output: (B, 128, 32, 32)                                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 0: ResBlock×3(128→64) + Upsample                         │
│    Output: (B, 64, 64, 64)                                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Output Layer:                                                   │
│    GroupNorm → SiLU → Conv3×3(64→3)                            │
│    Output: (B, 3, 64, 64) - Reconstructed Image                │
└─────────────────────────────────────────────────────────────────┘
```

#### ResnetBlock 상세

```python
class ResnetBlock(nn.Module):
    """
    ResNet Block: VAE의 기본 구성 요소
    
    구조: (GroupNorm → SiLU → Conv3×3) × 2 + Skip Connection
    
    수식:
        h = SiLU(GroupNorm(x))
        h = Conv3×3(h)
        h = SiLU(GroupNorm(h))
        h = Dropout(h)
        h = Conv3×3(h)
        output = h + shortcut(x)
    """
```

#### AttnBlock (Self-Attention)

```python
class AttnBlock(nn.Module):
    """
    Self-Attention for capturing global context
    
    수식:
        Q = W_q(x), K = W_k(x), V = W_v(x)
        Attention = softmax(Q @ K^T / sqrt(d))
        output = x + proj(Attention @ V)
    
    위치: Encoder/Decoder의 bottleneck (8×8 해상도)에서만 사용
    이유: 작은 해상도에서 global context 파악이 효율적
    """
```

### VAE Loss Function

```python
def training_loss(self, x, kl_weight=1e-6):
    """
    VAE Training Loss = Reconstruction Loss + β × KL Divergence
    
    1. Reconstruction Loss (MSE):
       L_recon = ||x - x̂||²
       
    2. KL Divergence Loss:
       L_KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
       
    3. Total Loss:
       L = L_recon + β × L_KL
       
    β (kl_weight) 권장값: 1e-6
    - 너무 크면: reconstruction 품질 저하
    - 너무 작으면: posterior collapse 위험
    """
```

### Reparameterization Trick

```python
def reparameterize(self, mean, logvar):
    """
    Reparameterization Trick for backpropagation through sampling
    
    문제: z ~ N(μ, σ²) 샘플링은 미분 불가능
    해결: z = μ + σ × ε, where ε ~ N(0, 1)
    
    코드:
        std = exp(0.5 × logvar)  # σ = exp(log(σ²)/2)
        eps = randn_like(std)    # ε ~ N(0, 1)
        z = mean + eps × std     # z = μ + ε × σ
    """
```

---

## Diffusion Process

> 파일 위치: `src/models/diffusion.py`

Diffusion은 노이즈 추가(forward)와 제거(reverse) 과정을 정의합니다.

### Forward Process (q-sample)

이미지에 점진적으로 노이즈를 추가합니다.

```
x_0 (clean) ──▶ x_1 ──▶ x_2 ──▶ ... ──▶ x_T (pure noise)
```

#### 수학적 정의

```
q(x_t | x_0) = N(x_t; √(ᾱ_t) × x_0, (1 - ᾱ_t) × I)

여기서:
- α_t = 1 - β_t (noise schedule에서 정의)
- ᾱ_t = ∏_{s=1}^{t} α_s (cumulative product)
- β_t = noise schedule (cosine 사용)

Closed-form sampling:
x_t = √(ᾱ_t) × x_0 + √(1 - ᾱ_t) × ε
where ε ~ N(0, I)
```

#### Cosine Beta Schedule

```python
def _cosine_beta_schedule(self, s=0.008):
    """
    Improved DDPM의 cosine schedule
    
    ᾱ_t = cos²((t/T + s) / (1+s) × π/2)
    
    장점:
    - Linear schedule보다 더 부드러운 노이즈 추가
    - 중간 timestep에서 더 많은 정보 보존
    - 학습 안정성 향상
    """
```

```
Noise Level (β_t) over Timesteps:

β_t
│
│    ╭────────────────────╮
│   ╱                      ╲
│  ╱     Cosine Schedule    ╲
│ ╱                          ╲
├╱────────────────────────────╲──────▶ t
0                             T
        (1000 timesteps)

Comparison with Linear:
- Linear: 직선적으로 증가
- Cosine: S-curve 형태로 부드럽게 증가
```

### Reverse Process (Denoising)

#### DDPM (Stochastic)

```python
def p_sample(self, model, x_t, t, text_embeds):
    """
    DDPM reverse: 확률적 역방향 샘플링
    
    p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)
    
    1. Predict noise: ε_θ = model(x_t, t, text)
    
    2. Predict x_0:
       x̂_0 = (x_t - √(1-ᾱ_t) × ε_θ) / √(ᾱ_t)
       x̂_0 = clamp(x̂_0, -1, 1)  # stability
    
    3. Compute posterior mean:
       μ = coef1 × x̂_0 + coef2 × x_t
       where:
         coef1 = β_t × √(ᾱ_{t-1}) / (1 - ᾱ_t)
         coef2 = (1 - ᾱ_{t-1}) × √(α_t) / (1 - ᾱ_t)
    
    4. Add noise (except at t=0):
       x_{t-1} = μ + σ_t × z, where z ~ N(0, I)
    """
```

#### DDIM (Deterministic)

```python
def ddim_sample(self, model, x_t, t, text_embeds, eta=0.0):
    """
    DDIM reverse: 결정론적 샘플링 (eta=0일 때)
    
    장점:
    - 더 적은 step으로 샘플링 가능 (1000 → 50)
    - eta=0: 완전 결정론적 (같은 시드 = 같은 결과)
    - eta=1: DDPM과 동일
    
    수식:
    1. Predict x_0:
       x̂_0 = (x_t - √(1-α_t) × ε_θ) / √(α_t)
    
    2. Compute σ (variance):
       σ_t = η × √((1-α_{t-1})/(1-α_t)) × √(1-α_t/α_{t-1})
    
    3. Compute direction:
       dir = √(1 - α_{t-1} - σ²) × ε_θ
    
    4. Sample:
       x_{t-1} = √(α_{t-1}) × x̂_0 + dir + σ × z
    """
```

### Classifier-Free Guidance (CFG)

```python
# Training: 일정 확률로 text condition을 drop
if random() < cfg_probability:  # e.g., 10%
    text_embed = uncond_embed  # empty string ""

# Inference: conditional과 unconditional prediction 결합
noise_pred = uncond_pred + guidance_scale × (cond_pred - uncond_pred)
```

```
Guidance Scale 효과:

scale=1.0: 조건 없는 생성과 동일
scale=3.0: 약한 text 반영
scale=7.5: 권장값 (SD3 기본값)
scale=15+: 과도한 saturation 발생 가능

수식:
ε̃ = ε_uncond + s × (ε_cond - ε_uncond)
  = (1-s) × ε_uncond + s × ε_cond

s > 1이면 conditional 방향으로 더 강하게 이동
```

### Min-SNR Weighting

```python
def _get_min_snr_weights(self, timesteps):
    """
    Min-SNR loss weighting for stable training
    
    문제: 
    - 낮은 noise level (t 작을 때): SNR 높음 → 쉬운 task
    - 높은 noise level (t 클 때): SNR 낮음 → 어려운 task
    - 기본 MSE는 모든 timestep에 동일 가중치
    
    해결:
    weight(t) = min(SNR(t), γ) / SNR(t)
    
    SNR(t) = ᾱ_t / (1 - ᾱ_t)
    γ = 5.0 (default)
    
    효과:
    - 높은 SNR (쉬운 task) timestep 가중치 감소
    - 학습 안정성 향상
    """
```

---

## DiT (Diffusion Transformer)

> 파일 위치: `src/models/vanilla_dit.py`

Vanilla DiT는 Cross-Attention을 사용하여 text conditioning을 수행합니다.

### 전체 구조

```
Input Latent (B, 16, 8, 8)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Patch Embedding: Conv(16→384, kernel=2, stride=2)              │
│    Input: (B, 16, 8, 8)                                        │
│    Output: (B, 16, 384) - 16 patches = (8/2)²                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Position Embedding: Learned positional embeddings               │
│    pos_embed: (1, 16, 384) - learnable parameters              │
│    Output: x + pos_embed                                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│ Timestep Embedding  │              │ Text Embedding      │
│                     │              │                     │
│ t (scalar)          │              │ "a cute cat"        │
│    │                │              │    │                │
│    ▼                │              │    ▼                │
│ Sinusoidal Embed    │              │ CLIP Encoder        │
│    │                │              │    │                │
│    ▼                │              │    ▼                │
│ MLP (384→384)       │              │ Linear (512→384)    │
│    │                │              │    │                │
│    ▼                │              │    ▼                │
│ time_cond (B, 384)  │              │ text_cond (B, 1, 384)│
└─────────────────────┘              └─────────────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AdaLN-Zero                                 │
│                                                                 │
│ time_cond → MLP → [shift, scale, gate] × 2 × num_layers        │
│                                                                 │
│ 각 layer마다 6개 파라미터:                                      │
│ - shift_msa, scale_msa, gate_msa (Self-Attention용)            │
│ - shift_mlp, scale_mlp, gate_mlp (MLP용)                       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DiT Block × 12 (Size S)                      │
│                                                                 │
│  x ──▶ LayerNorm ──▶ modulate(shift, scale) ──▶ Self-Attention │
│  │                                              │               │
│  │◀────────────────── gate × ───────────────────┘               │
│  │                                                              │
│  └──▶ LayerNorm ──▶ Cross-Attention(with text_cond) ──────────▶│
│  │                                              │               │
│  │◀─────────────────────────────────────────────┘               │
│  │                                                              │
│  └──▶ LayerNorm ──▶ modulate(shift, scale) ──▶ MLP             │
│  │                                              │               │
│  │◀────────────────── gate × ───────────────────┘               │
│  │                                                              │
│  output                                                         │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Final Layer:                                                    │
│    LayerNorm → modulate(time_cond) → Linear(384 → 2×2×16)      │
│    Unpatchify: (B, 16, 64) → (B, 16, 8, 8)                     │
│                                                                 │
│ Output: Predicted Noise (B, 16, 8, 8)                          │
└─────────────────────────────────────────────────────────────────┘
```

### DiT Block 상세

```python
class DiTBlock(nn.Module):
    """
    DiT Transformer Block with Cross-Attention
    
    구조:
    1. Self-Attention (image tokens끼리)
       - AdaLN-Zero modulation 적용
       - x = x + gate × SelfAttn(modulate(LayerNorm(x)))
       
    2. Cross-Attention (image ← text)
       - text conditioning 주입
       - x = x + CrossAttn(LayerNorm(x), text_embeds)
       
    3. MLP
       - AdaLN-Zero modulation 적용
       - x = x + gate × MLP(modulate(LayerNorm(x)))
    """
```

#### AdaLN-Zero Modulation

```python
def modulate(x, shift, scale):
    """
    Adaptive Layer Normalization
    
    일반 LayerNorm:
        y = (x - μ) / σ × γ + β
        
    AdaLN:
        y = (x - μ) / σ × (1 + scale) + shift
        
    AdaLN-Zero:
        - scale, shift가 timestep에서 동적으로 생성
        - gate도 추가되어 residual connection 제어
        - 모든 modulation 파라미터 zero-initialized
    
    효과:
        - 초기: 모든 block이 identity mapping
        - 학습: 점진적으로 의미있는 transformation 학습
    """
    return x * (1 + scale) + shift
```

### 모델 크기별 설정

```python
MODEL_CONFIGS = {
    "S":  {"layers": 12, "heads": 6,  "hidden": 384},   # ~40M params
    "B":  {"layers": 12, "heads": 12, "hidden": 768},   # ~160M params
    "L":  {"layers": 24, "heads": 16, "hidden": 1024},  # ~560M params
    "XL": {"layers": 28, "heads": 16, "hidden": 1152},  # ~820M params
}
```

---

## MMDiT (Multi-Modal DiT)

> 파일 위치: `src/models/mmdit.py`

MMDiT는 SD3에서 사용하는 아키텍처로, Joint Attention을 사용합니다.

### DiT vs MMDiT 비교

```
┌─────────────────────────────────────────────────────────────────┐
│                        VanillaDiT                               │
│                                                                 │
│  Image Tokens ──▶ Self-Attention ──▶ Cross-Attention ──▶ MLP   │
│                                           ▲                     │
│                                           │                     │
│  Text Tokens ─────────────────────────────┘                     │
│                    (separate processing)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          MMDiT                                  │
│                                                                 │
│  ┌─────────────────────────────────────────┐                   │
│  │ [Text Tokens ; Image Tokens]            │                   │
│  │         (concatenated)                  │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│               Joint Self-Attention                              │
│           (text와 image가 서로 attend)                          │
│                       │                                         │
│                       ▼                                         │
│                    Split                                        │
│                   ╱      ╲                                      │
│                  ╱        ╲                                     │
│          Text MLP      Image MLP                                │
│          (separate)    (separate)                               │
└─────────────────────────────────────────────────────────────────┘
```

### MMDiT 장점

| 특성 | VanillaDiT | MMDiT |
|------|-----------|-------|
| Text-Image Interaction | Cross-Attention (단방향) | Joint Attention (양방향) |
| 정보 흐름 | Image ← Text | Image ↔ Text |
| 안정성 | 좋음 | **더 좋음** (QK-RMSNorm) |
| SD3 호환 | X | **O** |

### QK-RMSNorm

```python
# MMDiT의 안정성 향상 기법
class QKRMSNorm:
    """
    Query와 Key에 RMSNorm 적용
    
    일반 Attention:
        attn = softmax(Q @ K^T / √d)
        
    QK-RMSNorm:
        Q' = RMSNorm(Q)
        K' = RMSNorm(K)
        attn = softmax(Q' @ K'^T / √d)
        
    효과:
        - attention logit의 scale 안정화
        - 깊은 네트워크에서 gradient 안정성 향상
        - 학습 초기 발산 방지
    """
```

### Register Tokens

```python
# 추가 learnable tokens (선택적)
register_tokens: int = 0

"""
Register Tokens:
- 학습 가능한 추가 토큰
- image/text 토큰과 함께 attention에 참여
- global context 저장소 역할
- Vision Transformer에서 효과 입증됨

사용:
- register_tokens=4: 4개의 추가 토큰
- 실험적 기능, 기본값은 0
"""
```

---

## Text Encoder (CLIP)

> 파일 위치: `src/text_encoder/clip_encoder.py`

CLIP을 사용하여 text prompt를 embedding으로 변환합니다.

### CLIP 구조

```
Text Prompt: "a cute cat sitting on a couch"
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Tokenization (CLIP Tokenizer)                                   │
│    - BPE (Byte Pair Encoding) 사용                             │
│    - Max length: 77 tokens                                      │
│    - [SOT] text tokens [EOT] [PAD]...                          │
│    Output: token_ids (1, 77)                                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ CLIP Text Encoder (ViT-B/32)                                   │
│    - 12 Transformer layers                                      │
│    - Hidden size: 512                                          │
│    - Attention heads: 8                                        │
│                                                                 │
│    Token Embedding → Positional Embedding → Transformer        │
│                                                                 │
│    Output: text_features (1, 512) - [EOT] token의 hidden state │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ L2 Normalization                                                │
│    text_embed = text_features / ||text_features||_2            │
│                                                                 │
│    Output: normalized embedding (1, 512)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 실제 SD3와의 차이

| 항목 | Stable Diffusion 3 | tiny-stable-diffusion |
|------|-------------------|----------------------|
| Text Encoder | T5-XXL + CLIP-G + CLIP-L | **CLIP ViT-B/32** |
| Embedding Dim | 4096 + 1280 + 768 | **512** |
| Parameters | ~5B | **~63M** |
| 목적 | Production | **Education** |

### Unconditional Embedding

```python
# CFG를 위한 unconditional embedding 생성
uncond_embed = clip_encoder.encode([""])  # empty string

"""
역할:
- CFG에서 "조건 없는" 예측을 위해 사용
- Training: cfg_probability로 text를 uncond_embed로 대체
- Inference: guidance_scale로 cond와 uncond 예측 조합

empty string ""의 embedding이 unconditional 역할:
- CLIP은 빈 문자열도 valid embedding 생성
- 이 embedding은 "특정 조건 없음"을 의미
"""
```

---

## 파라미터 수 요약

### VAE (ch=64, ch_mult=[1,2,4,4])

| 컴포넌트 | Parameters |
|----------|------------|
| Encoder | ~10.5M |
| Decoder | ~10.5M |
| **Total** | **~21M** |

### DiT-S (Vanilla)

| 컴포넌트 | Parameters |
|----------|------------|
| Patch Embed | 6K |
| Position Embed | 6K |
| Timestep Embed | 295K |
| Text Projection | 197K |
| AdaLN-Zero | 1.77M |
| DiT Blocks (×12) | 35.5M |
| Final Layer | 2.1M |
| **Total** | **~40M** |

### MMDiT-S

| 컴포넌트 | Parameters |
|----------|------------|
| Patch Embed | 6K |
| Position Embed | 6K |
| Timestep Embed | 295K |
| MMDiT Core | 84.5M |
| Final Layer | 2.1M |
| **Total** | **~87M** |

### 전체 시스템

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
- [Min-SNR Paper](https://arxiv.org/abs/2303.09556) - Efficient Diffusion Training via Min-SNR
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models
