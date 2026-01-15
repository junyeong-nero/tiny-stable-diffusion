# VAE Training Issue Analysis

> Date: 2025-01-15
> Model: AutoencoderKL (f8 compression, 16 latent channels)
> Dataset: LAION-300k
> Training: 30 epochs

## Summary

VAE reconstruction 성능은 우수하나, latent space에서 random sampling을 통한 이미지 생성이 실패함.

## Problem Analysis

### 1. Latent Space Distribution

학습된 VAE의 latent space 분포를 분석한 결과:

| Metric | Observed | Expected (N(0,1)) | Status |
|--------|----------|-------------------|--------|
| Mean (μ) | -10.5 ~ 9.2 | ≈ 0 | ❌ |
| Std (σ) | **0.002** | ≈ 1.0 | ❌ |
| KL Divergence | 8.37 | < 1 | ❌ |

```
Per-channel Mean range: [-1.818, 1.601]
Per-channel Std range:  [0.0018, 0.0028]
```

### 2. Root Cause: KL Weight Too Low

**Config (문제 설정):**
```yaml
kl_weight: 1.0e-6  # 너무 낮음!
```

KL weight가 `1e-6` (0.000001)으로 설정되어 있어서:

1. **KL loss가 사실상 무시됨**
   ```
   total_loss = recon_loss + 1e-6 * kl_loss
                ~~~~~~~~     ~~~~~~~~~~~~~~
                   ~0.02    +    ~0.000008   ← KL term 무의미
   ```

2. **Encoder가 deterministic하게 학습됨**
   - σ ≈ 0.002 (거의 0)
   - `z = μ + σ * ε` 에서 σ가 0이라 `z ≈ μ`
   - Reparameterization trick이 무력화됨

3. **Mean이 제약 없이 자유롭게 학습됨**
   - KL divergence가 mean을 0으로 끌어당기지 못함
   - Encoder가 임의의 분포로 latent 인코딩

### 3. Visual Representation

```
Encoder 출력 분포 (실제):          Prior 분포 (기대):
      ▲                                 ▲
      │    ██                           │
      │   ████                          │      ████
      │  ██████                         │    ████████
      │ ████████                        │  ████████████
      └──────────────────►              └──────────────────►
         -0.01  0  0.01                    -3  -2  -1  0  1  2  3

   σ ≈ 0.002 (매우 좁음)              σ = 1.0 (넓음)
```

### 4. Why Reconstruction Works but Generation Fails

| Operation | Latent Source | Result |
|-----------|---------------|--------|
| **Reconstruction** | `z = encoder(image)` → `decoder(z)` | ✅ 성공 |
| **Generation** | `z ~ N(0, 1)` → `decoder(z)` | ❌ 실패 |

- Reconstruction: Encoder가 특정 위치(μ)에 정보를 인코딩하고, Decoder가 그 위치에서 디코딩
- Generation: N(0,1)에서 샘플링한 z는 Encoder가 사용하는 영역과 완전히 다름

## Additional Issues Found

### 5. No KL Annealing

학습 초기부터 KL loss를 적용하면:
- Encoder가 제대로 학습되기 전에 latent를 N(0,1)로 강제
- 정보 손실 발생 가능

**권장 패턴:**
```
Epoch 0-10:  kl_weight = 0        (reconstruction만 학습)
Epoch 10-50: kl_weight 점진적 증가
Epoch 50+:   kl_weight = 최종값
```

### 6. Loss Function

현재 MSE loss 사용:
```python
recon_loss = F.mse_loss(reconstruction, x, reduction="mean")
```

이미지 생성에서는 L1 loss나 perceptual loss가 더 효과적일 수 있음.

## Solutions Implemented

### Fix 1: Increase KL Weight

```yaml
# Before
kl_weight: 1.0e-6

# After
kl_weight: 1.0e-3  # 1000배 증가
```

### Fix 2: Cyclical KL Annealing

```yaml
kl_annealing: cyclical
kl_n_cycles: 4
kl_cycle_ratio: 0.5
```

각 cycle에서:
1. **0 → 50%**: β가 0에서 max까지 증가 (reconstruction 먼저 학습)
2. **50 → 100%**: β가 max로 유지 (latent regularization)

```
β (KL weight)
│
│         ████████████       ████████████       ← max_weight
│    ████              ████
│████                 █
└──────────────────────────────────────── step
   Cycle 1            Cycle 2
```

### Fix 3: Latent Statistics Monitoring

Progress bar와 epoch 출력에 latent 통계 추가:
```
VAE Training: loss=0.0234 recon=0.0230 kl=3.42 β=5.0e-04 μ=-0.02 σ=0.892

Epoch 25/100: Loss=0.0234 | β=1.0e-03 | Cycle 1/4 | Latent μ=-0.021, σ=0.892 [target: μ≈0, σ≈1]
```

## Expected Outcome After Fix

학습이 진행되면서:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Mean (μ) | -10 ~ 9 | -0.5 ~ 0.5 |
| Std (σ) | 0.002 | 0.8 ~ 1.2 |
| KL Divergence | 8.37 | 0.5 ~ 2.0 |
| Random Generation | ❌ | ✅ |

## References

- [Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing](https://arxiv.org/abs/1903.10145)
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

## Commits

- `3619d82` - feat: Add VAE latent space monitoring and random latent decode
- `0b9e746` - feat: Add cyclical KL annealing for VAE training
