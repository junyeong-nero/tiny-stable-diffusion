# tiny-stable-diffusion GIF 생성 확장 프로젝트 계획

## 프로젝트 개요

기존 tiny-stable-diffusion에서 학습된 **VAE**와 **MMDiT/DiT** 모델을 최대한 재활용하여, AnimateDiff 방식의 **Motion Module**을 추가해 GIF/애니메이션을 생성하는 기능을 구현한다.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         아키텍처 개요                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Text Prompt                                                           │
│       │                                                                 │
│       ▼                                                                 │
│   ┌──────────┐                                                          │
│   │   CLIP   │ (기존)                                                   │
│   └────┬─────┘                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │              Latent Diffusion (시간 축 확장)                  │      │
│   │  ┌────────────────────────────────────────────────────────┐  │      │
│   │  │   z_t: (B, F, C, H, W)  ←  F = num_frames (16~32)      │  │      │
│   │  └────────────────────────────────────────────────────────┘  │      │
│   │                          │                                   │      │
│   │                          ▼                                   │      │
│   │  ┌──────────────────────────────────────────────────────┐    │      │
│   │  │  MMDiT/DiT (frozen) + Motion Module (trainable)      │    │      │
│   │  │  ┌─────────────────┐  ┌───────────────────────────┐  │    │      │
│   │  │  │ Spatial Attn    │→ │ Temporal Attn (NEW)       │  │    │      │
│   │  │  │ (기존, frozen)   │  │ (frames 간 attention)     │  │    │      │
│   │  │  └─────────────────┘  └───────────────────────────┘  │    │      │
│   │  └──────────────────────────────────────────────────────┘    │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                          │                                              │
│                          ▼                                              │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │   VAE Decoder (기존, frozen)                                 │      │
│   │   z → (B, F, 3, 64, 64) → GIF                                │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Motion Module 설계 및 구현

### 1.1 TemporalTransformerBlock 구현

**위치**: `src/models/motion.py` (신규)

```python
class TemporalTransformerBlock(nn.Module):
    """프레임 간 temporal attention을 수행하는 블록"""
    
    def __init__(
        self,
        hidden_size: int,        # MMDiT hidden size와 동일
        num_heads: int = 8,
        num_frames: int = 16,
        dropout: float = 0.0,
    ):
        # Temporal self-attention
        # Position embedding for frames
        # Feed-forward network
```

**핵심 구현 사항**:
- **입력 reshape**: `(B*F, N, D)` → `(B*N, F, D)` (spatial → temporal)
- **Sinusoidal position embedding**: 프레임 순서 인코딩
- **Residual connection**: `output = x + temporal_attn(x)`
- **Zero-initialized output projection**: 학습 초기 안정성

### 1.2 MotionModule 구현

**위치**: `src/models/motion.py`

```python
class MotionModule(nn.Module):
    """MMDiT 레이어 뒤에 삽입되는 motion module"""
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,      # temporal transformer 레이어 수
        num_heads: int = 8,
        num_frames: int = 16,
    ):
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(...)
            for _ in range(num_layers)
        ])
```

### 1.3 AnimatedMMDiT 래퍼 구현

**위치**: `src/models/animated_mmdit.py` (신규)

```python
class AnimatedMMDiT(nn.Module):
    """기존 MMDiT에 Motion Module을 주입한 래퍼"""
    
    def __init__(
        self,
        base_model: MMDiT,           # 기존 학습된 모델 (frozen)
        motion_module: MotionModule, # 새로 학습할 모듈
        num_frames: int = 16,
    ):
        self.base_model = base_model
        self.motion_module = motion_module
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
```

**forward 로직**:
1. 입력: `(B, F, C, H, W)` → `(B*F, C, H, W)`로 reshape
2. 기존 MMDiT의 각 레이어 실행
3. 각 레이어 출력에 Motion Module 적용
4. 출력: `(B*F, C, H, W)` → `(B, F, C, H, W)`로 reshape

---

## Phase 2: 데이터셋 및 데이터로더

### 2.1 비디오/GIF 데이터셋 클래스

**위치**: `src/data/video_dataset.py` (신규)

```python
class VideoDataset(Dataset):
    """짧은 비디오 클립을 로드하는 데이터셋"""
    
    def __init__(
        self,
        dataset_name: str,        # HuggingFace 데이터셋 또는 로컬 경로
        num_frames: int = 16,
        frame_skip: int = 1,      # 프레임 샘플링 간격
        image_size: int = 64,
    ):
        pass
```

### 2.2 추천 데이터셋

| 데이터셋 | 규모 | 특징 | HuggingFace ID |
|---------|------|------|----------------|
| WebVid-10M | 10M clips | 다양한 카테고리, 캡션 포함 | `TempoFunk/webvid-10M-subset` |
| Pexels Videos | ~100K | 고품질, 짧은 클립 | 직접 수집 필요 |
| UCF101 | 13K | 액션 인식용, 다양한 동작 | `sayakpaul/ucf101-subset` |

### 2.3 데이터 전처리 파이프라인

```python
def preprocess_video(video: torch.Tensor) -> torch.Tensor:
    """비디오 전처리
    
    1. 균일한 프레임 샘플링 (16 or 32 frames)
    2. 중앙 크롭 + 리사이즈 (64x64)
    3. 정규화 [-1, 1]
    4. VAE 인코딩하여 latent 생성
    """
    pass
```

---

## Phase 3: 학습 파이프라인

### 3.1 AnimatedDiffusion 클래스

**위치**: `src/models/animated_diffusion.py` (신규)

```python
class AnimatedDiffusion(Diffusion):
    """시간 축이 추가된 Rectified Flow diffusion"""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        num_frames: int = 16,
        **kwargs,
    ):
        super().__init__(num_timesteps=num_timesteps, **kwargs)
        self.num_frames = num_frames
    
    def q_sample(
        self,
        x_0: torch.Tensor,  # (B, F, C, H, W)
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """시간 축을 포함한 forward process"""
        # 모든 프레임에 동일한 timestep 적용
        # 또는 프레임별 다른 noise level (progressive)
        pass
```

### 3.2 MotionTrainer 클래스

**위치**: `src/training/motion_trainer.py` (신규)

```python
class MotionTrainer:
    """Motion Module만 학습하는 trainer"""
    
    def __init__(
        self,
        model: AnimatedMMDiT,
        vae: AutoencoderKL,      # frozen
        diffusion: AnimatedDiffusion,
        optimizer: torch.optim.Optimizer,
        config: MotionTrainConfig,
    ):
        pass
    
    def train_step(self, batch: dict) -> dict:
        """
        1. 비디오 → VAE encoder → latents (B, F, C, H, W)
        2. 랜덤 timestep 샘플링
        3. Noise 추가 (q_sample)
        4. 모델 forward (velocity prediction)
        5. Loss 계산 (MSE)
        6. Motion module만 gradient 업데이트
        """
        pass
```

### 3.3 config.yaml 확장

```yaml
# Stage 3: Motion Module Training
motion_train:
  # Model
  base_checkpoint: checkpoints/diffusion.pt  # 기존 MMDiT
  vae_checkpoint: checkpoints/vae.pt
  
  # Motion Module Architecture
  motion_num_layers: 2
  motion_num_heads: 8
  num_frames: 16
  
  # Dataset
  dataset_name: TempoFunk/webvid-10M-subset
  video_field: mp4
  caption_field: txt
  frame_skip: 2
  
  # Training
  epochs: 100
  batch_size: 8                # 메모리 제약으로 작게
  learning_rate: 1.0e-4
  gradient_accumulation: 4     # effective batch = 32
  
  # Checkpoint
  checkpoint_path: checkpoints/motion.pt
```

---

## Phase 4: 추론 및 GIF 생성

### 4.1 AnimationGenerator 클래스

**위치**: `src/inference/animation_generator.py` (신규)

```python
class AnimationGenerator:
    """텍스트 프롬프트로 GIF 생성"""
    
    def __init__(
        self,
        model: AnimatedMMDiT,
        vae: AutoencoderKL,
        text_encoder: CLIPTextEncoder,
        diffusion: AnimatedDiffusion,
    ):
        pass
    
    def generate(
        self,
        prompt: str,
        num_frames: int = 16,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        fps: int = 8,
        seed: int | None = None,
    ) -> list[PIL.Image.Image]:
        """
        1. Text → CLIP embedding
        2. Random noise (B, F, C, H, W)
        3. Euler sampling (Rectified Flow)
        4. VAE decode → 이미지 시퀀스
        5. 반환: PIL Image 리스트
        """
        pass
    
    def save_gif(
        self,
        frames: list[PIL.Image.Image],
        path: str,
        fps: int = 8,
        loop: bool = True,
    ) -> None:
        """PIL 이미지 리스트를 GIF로 저장"""
        pass
```

### 4.2 프레임 보간 (선택적 개선)

**위치**: `src/inference/interpolation.py` (신규)

RIFE 또는 FILM을 활용한 프레임 보간으로 부드러운 GIF 생성:

```python
class FrameInterpolator:
    """AI 기반 프레임 보간"""
    
    def __init__(self, method: Literal["rife", "film"] = "rife"):
        # 사전학습된 보간 모델 로드
        pass
    
    def interpolate(
        self,
        frames: list[PIL.Image.Image],
        factor: int = 2,  # 2x, 4x 등
    ) -> list[PIL.Image.Image]:
        """프레임 수를 factor배 증가"""
        pass
```

---

## Phase 5: CLI 및 데모

### 5.1 main.py 확장

```bash
# Motion Module 학습
uv run main.py --train-motion --epochs 100 --batch-size 8

# GIF 생성
uv run main.py --generate-gif --prompt "a cat walking" --frames 16 --fps 8

# Interactive 데모
uv run main.py --demo-gif
```

### 5.2 Gradio 데모 확장

**위치**: `src/demo/app.py` 수정

```python
def create_animation_demo():
    """GIF 생성 Gradio 인터페이스"""
    with gr.Blocks() as demo:
        prompt = gr.Textbox(label="Prompt")
        num_frames = gr.Slider(8, 32, value=16, label="Frames")
        fps = gr.Slider(4, 24, value=8, label="FPS")
        
        generate_btn = gr.Button("Generate GIF")
        output_gif = gr.Image(label="Generated GIF")
        
    return demo
```

---

## 구현 순서 및 마일스톤

### 마일스톤 1: 핵심 모듈 구현 ✅ 완료
- [x] `src/models/motion.py` - TemporalTransformerBlock, MotionModule
- [x] `src/models/animated_mmdit.py` - AnimatedMMDiT 래퍼
- [x] 단위 테스트: `tests/test_motion.py` (25개 테스트 통과)

### 마일스톤 2: 데이터 파이프라인 ✅ 완료
- [x] `src/data/video_dataset.py` - VideoDataset, GIFDataset, SyntheticVideoDataset
- [x] `src/data/video_transforms.py` - VideoTransform, TemporalAugmentation, frame sampling
- [x] 단위 테스트: `tests/test_video_dataset.py` (27개 테스트 통과)

### 마일스톤 3: 학습 파이프라인 ✅ 완료
- [x] `src/models/animated_diffusion.py` - AnimatedDiffusion (시간 축 확장)
- [x] `src/training/motion_trainer.py` - train_motion, MotionTrainConfig
- [x] `config.yaml` 확장 (motion_train 섹션 추가)
- [x] 단위 테스트: `tests/test_animated_diffusion.py` (12개 테스트 통과)

### 마일스톤 4: 추론 및 데모
- [x] `src/inference/animation_generator.py` - GIF 생성 (AnimationGenerator 클래스)
- [ ] `src/inference/interpolation.py` - 프레임 보간 (선택 - 미구현)
- [x] `main.py` CLI 확장 (--train-motion, --generate-gif, --animation-demo)
- [x] `src/demo/app.py` Streamlit 데모 (GIF 생성 페이지 추가)
- [x] 단위 테스트: `tests/test_animation_generator.py` (11개 테스트 통과)

### 마일스톤 5: 최적화 및 배포
- [x] Mixed precision 지원 (`motion_trainer.py`에서 AMP 사용)
- [x] Gradient checkpointing (메모리 최적화) - `MotionModule`, `AnimatedMMDiT`
- [x] HuggingFace Hub 업로드 (`src/utils/hf_upload.py`에 motion 타입 추가)
- [x] 문서화 (`docs/extensions/MotionModule.md` 전면 업데이트)

---

## 예상 리소스 요구사항

| 항목 | 요구사항 |
|------|----------|
| GPU VRAM | 16GB+ (16 frames, batch=4) |
| 학습 시간 | ~24-48시간 (A100, WebVid 1M subset) |
| 디스크 | 50GB+ (비디오 데이터셋) |
| 메모리 | 32GB+ RAM |

### 메모리 최적화 전략

1. **Gradient Checkpointing**: Motion Module 레이어에 적용
2. **VAE Slicing**: 프레임별 VAE 인코딩/디코딩
3. **Mixed Precision (FP16)**: 메모리 50% 절감
4. **Gradient Accumulation**: 작은 batch로 큰 effective batch

---

## 기존 코드 재활용 요약

| 컴포넌트 | 파일 | 재활용 방식 |
|----------|------|-------------|
| VAE (AutoencoderKL) | `src/models/vae.py` | Frozen, 그대로 사용 |
| MMDiT | `src/models/mmdit.py` | Frozen, AnimatedMMDiT 래퍼로 감싸서 사용 |
| Diffusion | `src/models/diffusion.py` | 상속하여 시간 축 확장 |
| CLIP Encoder | `src/text_encoder/clip_encoder.py` | 그대로 사용 |
| EMA | `src/training/ema.py` | 그대로 사용 |
| Checkpoint Manager | `src/training/checkpoint.py` | 그대로 사용 |
| Config Loader | `src/config/loader.py` | dataclass 추가 |

---

## 위험 요소 및 대응

| 위험 | 가능성 | 영향 | 대응 |
|------|--------|------|------|
| 메모리 부족 | 높음 | 높음 | Gradient checkpointing, frame 수 감소 |
| 낮은 품질 | 중간 | 중간 | 더 큰 데이터셋, 더 긴 학습 |
| 시간적 일관성 부족 | 중간 | 높음 | Temporal attention 레이어 증가 |
| 느린 학습 | 낮음 | 중간 | LR 스케줄러 튜닝 |

---

## References

### 핵심 논문

1. **AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning**
   - Authors: Yuwei Guo et al. (Shanghai AI Lab)
   - arXiv: https://arxiv.org/abs/2307.04725
   - 핵심 아이디어: Motion Module을 기존 SD 모델에 플러그인 방식으로 추가

2. **Pix2Gif: Motion-Guided Diffusion for GIF Generation**
   - arXiv: https://arxiv.org/abs/2403.04634
   - 핵심 아이디어: Image-to-GIF 변환, motion magnitude로 움직임 제어

3. **Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets**
   - Authors: Stability AI
   - arXiv: https://arxiv.org/abs/2311.15127
   - 핵심 아이디어: 대규모 비디오 학습, temporal layers

4. **RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation**
   - ECCV 2022
   - GitHub: https://github.com/hzwer/ECCV2022-RIFE
   - 핵심 아이디어: 실시간 프레임 보간

5. **FILM: Frame Interpolation for Large Motion**
   - ECCV 2022
   - GitHub: https://github.com/google-research/frame-interpolation
   - 핵심 아이디어: 큰 움직임에서도 안정적인 보간

### 오픈소스 구현

1. **HuggingFace Diffusers - AnimateDiff**
   - Docs: https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff
   - 코드: https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/unets/unet_motion_model.py

2. **sd-webui-animatediff**
   - GitHub: https://github.com/continue-revolution/sd-webui-animatediff
   - AUTOMATIC1111 WebUI 확장

3. **Giffusion**
   - GitHub: https://github.com/DN6/giffusion
   - 키프레임 보간 기반 GIF 생성

4. **Deforum**
   - GitHub: https://github.com/deforum-art/deforum-stable-diffusion
   - 카메라 모션 기반 애니메이션

### 데이터셋

1. **WebVid-10M**
   - 규모: 10.7M 비디오-텍스트 쌍
   - 링크: https://github.com/m-bain/webvid

2. **Pexels Videos**
   - 링크: https://www.pexels.com/videos/
   - 고품질, 무료 사용

3. **UCF101**
   - 규모: 13,320 비디오
   - 링크: https://www.crcv.ucf.edu/data/UCF101.php

### 튜토리얼 및 가이드

1. **AnimateDiff Tutorial (Stable Diffusion Art)**
   - https://stable-diffusion-art.com/animatediff/

2. **Beginner's Guide to AnimateDiff (Aituts)**
   - https://aituts.com/animatediff/

3. **ComfyUI AnimateDiff Workflow**
   - https://www.runcomfy.com/tutorials/how-to-use-animatediff-to-create-ai-animations-in-comfyui
