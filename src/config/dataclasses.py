"""Configuration dataclasses for tiny-stable-diffusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """DiT model configuration."""

    model_size: Literal["S", "B", "L", "XL"] = "S"
    patch_size: int = 2
    image_size: int = 32
    in_channels: int = 3
    clip_embed_dim: int = 512
    num_layers: int = 12
    num_heads: int = 6
    hidden_size: int = 384
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0

    MODEL_CONFIGS: dict = field(
        default_factory=lambda: {
            "S": {"num_layers": 12, "num_heads": 6, "hidden_size": 384, "parameters": "~30M"},
            "B": {"num_layers": 12, "num_heads": 12, "hidden_size": 768, "parameters": "~130M"},
            "L": {"num_layers": 24, "num_heads": 16, "hidden_size": 1024, "parameters": "~300M"},
            "XL": {"num_layers": 28, "num_heads": 16, "hidden_size": 1152, "parameters": "~675M"},
        }
    )

    def __post_init__(self) -> None:
        """Apply model size configuration if model_size is set."""
        if self.model_size in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[self.model_size]
            self.num_layers = config["num_layers"]
            self.num_heads = config["num_heads"]
            self.hidden_size = config["hidden_size"]


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""

    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    linear_variance: bool = False
    epsilon: float = 1e-8
    loss_type: Literal["mse", "kl"] = "mse"
    cfg_probability: float = 0.1
    guidance_scale: float = 7.5


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 5e-4
    min_lr: float = 1e-6
    warmup_steps: int = 100
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    ema_decay: float = 0.9999
    ema_update_freq: int = 1
    use_mixed_precision: bool = True
    checkpoint_interval: int = 10
    save_best: bool = True
    resume_from: str | None = None
    num_validation_samples: int = 4
    validation_prompt: str = "a cute robot"


@dataclass
class DataConfig:
    """Data configuration."""

    source: str = "huggingface"
    dataset_name: str = "junyeong-nero/emoji-32"
    split: str = "train"
    image_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int | None = None
    cache_dir: str = "~/.cache/tiny-stable-diffusion"
    streaming: bool = False
    use_coarse_labels: bool = False
    use_augmentation: bool = True
    flip_prob: float = 0.5
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1


@dataclass
class InferenceConfig:
    """Inference configuration."""

    checkpoint: str = "checkpoints/model.pt"
    num_samples: int = 4
    batch_size: int = 4
    num_steps: int = 50
    use_ddim: bool = True
    guidance_scale: float = 7.5
    seed: int = 42
    output_dir: str = "assets"
    upscale_size: int = 256


@dataclass
class ProjectConfig:
    """Project-wide configuration."""

    name: str = "tiny-stable-diffusion"
    experiment_name: str = "experiment_1"
    seed: int = 42
    device: str = "auto"
    use_tensorboard: bool = True
    use_wandb: bool = False
    output_dir: str = "checkpoints"
    assets_dir: str = "assets"

    def __post_init__(self) -> None:
        """Set device automatically if not specified."""
        if self.device == "auto":
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


def get_default_configs() -> dict:
    """Get default configuration objects.

    Returns:
        Dictionary with all configuration dataclass instances
    """
    return {
        "model": ModelConfig(),
        "diffusion": DiffusionConfig(),
        "training": TrainingConfig(),
        "data": DataConfig(),
        "inference": InferenceConfig(),
        "project": ProjectConfig(),
    }
