"""Centralized configuration and hyperparameters for PixMoji-Diffusion."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """DiT model configuration."""

    # Model size: S (small), B (base), L (large), XL (extra large)
    model_size: Literal["S", "B", "L", "XL"] = "S"
    # Patch size for image tokenization (2, 4, or 8)
    patch_size: int = 2
    # Input image resolution
    image_size: int = 32
    # Number of input channels (RGB = 3)
    in_channels: int = 3
    # CLIP text embedding dimension
    clip_embed_dim: int = 512
    # Number of DiT blocks (layers)
    num_layers: int = 12
    # Attention heads
    num_heads: int = 6
    # Hidden dimension
    hidden_size: int = 384
    # MLP hidden dimension multiplier
    mlp_ratio: float = 4.0
    # Attention dropout rate
    attn_dropout: float = 0.0
    # MLP dropout rate
    mlp_dropout: float = 0.0

    # Model size configurations
    MODEL_CONFIGS: dict = field(
        default_factory=lambda: {
            "S": {
                "num_layers": 12,
                "num_heads": 6,
                "hidden_size": 384,
                "parameters": "~30M",
            },
            "B": {
                "num_layers": 12,
                "num_heads": 12,
                "hidden_size": 768,
                "parameters": "~130M",
            },
            "L": {
                "num_layers": 24,
                "num_heads": 16,
                "hidden_size": 1024,
                "parameters": "~300M",
            },
            "XL": {
                "num_layers": 28,
                "num_heads": 16,
                "hidden_size": 1152,
                "parameters": "~675M",
            },
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

    # Number of diffusion timesteps
    num_timesteps: int = 1000
    # Beta schedule type: "linear", "cosine", "quadratic"
    beta_schedule: str = "cosine"
    # Start value for beta (if linear/quadratic)
    beta_start: float = 0.0001
    # End value for beta (if linear/quadratic)
    beta_end: float = 0.02
    # Use linear variance schedule (True) or cosine (False)
    linear_variance: bool = False
    # Small epsilon for numerical stability
    epsilon: float = 1e-8

    # Training loss type
    loss_type: Literal["mse", "kl"] = "mse"

    # Classifier-Free Guidance settings
    # Probability of dropping text condition during training
    cfg_probability: float = 0.1
    # Guidance scale during inference (typically 4.0-8.0)
    guidance_scale: float = 7.5


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Batch size
    batch_size: int = 64
    # Number of training epochs
    epochs: int = 100
    # Learning rate (increased from 1e-4 for better convergence)
    learning_rate: float = 5e-4
    # Minimum learning rate for cosine scheduler
    min_lr: float = 1e-6
    # Warmup steps (reduced for small datasets - ~1 epoch worth)
    warmup_steps: int = 100
    # Optimizer betas (AdamW)
    betas: tuple = (0.9, 0.999)
    # Optimizer epsilon
    eps: float = 1e-8
    # Weight decay
    weight_decay: float = 0.01
    # Gradient clipping value (0 = no clipping)
    gradient_clip_val: float = 1.0
    # EMA decay rate (0 = no EMA)
    ema_decay: float = 0.9999
    # EMA update frequency
    ema_update_freq: int = 1

    # Mixed precision training
    use_mixed_precision: bool = True

    # Checkpoint and logging
    checkpoint_interval: int = 10
    # Save best checkpoint based on loss
    save_best: bool = True
    # Resume from checkpoint
    resume_from: str | None = None

    # Number of validation samples to generate
    num_validation_samples: int = 4
    # Validation prompt
    validation_prompt: str = "a cute robot"


@dataclass
class DataConfig:
    """Data configuration.

    Uses junyeong-nero/emoji-32 dataset from Hugging Face.
    Images are already 32x32 RGB, no resizing needed.
    """

    # Dataset source: "huggingface" or "local"
    source: str = "huggingface"
    # Hugging Face dataset name
    dataset_name: str = "junyeong-nero/emoji-32"
    # Dataset split (train, validation, test)
    split: str = "train"
    # Image resolution (no resize needed for emoji-32)
    image_size: int = 32
    # Number of data loading workers (0 = main process)
    num_workers: int = 0
    # Pin memory for DataLoader
    pin_memory: bool = True
    # Prefetch factor for DataLoader
    prefetch_factor: int | None = None
    # Cache directory for Hugging Face datasets
    cache_dir: str = "~/.cache/pixmoji"
    # Use streaming mode for large datasets
    streaming: bool = False

    # Data augmentation
    use_augmentation: bool = True
    # Random horizontal flip probability
    flip_prob: float = 0.5
    # Color jitter brightness
    brightness: float = 0.1
    # Color jitter contrast
    contrast: float = 0.1
    # Color jitter saturation
    saturation: float = 0.1


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # Checkpoint path
    checkpoint: str = "checkpoints/model.pt"
    # Number of samples to generate
    num_samples: int = 4
    # Batch size for generation
    batch_size: int = 4
    # Diffusion steps (DDPM: 1000, DDIM: 50-100)
    num_steps: int = 50
    # Use DDIM sampling (True) or DDPM (False)
    use_ddim: bool = True
    # Guidance scale
    guidance_scale: float = 7.5
    # Random seed for reproducibility
    seed: int = 42
    # Output directory
    output_dir: str = "assets"
    # Save at higher resolution (nearest neighbor upsampling)
    upscale_size: int = 256


@dataclass
class ProjectConfig:
    """Project-wide configuration."""

    # Project name
    name: str = "pixmoji-diffusion"
    # Experiment name (for logging)
    experiment_name: str = "experiment_1"
    # Random seed
    seed: int = 42
    # Device: "auto", "cuda", "mps", "cpu"
    device: str = "auto"
    # Enable TensorBoard logging
    use_tensorboard: bool = True
    # Enable Weights & Biases logging
    use_wandb: bool = False
    # Project directory for outputs
    output_dir: str = "checkpoints"
    # Assets directory
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


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser for command-line interface."""
    parser = argparse.ArgumentParser(description="PixMoji-Diffusion: Text-to-Pixel Art Generator")

    # Project arguments
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )

    # Model arguments
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["S", "B", "L", "XL"],
        default="S",
        help="DiT model size",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=2,
        help="Patch size for image tokenization",
    )

    # Diffusion arguments
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Dataset directory",
    )

    # Inference arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Sampling steps",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=True,
        help="Use DDIM sampling",
    )
    # Note: --seed is defined in project arguments section

    return parser


def get_config(args: argparse.Namespace | None = None) -> dict:
    """Get configuration from arguments or defaults.

    Args:
        args: Command-line arguments (optional)

    Returns:
        Dictionary with all configuration sections
    """
    if args is not None:
        # Override defaults with command-line arguments
        model_config = ModelConfig(
            model_size=getattr(args, "model_size", "S") or "S",
            patch_size=getattr(args, "patch_size", 2) or 2,
        )
        diffusion_config = DiffusionConfig(
            num_timesteps=getattr(args, "num_timesteps", 1000) or 1000,
            guidance_scale=getattr(args, "guidance_scale", 7.5) or 7.5,
        )
        training_config = TrainingConfig(
            epochs=getattr(args, "epochs", 100) or 100,
            batch_size=getattr(args, "batch_size", 64) or 64,
            learning_rate=getattr(args, "learning_rate", 5e-4) or 5e-4,
            resume_from=getattr(args, "resume", None),
        )
        data_config = DataConfig(
            source=getattr(args, "data_source", "huggingface") or "huggingface",
            dataset_name=getattr(args, "dataset_name", "junyeong-nero/emoji-32")
            or "junyeong-nero/emoji-32",
            split=getattr(args, "split", "train") or "train",
            streaming=getattr(args, "streaming", False) or False,
        )
        inference_config = InferenceConfig(
            checkpoint=getattr(args, "checkpoint", None),
            num_samples=getattr(args, "num_samples", 4) or 4,
            num_steps=getattr(args, "steps", 50) or 50,
            guidance_scale=getattr(args, "guidance_scale", 7.5) or 7.5,
            seed=getattr(args, "seed", 42) or 42,
        )
        project_config = ProjectConfig(
            name=getattr(args, "name", None) or "pixmoji-diffusion",
            seed=getattr(args, "seed", 42) or 42,
            device=getattr(args, "device", "auto") or "auto",
            output_dir=getattr(args, "output_dir", "checkpoints") or "checkpoints",
        )
    else:
        model_config = ModelConfig()
        diffusion_config = DiffusionConfig()
        training_config = TrainingConfig()
        data_config = DataConfig()
        inference_config = InferenceConfig()
        project_config = ProjectConfig()

    return {
        "model": model_config,
        "diffusion": diffusion_config,
        "training": training_config,
        "data": data_config,
        "inference": inference_config,
        "project": project_config,
    }


def print_config(config: dict) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("PixMoji-Diffusion Configuration")
    print("=" * 60)

    for section, cfg in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in cfg.__dict__.items():
            if not key.startswith("_"):
                print(f"  {key}: {value}")

    print("=" * 60 + "\n")
