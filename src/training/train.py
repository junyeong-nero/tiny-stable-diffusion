#!/usr/bin/env python3
"""PixMoji-Diffusion Training Script.

Two-stage training pipeline:
- Stage 1: Pretrain on CIFAR-100 with text conditioning
- Stage 2: Fine-tune on emoji dataset

Usage:
    python train.py

Configure training settings in the CONFIG section below.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from local modules
from src.config import DataConfig, DiffusionConfig, ModelConfig, TrainingConfig
from src.data.dataset import CIFAR100Dataset, EmojiDataset
from src.models.diffusion import Diffusion
from src.models.dit import DiT
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.ema import EMA


# =============================================================================
# CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Training Stage: "pretrain" or "finetune"
# -----------------------------------------------------------------------------
TRAINING_STAGE: str = "pretrain"  # "pretrain" or "finetune"

# -----------------------------------------------------------------------------
# Pretraining Settings (Stage 1)
# -----------------------------------------------------------------------------
PRETRAIN_CONFIG: dict[str, Any] = {
    "data_source": "cifar100",
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "image_size": 32,
    # CFG (Classifier-Free Guidance) settings
    "initial_cfg_prob": 0.0,  # Start with 0 (unconditional)
    "final_cfg_prob": 0.1,  # End with 0.1 (10% dropout)
    "cfg_warmup_epochs": 20,  # Warmup over 20 epochs
    # Checkpoint
    "checkpoint_path": "checkpoints/pretrain_cifar100.pt",
}

# -----------------------------------------------------------------------------
# Fine-tuning Settings (Stage 2)
# -----------------------------------------------------------------------------
FINETUNE_CONFIG: dict[str, Any] = {
    "data_source": "huggingface",
    "dataset_name": "junyeong-nero/emoji-32",
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "image_size": 32,
    # CFG settings (keep higher for fine-tuning)
    "cfg_prob": 0.1,
    # Load pretrained weights
    "pretrain_checkpoint": "checkpoints/pretrain_cifar100.pt",
    "reset_cross_attention": True,  # Reset cross-attention for fine-tuning
}

# -----------------------------------------------------------------------------
# Common Settings
# -----------------------------------------------------------------------------
COMMON_CONFIG: dict[str, Any] = {
    "model_size": "S",  # DiT model size: "XS", "S", "B", "L", "XL"
    "patch_size": 2,
    "num_timesteps": 1000,
    "beta_schedule": "cosine",
    "guidance_scale": 7.5,
    "use_ema": True,
    "ema_decay": 0.9999,
    "mixed_precision": False,
    "device": "auto",  # "auto", "cuda", "mps", "cpu"
    "seed": 42,
    "validation_prompts": [
        "rocket",
        "cat",
        "robot",
        "star",
        "heart",
    ],
    "validation_interval": 5,
    "sample_dir": "samples",
    "checkpoint_dir": "checkpoints",
}


# =============================================================================
# TRAINING CODE
# =============================================================================


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    """Get torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def get_dataset(config: dict[str, Any]) -> Any:
    """Create dataset based on config."""
    data_source = config["data_source"]

    if data_source == "huggingface":
        dataset_name = config.get("dataset_name", "junyeong-nero/emoji-32")
        print(f"Loading Hugging Face dataset: {dataset_name}")
        return EmojiDataset(dataset_name=dataset_name, split="train")

    elif data_source == "cifar100":
        print("Loading CIFAR-100 dataset")
        return CIFAR100Dataset(train=True, use_coarse_labels=False)

    else:
        raise ValueError(f"Unknown data_source: {data_source}")


def train_one_epoch(
    model: nn.Module,
    diffusion: Diffusion,
    dataloader: DataLoader,
    clip_encoder: CLIPTextEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    ema: EMA | None,
    device: torch.device,
    cfg_prob: float,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        images = batch["image"].to(device)
        captions = batch["caption"]

        # Get text embeddings and attention mask
        with torch.no_grad():
            text_embeds, text_mask = clip_encoder.encode(captions)
            text_embeds = text_embeds.to(device)
            text_mask = text_mask.to(device)

        # Random timesteps
        timesteps = torch.randint(
            0,
            diffusion.num_timesteps,
            (images.shape[0],),
            device=device,
        )

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss = diffusion.training_loss(model, images, timesteps, text_embeds, text_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = diffusion.training_loss(model, images, timesteps, text_embeds, text_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return epoch_loss / len(dataloader)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    diffusion: Diffusion,
    clip_encoder: CLIPTextEncoder,
    prompts: list[str],
    device: torch.device,
    guidance_scale: float = 7.5,
) -> torch.Tensor:
    """Generate validation samples."""
    model.eval()

    text_embeds, text_mask = clip_encoder.encode(prompts)
    text_embeds = text_embeds.to(device)
    text_mask = text_mask.to(device)

    original_scale = diffusion.guidance_scale
    diffusion.guidance_scale = guidance_scale

    images = diffusion.sample(
        model=model,
        shape=(len(prompts), 3, 32, 32),
        text_embeds=text_embeds,
        text_mask=text_mask,
        num_steps=50,
        use_ddim=True,
        use_cfg=True,
    )

    diffusion.guidance_scale = original_scale
    return images


def save_checkpoint(
    model: nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }

    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint: {path}")


def main() -> None:
    """Main training function."""
    # Select config based on stage
    if TRAINING_STAGE == "pretrain":
        config = {**COMMON_CONFIG, **PRETRAIN_CONFIG}
        print("=" * 60)
        print("STAGE 1: PRETRAINING")
        print("=" * 60)
    else:
        config = {**COMMON_CONFIG, **FINETUNE_CONFIG}
        print("=" * 60)
        print("STAGE 2: FINE-TUNING")
        print("=" * 60)

    # Print config
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Set random seed
    set_seed(config["seed"])

    # Get device
    device = get_device(config["device"])
    print(f"Using device: {device}")

    # Load dataset
    dataset = get_dataset(config)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Load CLIP encoder
    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Initialize model
    print(f"Initializing DiT-{config['model_size']}...")
    model = DiT(
        in_channels=3,
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        model_size=config["model_size"],
        clip_embed_dim=clip_encoder.embedding_dim,
    )
    model = model.to(device)

    # Load pretrained checkpoint if specified
    pretrain_checkpoint = config.get("pretrain_checkpoint")
    if pretrain_checkpoint and Path(pretrain_checkpoint).exists():
        print(f"Loading pretrained checkpoint: {pretrain_checkpoint}")
        checkpoint = torch.load(pretrain_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Pretrained weights loaded")

    # Initialize EMA
    ema = None
    if config["use_ema"]:
        ema = EMA(model, decay=config["ema_decay"])
        ema.to(device)
        print(f"EMA enabled with decay={config['ema_decay']}")

    # Initialize diffusion
    diffusion = Diffusion(
        num_timesteps=config["num_timesteps"],
        beta_schedule=config["beta_schedule"],
        guidance_scale=config["guidance_scale"],
        cfg_probability=config.get("cfg_prob", 0.1),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    # Learning rate scheduler
    num_steps_per_epoch = len(dataloader)
    total_steps = config["epochs"] * num_steps_per_epoch
    warmup_steps = total_steps // 20  # 5% warmup

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Mixed precision
    use_amp = config["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training enabled")

    # Get current CFG probability
    initial_cfg = config.get("initial_cfg_prob", 0.0)
    final_cfg = config.get("final_cfg_prob", 0.0)
    cfg_warmup = config.get("cfg_warmup_epochs", 0)

    diffusion.cfg_probability = initial_cfg
    print(f"Initial CFG probability: {initial_cfg}")

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    best_loss = float("inf")

    for epoch in range(config["epochs"]):
        # CFG warmup
        if cfg_warmup > 0 and epoch < cfg_warmup:
            progress = epoch / cfg_warmup
            current_cfg = initial_cfg + (final_cfg - initial_cfg) * progress
            diffusion.cfg_probability = current_cfg
            if epoch % 10 == 0:
                print(f"  CFG warmup: {current_cfg:.3f} (epoch {epoch}/{cfg_warmup})")
        elif cfg_warmup > 0:
            diffusion.cfg_probability = final_cfg

        # Train
        avg_loss = train_one_epoch(
            model=model,
            diffusion=diffusion,
            dataloader=dataloader,
            clip_encoder=clip_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            cfg_prob=diffusion.cfg_probability,
            use_amp=use_amp,
            scaler=scaler,
        )

        print(f"Epoch {epoch + 1}/{config['epochs']}: Avg Loss = {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = config["checkpoint_path"]
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=avg_loss,
                path=checkpoint_path,
            )

        # Generate validation samples
        if (epoch + 1) % config["validation_interval"] == 0:
            print(f"\nGenerating validation samples...")
            samples = generate_samples(
                model=model,
                diffusion=diffusion,
                clip_encoder=clip_encoder,
                prompts=config["validation_prompts"],
                device=device,
                guidance_scale=config["guidance_scale"],
            )

            # Save samples
            sample_dir = Path(config["sample_dir"]) / f"epoch_{epoch + 1}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            for i, prompt in enumerate(config["validation_prompts"]):
                img = samples[i]
                img = (img + 1) / 2  # Denormalize
                img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                safe_prompt = prompt.replace(" ", "_")[:20]
                img.save(sample_dir / f"{i:02d}_{safe_prompt}.png")

            print(f"✓ Saved samples to {sample_dir}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
