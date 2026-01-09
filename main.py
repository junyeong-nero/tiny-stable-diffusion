#!/usr/bin/env python3
"""text-to-emoji - Text-to-Emoji Generator.

A diffusion transformer model for generating emoji images from text prompts.

Usage:
    python main.py --train          # Train the model
    python main.py --generate       # Generate images
    python main.py --demo           # Interactive demo

For configuration, edit the CONFIG section below.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DataConfig, DiffusionConfig, ModelConfig, TrainingConfig
from src.data.dataset import CIFAR100Dataset, EmojiDataset
from src.models.diffusion import Diffusion
from src.models.dit import DiT
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.ema import EMA


# =============================================================================
# CONFIGURATION
# =============================================================================

# Training stage: "pretrain" or "finetune"
TRAINING_STAGE: str = "pretrain"  # Change to "finetune" for Stage 2

# Pretraining Settings (Stage 1)
PRETRAIN_CONFIG: dict[str, Any] = {
    "data_source": "cifar100",
    "epochs": 2,  # Quick test run
    "batch_size": 32,  # Smaller batch for MPS
    "learning_rate": 1e-4,
    "image_size": 32,
    "initial_cfg_prob": 0.0,
    "final_cfg_prob": 0.1,
    "cfg_warmup_epochs": 1,
    "checkpoint_path": "checkpoints/pretrain_cifar100_test.pt",
}

# Fine-tuning Settings (Stage 2)
FINETUNE_CONFIG: dict[str, Any] = {
    "data_source": "huggingface",
    "dataset_name": "junyeong-nero/emoji-32",
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "image_size": 32,
    "cfg_prob": 0.1,
    "pretrain_checkpoint": "checkpoints/pretrain_cifar100.pt",
    "reset_cross_attention": True,
    "checkpoint_path": "checkpoints/finetune_emoji.pt",
}

# Common Settings
COMMON_CONFIG: dict[str, Any] = {
    "model_size": "S",
    "patch_size": 2,
    "num_timesteps": 1000,
    "beta_schedule": "cosine",
    "guidance_scale": 7.5,
    "use_ema": True,
    "ema_decay": 0.9999,
    "mixed_precision": False,
    "device": "auto",
    "seed": 42,
    "validation_prompts": ["rocket", "cat", "robot", "star", "heart"],
    "validation_interval": 5,
    "sample_dir": "samples",
    "checkpoint_dir": "checkpoints",
}


# =============================================================================
# TRAINING FUNCTIONS
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

    if data_source == "local":
        local_path = config.get("local_dataset_path")
        print(f"Loading local dataset from: {local_path}")
        return EmojiDataset(dataset_name=local_path, split="train")

    elif data_source == "huggingface":
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

        with torch.no_grad():
            text_embeds, text_mask = clip_encoder.encode(captions)
            text_embeds = text_embeds.to(device)
            text_mask = text_mask.to(device)

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


def train(config: dict[str, Any]) -> None:
    """Main training function."""
    if TRAINING_STAGE == "pretrain":
        print("=" * 60)
        print("STAGE 1: PRETRAINING")
        print("=" * 60)
    else:
        print("=" * 60)
        print("STAGE 2: FINE-TUNING")
        print("=" * 60)

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    set_seed(config["seed"])
    device = get_device(config["device"])
    print(f"Using device: {device}")

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

    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Compute unconditional embedding for classifier-free guidance (empty string)
    print("Computing unconditional embedding...")
    with torch.no_grad():
        uncond_embed, uncond_mask = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)
    uncond_mask = uncond_mask.to(device) if uncond_mask is not None else None

    print(f"Initializing DiT-{config['model_size']}...")
    model = DiT(
        in_channels=3,
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        model_size=config["model_size"],
        clip_embed_dim=clip_encoder.embedding_dim,
    )
    model = model.to(device)

    pretrain_checkpoint = config.get("pretrain_checkpoint")
    if pretrain_checkpoint and Path(pretrain_checkpoint).exists():
        print(f"Loading pretrained checkpoint: {pretrain_checkpoint}")
        checkpoint = torch.load(pretrain_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Pretrained weights loaded")

    ema = None
    if config["use_ema"]:
        ema = EMA(model, decay=config["ema_decay"])
        ema.to(device)
        print(f"EMA enabled with decay={config['ema_decay']}")

    diffusion = Diffusion(
        num_timesteps=config["num_timesteps"],
        beta_schedule=config["beta_schedule"],
        guidance_scale=config["guidance_scale"],
        cfg_probability=config.get("cfg_prob", config.get("initial_cfg_prob", 0.1)),
        uncond_embed=uncond_embed,
        uncond_mask=uncond_mask,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    num_steps_per_epoch = len(dataloader)
    total_steps = config["epochs"] * num_steps_per_epoch
    warmup_steps = total_steps // 20

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_amp = config["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training enabled")

    initial_cfg = config.get("initial_cfg_prob", 0.0)
    final_cfg = config.get("final_cfg_prob", config.get("cfg_prob", 0.1))
    cfg_warmup = config.get("cfg_warmup_epochs", 0)

    diffusion.cfg_probability = initial_cfg
    print(f"Initial CFG probability: {initial_cfg}")

    print(f"\nStarting training for {config['epochs']} epochs...")
    best_loss = float("inf")

    for epoch in range(config["epochs"]):
        if cfg_warmup > 0 and epoch < cfg_warmup:
            progress = epoch / cfg_warmup
            current_cfg = initial_cfg + (final_cfg - initial_cfg) * progress
            diffusion.cfg_probability = current_cfg
            if epoch % 10 == 0:
                print(f"  CFG warmup: {current_cfg:.3f} (epoch {epoch}/{cfg_warmup})")
        elif cfg_warmup > 0:
            diffusion.cfg_probability = final_cfg

        avg_loss = train_one_epoch(
            model=model,
            diffusion=diffusion,
            dataloader=dataloader,
            clip_encoder=clip_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )

        print(f"Epoch {epoch + 1}/{config['epochs']}: Avg Loss = {avg_loss:.4f}")

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

            sample_dir = Path(config["sample_dir"]) / f"epoch_{epoch + 1}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            for i, prompt in enumerate(config["validation_prompts"]):
                img = samples[i]
                img = (img + 1) / 2
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


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================


@torch.no_grad()
def generate(
    prompts: list[str],
    checkpoint: str = "checkpoints/model_best.pt",
    num_samples: int = 1,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    device: str = "auto",
) -> list[Image.Image]:
    """Generate images from text prompts.

    Args:
        prompts: List of text prompts
        checkpoint: Path to model checkpoint
        num_samples: Number of samples per prompt
        num_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        device: Device to use

    Returns:
        List of generated PIL images
    """
    if seed is not None:
        set_seed(seed)

    device = get_device(device)
    print(f"Using device: {device}")

    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Compute unconditional embedding for classifier-free guidance
    with torch.no_grad():
        uncond_embed, uncond_mask = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)
    uncond_mask = uncond_mask.to(device) if uncond_mask is not None else None

    print(f"Loading checkpoint: {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location=device)

    model_config = checkpoint.get("model_config", {})
    model_size = model_config.get("model_size", "S")
    patch_size = model_config.get("patch_size", 2)
    image_size = model_config.get("image_size", 32)

    print(f"Initializing DiT-{model_size}...")
    model = DiT(
        in_channels=3,
        image_size=image_size,
        patch_size=patch_size,
        model_size=model_size,
        clip_embed_dim=clip_encoder.embedding_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    diffusion = Diffusion(
        num_timesteps=1000,
        beta_schedule="cosine",
        guidance_scale=guidance_scale,
        uncond_embed=uncond_embed,
        uncond_mask=uncond_mask,
    )

    print(f"\nGenerating {num_samples} image(s) for {len(prompts)} prompt(s)...")
    all_images = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i + 1}/{len(prompts)}] '{prompt}'")

        text_embeds, text_mask = clip_encoder.encode([prompt] * num_samples)
        text_embeds = text_embeds.to(device)
        text_mask = text_mask.to(device)

        images = diffusion.sample(
            model=model,
            shape=(num_samples, 3, image_size, image_size),
            text_embeds=text_embeds,
            text_mask=text_mask,
            num_steps=num_steps,
            use_ddim=True,
            use_cfg=True,
        )

        for j, img in enumerate(images):
            img = (img + 1) / 2  # Denormalize
            img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            pil_img = Image.fromarray(img)
            all_images.append(pil_img)

    print(f"✓ Generated {len(all_images)} image(s)")
    return all_images


def demo() -> None:
    """Interactive demo mode."""
    print("=" * 60)
    print("text-to-emoji Interactive Demo")
    print("=" * 60)
    print("\nEnter prompts to generate images. Type 'quit' to exit.\n")

    checkpoint = "checkpoints/model_best.pt"
    if not Path(checkpoint).exists():
        print(f"Error: Checkpoint not found: {checkpoint}")
        print("Please train the model first: python main.py --train")
        return

    device = get_device("auto")
    print(f"Using device: {device}")

    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Compute unconditional embedding for classifier-free guidance
    with torch.no_grad():
        uncond_embed, uncond_mask = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)
    uncond_mask = uncond_mask.to(device) if uncond_mask is not None else None

    checkpoint = torch.load(checkpoint, map_location=device)
    model_config = checkpoint.get("model_config", {})
    model = DiT(
        in_channels=3,
        image_size=32,
        patch_size=2,
        model_size=model_config.get("model_size", "S"),
        clip_embed_dim=clip_encoder.embedding_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    diffusion = Diffusion(
        num_timesteps=1000,
        beta_schedule="cosine",
        guidance_scale=7.5,
        uncond_embed=uncond_embed,
        uncond_mask=uncond_mask,
    )

    while True:
        try:
            prompt = input("\nEnter prompt (or 'quit'): ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt:
                continue

            print(f"Generating: '{prompt}'...")
            text_embeds, text_mask = clip_encoder.encode([prompt])
            text_embeds = text_embeds.to(device)
            text_mask = text_mask.to(device)

            images = diffusion.sample(
                model=model,
                shape=(1, 3, 32, 32),
                text_embeds=text_embeds,
                text_mask=text_mask,
                num_steps=50,
                use_ddim=True,
                use_cfg=True,
            )

            img = images[0]
            img = (img + 1) / 2
            img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            pil_img = Image.fromarray(img)

            # Save and display
            output_path = f"demo_{prompt[:10]}.png"
            pil_img.save(output_path)
            print(f"✓ Saved: {output_path}")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="text-to-emoji - Text-to-Pixel Art Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Run pretraining on CIFAR-100",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Run fine-tuning on emoji dataset",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model (uses TRAINING_STAGE from config)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate images from prompts",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo",
    )

    # Training arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset path or name (e.g., 'cifar100', 'path/to/dataset', 'user/dataset-name')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (required for finetune)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for generation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output file path for generated image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.pretrain:
        config = {**COMMON_CONFIG, **PRETRAIN_CONFIG}
        # Override with CLI args if provided
        if args.dataset is not None:
            if Path(args.dataset).exists():
                # Local dataset path
                config["data_source"] = "local"
                config["local_dataset_path"] = args.dataset
            else:
                # HuggingFace dataset name
                config["data_source"] = "huggingface"
                config["dataset_name"] = args.dataset
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.learning_rate is not None:
            config["learning_rate"] = args.learning_rate
        if args.checkpoint is not None:
            config["checkpoint_path"] = args.checkpoint
        train(config)

    elif args.finetune:
        config = {**COMMON_CONFIG, **FINETUNE_CONFIG}
        # Override with CLI args if provided
        if args.dataset is not None:
            if Path(args.dataset).exists():
                # Local dataset path
                config["data_source"] = "local"
                config["local_dataset_path"] = args.dataset
            else:
                # HuggingFace dataset name
                config["data_source"] = "huggingface"
                config["dataset_name"] = args.dataset
        if args.checkpoint is not None:
            config["pretrain_checkpoint"] = args.checkpoint
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.learning_rate is not None:
            config["learning_rate"] = args.learning_rate
        train(config)

    elif args.train:
        # Select config based on TRAINING_STAGE constant
        if TRAINING_STAGE == "pretrain":
            config = {**COMMON_CONFIG, **PRETRAIN_CONFIG}
        else:
            config = {**COMMON_CONFIG, **FINETUNE_CONFIG}
        train(config)

    elif args.generate:
        if args.prompt is None:
            print("Error: --prompt required for --generate")
            return

        prompts = [p.strip() for p in args.prompt.split(",")]
        images = generate(
            prompts=prompts,
            checkpoint=args.checkpoint,
            num_samples=args.num_samples,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )

        for i, img in enumerate(images):
            if len(prompts) > 1:
                output_path = f"output_{i}.png"
            else:
                output_path = args.output
            img.save(output_path)
            print(f"✓ Saved: {output_path}")

    elif args.demo:
        demo()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
