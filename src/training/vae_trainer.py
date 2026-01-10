"""VAE training utilities for tiny-stable-diffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vae import AutoencoderKL, create_vae
from src.training.checkpoint import save_checkpoint

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_vae_one_epoch(
    model: AutoencoderKL,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    kl_weight: float = 1e-6,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_wandb: bool = False,
    global_step: int = 0,
) -> tuple[float, int]:
    """Train VAE for one epoch.

    Args:
        model: VAE model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        kl_weight: Weight for KL divergence loss
        use_amp: Use automatic mixed precision
        scaler: Gradient scaler for AMP
        use_wandb: Log to wandb
        global_step: Current global step

    Returns:
        Tuple of (average loss, updated global step)
    """
    model.train()
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0

    progress_bar = tqdm(dataloader, desc="VAE Training")

    for batch in progress_bar:
        images = batch["image"].to(device)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss, loss_dict = model.training_loss(images, kl_weight=kl_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = model.training_loss(images, kl_weight=kl_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        epoch_loss += loss_dict["total_loss"]
        epoch_recon_loss += loss_dict["recon_loss"]
        epoch_kl_loss += loss_dict["kl_loss"]

        progress_bar.set_postfix(
            {
                "loss": f"{loss_dict['total_loss']:.4f}",
                "recon": f"{loss_dict['recon_loss']:.4f}",
                "kl": f"{loss_dict['kl_loss']:.4f}",
            }
        )

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "vae/total_loss": loss_dict["total_loss"],
                    "vae/recon_loss": loss_dict["recon_loss"],
                    "vae/kl_loss": loss_dict["kl_loss"],
                    "vae/learning_rate": scheduler.get_last_lr()[0],
                    "vae/global_step": global_step,
                },
                step=global_step,
            )

        global_step += 1

    num_batches = len(dataloader)
    return epoch_loss / num_batches, global_step


@torch.no_grad()
def generate_vae_samples(
    model: AutoencoderKL,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate VAE reconstruction samples.

    Args:
        model: VAE model
        dataloader: Data loader
        device: Device
        num_samples: Number of samples to generate

    Returns:
        Tuple of (original images, reconstructed images)
    """
    model.eval()

    # Get a batch of images
    batch = next(iter(dataloader))
    images = batch["image"][:num_samples].to(device)

    # Reconstruct
    reconstructions, _, _ = model(images, sample_posterior=True)

    return images, reconstructions


def train_vae(config: dict[str, Any], use_wandb: bool = False) -> None:
    """Main VAE training function.

    Args:
        config: Training configuration dictionary
        use_wandb: Enable wandb logging
    """
    import math

    from src.data.loader import create_dataloader, get_dataset
    from src.utils.common import get_device, set_seed

    print("=" * 60)
    print("STAGE 1: VAE TRAINING")
    print("=" * 60)

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Initialize wandb
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Disabling wandb logging.")
            use_wandb = False
        else:
            wandb.init(
                project=config.get("wandb_project", "tiny-stable-diffusion"),
                name=config.get("wandb_run_name", "vae-training"),
                config=config,
            )
            print("Wandb logging enabled")

    set_seed(config["seed"])
    device = get_device(config["device"])
    print(f"Using device: {device}")

    # Load dataset
    dataset = get_dataset(config)
    if hasattr(dataset, "__len__"):
        print(f"Dataset size: {len(dataset)}")
        if len(dataset) == 0:
            print("Error: Dataset is empty!")
            return
    else:
        print("Dataset: streaming mode (size unknown)")

    dataloader = create_dataloader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Initialize VAE model
    print("Initializing VAE model...")
    model = create_vae(
        image_size=config["image_size"],
        z_channels=config.get("latent_channels", 16),
        ch=config.get("vae_ch", 64),
        ch_mult=tuple(config.get("vae_ch_mult", [1, 2, 4, 4])),
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"VAE parameters: {num_params / 1_000_000:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    # Learning rate scheduler
    # For streaming datasets, use config value or estimate
    if hasattr(dataloader.dataset, "__len__"):
        num_steps_per_epoch = len(dataloader)
    else:
        num_steps_per_epoch = config.get("steps_per_epoch", 1000)
        print(f"Using configured steps_per_epoch: {num_steps_per_epoch}")
    total_steps = config["epochs"] * num_steps_per_epoch
    warmup_steps = total_steps // 20

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Mixed precision
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training enabled")

    # Training loop
    print(f"\nStarting VAE training for {config['epochs']} epochs...")
    best_loss = float("inf")
    global_step = 0
    kl_weight = config.get("kl_weight", 1e-6)

    for epoch in range(config["epochs"]):
        avg_loss, global_step = train_vae_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            kl_weight=kl_weight,
            use_amp=use_amp,
            scaler=scaler,
            use_wandb=use_wandb,
            global_step=global_step,
        )

        print(f"Epoch {epoch + 1}/{config['epochs']}: Avg Loss = {avg_loss:.4f}")

        # Log epoch metrics to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch/vae_loss": avg_loss,
                    "epoch/epoch": epoch + 1,
                },
                step=global_step,
            )

        # Save checkpoint
        checkpoint_path = config["checkpoint_path"]
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=avg_loss,
                path=checkpoint_path,
                config=config,
            )

        # Generate validation samples
        if (epoch + 1) % config.get("validation_interval", 10) == 0:
            print("\nGenerating VAE reconstruction samples...")
            originals, reconstructions = generate_vae_samples(
                model=model,
                dataloader=dataloader,
                device=device,
                num_samples=4,
            )

            sample_dir = Path(config.get("sample_dir", "samples")) / f"vae_epoch_{epoch + 1}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            for i in range(min(4, originals.shape[0])):
                # Original
                orig_img = originals[i]
                orig_img = (orig_img + 1) / 2  # [-1, 1] -> [0, 1]
                orig_img = orig_img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(orig_img).save(sample_dir / f"{i:02d}_original.png")

                # Reconstruction
                recon_img = reconstructions[i]
                recon_img = (recon_img + 1) / 2
                recon_img = recon_img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(recon_img).save(sample_dir / f"{i:02d}_reconstruction.png")

            print(f"Saved samples to {sample_dir}")

    print("\n" + "=" * 60)
    print("VAE Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print("=" * 60)

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
