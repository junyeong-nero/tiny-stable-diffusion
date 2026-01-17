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
from src.training.checkpoint import load_checkpoint, save_checkpoint

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_kl_weight(
    step: int,
    total_steps: int,
    max_weight: float,
    annealing: str = "cyclical",
    n_cycles: int = 4,
    cycle_ratio: float = 0.5,
) -> float:
    """Get KL weight based on annealing schedule.

    Args:
        step: Current training step (global)
        total_steps: Total training steps
        max_weight: Maximum/target KL weight
        annealing: Annealing type ("none", "linear", "cyclical")
        n_cycles: Number of cycles for cyclical annealing
        cycle_ratio: Proportion of cycle spent increasing (0-1)

    Returns:
        Current KL weight value
    """
    if annealing == "none":
        return max_weight

    elif annealing == "linear":
        # Linear warmup from 0 to max_weight over all steps
        return max_weight * min(1.0, step / max(1, total_steps))

    elif annealing == "cyclical":
        # Cyclical annealing: repeat warmup cycles
        # Reference: "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
        cycle_length = total_steps // n_cycles
        if cycle_length == 0:
            return max_weight

        current_cycle_step = step % cycle_length
        warmup_steps = int(cycle_length * cycle_ratio)

        if warmup_steps == 0:
            return max_weight
        elif current_cycle_step < warmup_steps:
            # Linear increase from 0 to max_weight
            return max_weight * (current_cycle_step / warmup_steps)
        else:
            # At max_weight for rest of cycle
            return max_weight

    else:
        raise ValueError(f"Unknown annealing type: {annealing}")


def train_vae_one_epoch(
    model: AutoencoderKL,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    kl_max_weight: float = 1e-3,
    kl_annealing: str = "cyclical",
    kl_n_cycles: int = 4,
    kl_cycle_ratio: float = 0.5,
    total_steps: int = 0,
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
        kl_max_weight: Maximum KL weight (target value)
        kl_annealing: Annealing type ("none", "linear", "cyclical")
        kl_n_cycles: Number of cycles for cyclical annealing
        kl_cycle_ratio: Proportion of cycle spent increasing
        total_steps: Total training steps (for annealing schedule)
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
    epoch_latent_mean = 0.0
    epoch_latent_std = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="VAE Training")

    for batch in progress_bar:
        num_batches += 1
        images = batch["image"].to(device)

        # Compute current KL weight based on annealing schedule
        kl_weight = get_kl_weight(
            step=global_step,
            total_steps=total_steps,
            max_weight=kl_max_weight,
            annealing=kl_annealing,
            n_cycles=kl_n_cycles,
            cycle_ratio=kl_cycle_ratio,
        )

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
        epoch_latent_mean += loss_dict["latent_mean"]
        epoch_latent_std += loss_dict["latent_std"]

        # Show latent stats in progress bar (μ should → 0, σ should → 1)
        progress_bar.set_postfix(
            {
                "loss": f"{loss_dict['total_loss']:.4f}",
                "recon": f"{loss_dict['recon_loss']:.4f}",
                "kl": f"{loss_dict['kl_loss']:.2f}",
                "β": f"{kl_weight:.1e}",
                "μ": f"{loss_dict['latent_mean']:.2f}",
                "σ": f"{loss_dict['latent_std']:.3f}",
            }
        )

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "vae/total_loss": loss_dict["total_loss"],
                    "vae/recon_loss": loss_dict["recon_loss"],
                    "vae/kl_loss": loss_dict["kl_loss"],
                    "vae/kl_weight": kl_weight,
                    "vae/learning_rate": scheduler.get_last_lr()[0],
                    "vae/global_step": global_step,
                    # Latent space statistics
                    "latent/mean": loss_dict["latent_mean"],
                    "latent/mean_std": loss_dict["latent_mean_std"],
                    "latent/std": loss_dict["latent_std"],
                    "latent/std_std": loss_dict["latent_std_std"],
                },
                step=global_step,
            )

        global_step += 1

    return epoch_loss / max(1, num_batches), global_step


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
    resume = config.get("resume", False)
    checkpoint_path = Path(config["checkpoint_path"])
    wandb_run_id = None

    # Load wandb_run_id from checkpoint if resuming
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        wandb_run_id = checkpoint.get("wandb_run_id")

    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Disabling wandb logging.")
            use_wandb = False
        else:
            wandb.init(
                project=config.get("wandb_project", "tiny-stable-diffusion"),
                name=config.get("wandb_run_name", "vae-training"),
                config=config,
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
            )
            # Save run id for future resume
            wandb_run_id = wandb.run.id
            print(f"Wandb logging enabled (run_id: {wandb_run_id})")

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

    # Resume from checkpoint if enabled
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if resume and checkpoint_path.exists():
        print(f"\nResuming training from checkpoint: {checkpoint_path}")
        checkpoint_info = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = checkpoint_info["epoch"] + 1
        global_step = checkpoint_info.get("global_step", 0)
        best_loss = checkpoint_info["loss"]
        print(f"Resumed from epoch {checkpoint_info['epoch']}, global_step {global_step}, loss {best_loss:.4f}")
    elif resume:
        print(f"\nResume enabled but no checkpoint found at {checkpoint_path}. Starting from scratch.")

    # Mixed precision
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training enabled")

    # KL Annealing settings
    kl_max_weight = config.get("kl_weight", 1e-3)
    kl_annealing = config.get("kl_annealing", "cyclical")
    kl_n_cycles = config.get("kl_n_cycles", 4)
    kl_cycle_ratio = config.get("kl_cycle_ratio", 0.5)

    print(f"\nKL Annealing: {kl_annealing}")
    print(f"  Max KL weight (β): {kl_max_weight:.1e}")
    if kl_annealing == "cyclical":
        print(f"  Cycles: {kl_n_cycles}, Ratio: {kl_cycle_ratio}")

    # Training loop
    if start_epoch > 0:
        print(f"\nResuming VAE training from epoch {start_epoch + 1}/{config['epochs']}...")
    else:
        print(f"\nStarting VAE training for {config['epochs']} epochs...")

    for epoch in range(start_epoch, config["epochs"]):
        avg_loss, global_step = train_vae_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            kl_max_weight=kl_max_weight,
            kl_annealing=kl_annealing,
            kl_n_cycles=kl_n_cycles,
            kl_cycle_ratio=kl_cycle_ratio,
            total_steps=total_steps,
            use_amp=use_amp,
            scaler=scaler,
            use_wandb=use_wandb,
            global_step=global_step,
        )

        # Log epoch summary with latent stats
        with torch.no_grad():
            # Get a batch to compute latent statistics
            sample_batch = next(iter(dataloader))
            sample_images = sample_batch["image"].to(device)
            mean, logvar = model.encode(sample_images)
            std = torch.exp(0.5 * logvar)

            latent_mean = mean.mean().item()
            latent_mean_std = mean.std().item()
            latent_std = std.mean().item()

        # Get current KL weight and cycle info
        current_kl_weight = get_kl_weight(
            step=global_step,
            total_steps=total_steps,
            max_weight=kl_max_weight,
            annealing=kl_annealing,
            n_cycles=kl_n_cycles,
            cycle_ratio=kl_cycle_ratio,
        )
        if kl_annealing == "cyclical" and total_steps > 0:
            cycle_length = total_steps // kl_n_cycles
            current_cycle = (global_step // cycle_length) + 1 if cycle_length > 0 else 1
            cycle_info = f" | Cycle {current_cycle}/{kl_n_cycles}"
        else:
            cycle_info = ""

        print(f"Epoch {epoch + 1}/{config['epochs']}: Loss={avg_loss:.4f} | "
              f"β={current_kl_weight:.1e}{cycle_info} | "
              f"Latent μ={latent_mean:.3f}, σ={latent_std:.3f} [target: μ≈0, σ≈1]")

        # Log epoch metrics to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch/vae_loss": avg_loss,
                    "epoch/epoch": epoch + 1,
                    "epoch/kl_weight": current_kl_weight,
                    "epoch/latent_mean": latent_mean,
                    "epoch/latent_mean_std": latent_mean_std,
                    "epoch/latent_std": latent_std,
                },
                step=global_step,
            )

        # Save checkpoint (always save for resume support)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=avg_loss,
            path=checkpoint_path,
            config=config,
            global_step=global_step,
            wandb_run_id=wandb_run_id,
        )
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Save periodic checkpoint every 10 epochs (weights only for minimal size)
        checkpoint_interval = config.get("checkpoint_interval", 10)
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = checkpoint_path.parent
            periodic_path = checkpoint_dir / f"vae_epoch_{epoch + 1}.pt"
            torch.save({"model_state_dict": model.state_dict()}, periodic_path)
            print(f"Saved periodic checkpoint: {periodic_path}")

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
