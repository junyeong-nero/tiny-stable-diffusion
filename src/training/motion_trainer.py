"""Motion Module training for video/GIF generation.

Implements the AnimateDiff-style training loop:
1. Load pre-trained VAE (frozen)
2. Load pre-trained DiT/MMDiT (frozen)
3. Train only the Motion Module on video data
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.animated_diffusion import AnimatedDiffusion
from src.models.animated_mmdit import AnimatedMMDiT
from src.models.vae import AutoencoderKL
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.ema import EMA
from src.training.wandb_logger import WandbLogger
from src.data.video_transforms import video_to_gif, denormalize_video


@dataclass
class MotionTrainConfig:
    """Configuration for motion module training."""

    # Model paths
    vae_checkpoint: str = "checkpoints/vae.pt"
    base_checkpoint: str = "checkpoints/diffusion.pt"

    # Motion module architecture
    motion_num_layers: int = 2
    motion_num_heads: int = 8
    num_frames: int = 16

    # Training
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0

    # Diffusion
    num_timesteps: int = 1000
    guidance_scale: float = 7.5
    min_snr_gamma: float | None = 5.0
    temporal_consistency_weight: float = 0.0

    # CFG
    initial_cfg_prob: float = 0.0
    final_cfg_prob: float = 0.1
    cfg_warmup_epochs: int = 10

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Checkpoint
    checkpoint_path: str = "checkpoints/motion.pt"
    checkpoint_interval: int = 10

    # Validation
    validation_interval: int = 10
    validation_prompts: list[str] | None = None
    sample_dir: str = "samples/motion"

    # Device
    device: str = "auto"
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42


def train_motion_one_epoch(
    model: AnimatedMMDiT,
    diffusion: AnimatedDiffusion,
    dataloader: DataLoader,
    clip_encoder: CLIPTextEncoder,
    vae_encoder: AutoencoderKL,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    config: MotionTrainConfig,
    ema: EMA | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    logger: WandbLogger | None = None,
    global_step: int = 0,
) -> tuple[float, dict[str, float], int]:
    """Train motion module for one epoch.

    Args:
        model: AnimatedMMDiT with motion module
        diffusion: AnimatedDiffusion for video
        dataloader: Video data loader
        clip_encoder: CLIP text encoder
        vae_encoder: Frozen VAE encoder
        optimizer: Optimizer (only motion module params)
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Training configuration
        ema: EMA model (optional)
        scaler: Gradient scaler for AMP
        logger: WandbLogger instance
        global_step: Current global step

    Returns:
        Tuple of (average loss, loss dict, updated global step)
    """
    model.train()
    vae_encoder.eval()

    epoch_losses = {
        "total_loss": 0.0,
        "velocity_loss": 0.0,
        "temporal_loss": 0.0,
    }
    step_count = 0
    accumulation_steps = 0

    progress_bar = tqdm(dataloader, desc="Training Motion")

    for batch in progress_bar:
        # Get video frames and captions
        frames = batch["frames"].to(device)  # (B, F, C, H, W)
        captions = batch["caption"]

        b, f, c, h, w = frames.shape

        # Encode frames to latent space using frozen VAE
        with torch.no_grad():
            # Reshape for VAE: (B, F, C, H, W) -> (B*F, C, H, W)
            frames_flat = frames.view(b * f, c, h, w)
            latents_flat = vae_encoder.encode_to_latent(frames_flat)
            # Reshape back: (B*F, C', H', W') -> (B, F, C', H', W')
            _, c_lat, h_lat, w_lat = latents_flat.shape
            latents = latents_flat.view(b, f, c_lat, h_lat, w_lat)

            # Encode text
            text_embeds = clip_encoder.encode(captions)
            text_embeds = text_embeds.to(device)

        # Sample timesteps (same for all frames in a video)
        timesteps = diffusion.sample_timesteps_logit_normal(
            batch_size=b,
            device=device,
        )

        # Forward and loss
        use_amp = config.mixed_precision and device.type == "cuda"

        if use_amp:
            with torch.cuda.amp.autocast():
                loss, loss_dict = diffusion.training_loss_video(
                    model, latents, timesteps, text_embeds
                )
                loss = loss / config.gradient_accumulation
            scaler.scale(loss).backward()
        else:
            loss, loss_dict = diffusion.training_loss_video(
                model, latents, timesteps, text_embeds
            )
            loss = loss / config.gradient_accumulation
            loss.backward()

        accumulation_steps += 1

        # Update weights after accumulation
        if accumulation_steps >= config.gradient_accumulation:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(), config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(), config.max_grad_norm
                )
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            if ema is not None:
                ema.update()

            accumulation_steps = 0

        # Accumulate losses
        for key in epoch_losses:
            if key in loss_dict:
                epoch_losses[key] += loss_dict[key]
        step_count += 1

        progress_bar.set_postfix({
            "loss": loss_dict["total_loss"],
            "v_loss": loss_dict["velocity_loss"],
        })

        # Log to wandb
        if logger is not None:
            logger.log(
                {
                    "train/loss": loss_dict["total_loss"],
                    "train/velocity_loss": loss_dict["velocity_loss"],
                    "train/temporal_loss": loss_dict["temporal_loss"],
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                },
                step=global_step,
            )

        global_step += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(step_count, 1)

    return epoch_losses["total_loss"], epoch_losses, global_step


@torch.no_grad()
def generate_motion_samples(
    model: AnimatedMMDiT,
    diffusion: AnimatedDiffusion,
    clip_encoder: CLIPTextEncoder,
    vae_decoder: AutoencoderKL,
    prompts: list[str],
    device: torch.device,
    num_frames: int = 16,
    guidance_scale: float = 7.5,
    latent_size: int = 8,
    latent_channels: int = 16,
    num_steps: int = 50,
) -> torch.Tensor:
    """Generate video samples for validation.

    Args:
        model: AnimatedMMDiT
        diffusion: AnimatedDiffusion
        clip_encoder: CLIP encoder
        vae_decoder: VAE decoder
        prompts: Text prompts
        device: Device
        num_frames: Frames per video
        guidance_scale: CFG scale
        latent_size: Latent spatial size
        latent_channels: Latent channels
        num_steps: Sampling steps

    Returns:
        Generated videos (B, F, 3, H, W) in [0, 1]
    """
    model.eval()

    text_embeds = clip_encoder.encode(prompts)
    text_embeds = text_embeds.to(device)

    original_scale = diffusion.guidance_scale
    diffusion.guidance_scale = guidance_scale

    videos = diffusion.sample_video(
        model=model,
        batch_size=len(prompts),
        num_frames=num_frames,
        latent_channels=latent_channels,
        latent_size=latent_size,
        text_embeds=text_embeds,
        num_steps=num_steps,
        use_cfg=True,
        vae_decoder=vae_decoder,
        device=device,
    )

    diffusion.guidance_scale = original_scale
    return videos


def save_motion_checkpoint(
    model: AnimatedMMDiT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    loss: float,
    path: str | Path,
    config: MotionTrainConfig,
    ema: EMA | None = None,
    global_step: int = 0,
) -> None:
    """Save motion module checkpoint.

    Only saves the motion module weights, not the frozen base model.
    """
    checkpoint = {
        "motion_module_state_dict": model.motion_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "global_step": global_step,
        "config": {
            "motion_num_layers": config.motion_num_layers,
            "motion_num_heads": config.motion_num_heads,
            "num_frames": config.num_frames,
        },
    }

    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_motion_checkpoint(
    path: str | Path,
    model: AnimatedMMDiT,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: EMA | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load motion module checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    model.motion_module.load_state_dict(checkpoint["motion_module_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "global_step": checkpoint.get("global_step", 0),
    }


def train_motion(config: dict[str, Any], use_wandb: bool = False) -> None:
    """Main motion module training function.

    Args:
        config: Training configuration dictionary
        use_wandb: Enable wandb logging
    """
    from src.data.video_dataset import SyntheticVideoDataset, create_video_dataloader
    from src.models.animated_mmdit import load_animated_mmdit
    from src.models.vae import create_vae
    from src.utils.common import get_device, set_seed

    print("=" * 60)
    print("STAGE 3: MOTION MODULE TRAINING")
    print("=" * 60)

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Initialize wandb
    logger = WandbLogger(
        enabled=use_wandb,
        project=config.get("wandb_project", "tiny-stable-diffusion"),
        run_name=config.get("wandb_run_name", "motion-training"),
        config=config,
    )

    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    print(f"Using device: {device}")

    # Load VAE
    vae_checkpoint = config.get("vae_checkpoint", "checkpoints/vae.pt")
    if not Path(vae_checkpoint).exists():
        print(f"Error: VAE checkpoint not found: {vae_checkpoint}")
        return

    print(f"Loading VAE from {vae_checkpoint}...")
    vae = create_vae(
        image_size=config.get("image_size", 64),
        z_channels=config.get("latent_channels", 16),
    )
    vae_state = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state["model_state_dict"])
    vae = vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    scaling_factor = config.get("scaling_factor", vae_state.get("scaling_factor", 1.0))
    vae.set_scaling_factor(scaling_factor)
    print(f"VAE loaded (scaling_factor={vae.scaling_factor:.4f})")

    # Load base diffusion model and create AnimatedMMDiT
    base_checkpoint = config.get("base_checkpoint", "checkpoints/diffusion.pt")
    if not Path(base_checkpoint).exists():
        print(f"Error: Base model checkpoint not found: {base_checkpoint}")
        return

    use_gradient_checkpointing = config.get("gradient_checkpointing", False)
    print(f"Loading base model from {base_checkpoint}...")
    model = load_animated_mmdit(
        base_checkpoint_path=base_checkpoint,
        motion_checkpoint_path=None,
        device=device,
        in_channels=config.get("latent_channels", 16),
        image_size=config.get("latent_size", 8),
        num_frames=config.get("num_frames", 16),
        motion_num_layers=config.get("motion_num_layers", 2),
        motion_num_heads=config.get("motion_num_heads", 8),
        freeze_base=True,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    model = model.to(device)

    if use_gradient_checkpointing:
        print("Gradient checkpointing enabled for motion module")

    param_counts = model.parameters_count()
    print(f"Base model parameters: {param_counts['base_total'] / 1e6:.2f}M (frozen)")
    print(f"Motion module parameters: {param_counts['motion_trainable'] / 1e6:.2f}M (trainable)")

    # Load CLIP encoder
    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Compute unconditional embedding
    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

    # Initialize AnimatedDiffusion
    diffusion = AnimatedDiffusion(
        num_timesteps=config.get("num_timesteps", 1000),
        num_frames=config.get("num_frames", 16),
        guidance_scale=config.get("guidance_scale", 7.5),
        cfg_probability=config.get("initial_cfg_prob", 0.0),
        uncond_embed=uncond_embed,
        min_snr_gamma=config.get("min_snr_gamma", 5.0),
        temporal_consistency_weight=config.get("temporal_consistency_weight", 0.0),
    )

    # Create dataset
    # Use synthetic data for testing, real video data for production
    use_synthetic = config.get("use_synthetic_data", True)
    if use_synthetic:
        print("Using synthetic video dataset for training...")
        dataset = SyntheticVideoDataset(
            size=config.get("synthetic_size", 1000),
            num_frames=config.get("num_frames", 16),
            image_size=config.get("image_size", 64),
            pattern=config.get("synthetic_pattern", "moving_circle"),
        )
    else:
        from src.data.video_dataset import VideoDataset
        dataset = VideoDataset(
            dataset_name=config.get("dataset_name"),
            num_frames=config.get("num_frames", 16),
            target_size=config.get("image_size", 64),
        )

    dataloader = create_video_dataloader(
        dataset,
        batch_size=config.get("batch_size", 4),
        num_workers=config.get("num_workers", 4),
        shuffle=True,
    )

    # Initialize EMA
    ema = None
    if config.get("use_ema", True):
        # Only EMA the motion module
        ema = EMA(model.motion_module, decay=config.get("ema_decay", 0.9999))
        ema.to(device)
        print(f"EMA enabled with decay={config.get('ema_decay', 0.9999)}")

    # Optimizer - only for motion module parameters
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config.get("learning_rate", 1e-4),
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    # Scheduler
    try:
        steps_per_epoch = len(dataloader)
    except TypeError:
        steps_per_epoch = config.get("steps_per_epoch", 1000)

    total_steps = config.get("epochs", 100) * steps_per_epoch
    warmup_steps = total_steps // 20

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Mixed precision
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Create config object
    train_config = MotionTrainConfig(
        vae_checkpoint=vae_checkpoint,
        base_checkpoint=base_checkpoint,
        motion_num_layers=config.get("motion_num_layers", 2),
        motion_num_heads=config.get("motion_num_heads", 8),
        num_frames=config.get("num_frames", 16),
        epochs=config.get("epochs", 100),
        batch_size=config.get("batch_size", 4),
        learning_rate=config.get("learning_rate", 1e-4),
        gradient_accumulation=config.get("gradient_accumulation", 1),
        mixed_precision=use_amp,
        gradient_checkpointing=use_gradient_checkpointing,
    )

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    checkpoint_path = Path(config.get("checkpoint_path", "checkpoints/motion.pt"))

    if config.get("resume", False) and checkpoint_path.exists():
        print(f"Resuming from {checkpoint_path}...")
        ckpt_info = load_motion_checkpoint(
            checkpoint_path, model, optimizer, scheduler, ema, device
        )
        start_epoch = ckpt_info["epoch"] + 1
        global_step = ckpt_info["global_step"]
        best_loss = ckpt_info["loss"]
        print(f"Resumed from epoch {ckpt_info['epoch']}, loss {best_loss:.4f}")

    # CFG warmup settings
    initial_cfg = config.get("initial_cfg_prob", 0.0)
    final_cfg = config.get("final_cfg_prob", 0.1)
    cfg_warmup = config.get("cfg_warmup_epochs", 10)

    # Training loop
    print(f"\nStarting motion training for {config.get('epochs', 100)} epochs...")

    for epoch in range(start_epoch, config.get("epochs", 100)):
        # CFG warmup
        if cfg_warmup > 0 and epoch < cfg_warmup:
            progress = epoch / cfg_warmup
            diffusion.cfg_probability = initial_cfg + (final_cfg - initial_cfg) * progress
        else:
            diffusion.cfg_probability = final_cfg

        avg_loss, loss_dict, global_step = train_motion_one_epoch(
            model=model,
            diffusion=diffusion,
            dataloader=dataloader,
            clip_encoder=clip_encoder,
            vae_encoder=vae,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=train_config,
            ema=ema,
            scaler=scaler,
            logger=logger,
            global_step=global_step,
        )

        print(
            f"Epoch {epoch + 1}/{config.get('epochs', 100)}: "
            f"Loss = {avg_loss:.4f}, "
            f"V_Loss = {loss_dict['velocity_loss']:.4f}"
        )

        # Log epoch metrics
        logger.log(
            {
                "epoch/train_loss": avg_loss,
                "epoch/velocity_loss": loss_dict["velocity_loss"],
                "epoch/temporal_loss": loss_dict["temporal_loss"],
                "epoch/epoch": epoch + 1,
            },
            step=global_step,
        )

        # Save checkpoint
        save_motion_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=avg_loss,
            path=checkpoint_path,
            config=train_config,
            ema=ema,
            global_step=global_step,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = checkpoint_path.parent / "motion_best.pt"
            save_motion_checkpoint(
                model, optimizer, scheduler, epoch, avg_loss,
                best_path, train_config, ema, global_step
            )
            print(f"  New best loss: {best_loss:.4f}")

        # Generate validation samples
        validation_interval = config.get("validation_interval", 10)
        if (epoch + 1) % validation_interval == 0:
            prompts = config.get("validation_prompts", [
                "a moving circle",
                "an animated shape",
            ])

            print("\nGenerating validation samples...")
            if ema is not None:
                ema.apply()

            videos = generate_motion_samples(
                model=model,
                diffusion=diffusion,
                clip_encoder=clip_encoder,
                vae_decoder=vae,
                prompts=prompts,
                device=device,
                num_frames=config.get("num_frames", 16),
                guidance_scale=config.get("guidance_scale", 7.5),
                latent_size=config.get("latent_size", 8),
                latent_channels=config.get("latent_channels", 16),
            )

            if ema is not None:
                ema.restore()

            # Save as GIFs
            sample_dir = Path(config.get("sample_dir", "samples/motion")) / f"epoch_{epoch + 1}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            for i, prompt in enumerate(prompts):
                video = videos[i]  # (F, C, H, W)
                safe_prompt = prompt.replace(" ", "_")[:20]
                gif_path = sample_dir / f"{i:02d}_{safe_prompt}.gif"
                video_to_gif(video, str(gif_path), fps=8)

            print(f"Saved samples to {sample_dir}")

    print("\n" + "=" * 60)
    print("Motion Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)

    logger.finish()
