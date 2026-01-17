"""Training utilities for tiny-stable-diffusion.

Implements latent-space diffusion training (Stable Diffusion 3 style):
1. Load pre-trained VAE
2. Encode images to latent space using frozen VAE encoder
3. Train diffusion model on latent space
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.diffusion import Diffusion
from src.models.vae import AutoencoderKL, create_vae
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.ema import EMA

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_one_epoch(
    model: nn.Module,
    diffusion: Diffusion,
    dataloader: DataLoader,
    clip_encoder: CLIPTextEncoder,
    vae_encoder: AutoencoderKL,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    ema: EMA | None = None,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_wandb: bool = False,
    global_step: int = 0,
) -> tuple[float, int]:
    """Train for one epoch on latent space.

    Args:
        model: Diffusion model (operates on latent space)
        diffusion: Diffusion process
        dataloader: Training data loader
        clip_encoder: CLIP text encoder
        vae_encoder: Frozen VAE encoder for image-to-latent conversion
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        ema: EMA model (optional)
        use_amp: Use automatic mixed precision
        scaler: Gradient scaler for AMP
        use_wandb: Log to wandb
        global_step: Current global step

    Returns:
        Tuple of (average loss, updated global step)
    """
    model.train()
    vae_encoder.eval()  # VAE is always frozen
    epoch_loss = 0.0
    step_count = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        images = batch["image"].to(device)
        captions = batch["caption"]

        # Encode images to latent space using frozen VAE
        with torch.no_grad():
            latents = vae_encoder.encode_to_latent(images)
            text_embeds = clip_encoder.encode(captions)
            text_embeds = text_embeds.to(device)

        # Sample timesteps using logit-normal distribution (SD3 style)
        timesteps = diffusion.sample_timesteps_logit_normal(
            batch_size=latents.shape[0],
            device=device,
        )

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss = diffusion.training_loss(model, latents, timesteps, text_embeds)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = diffusion.training_loss(model, latents, timesteps, text_embeds)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update()

        loss_value = loss.item()
        epoch_loss += loss_value
        step_count += 1
        progress_bar.set_postfix({"loss": loss_value})

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "train/loss": loss_value,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                },
                step=global_step,
            )

        global_step += 1

    return epoch_loss / max(step_count, 1), global_step


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    diffusion: Diffusion,
    clip_encoder: CLIPTextEncoder,
    vae_decoder: AutoencoderKL,
    prompts: list[str],
    device: torch.device,
    guidance_scale: float = 7.5,
    latent_size: int = 8,
    latent_channels: int = 16,
) -> torch.Tensor:
    """Generate validation samples using latent-space diffusion.

    Args:
        model: Diffusion model
        diffusion: Diffusion process
        clip_encoder: CLIP text encoder
        vae_decoder: VAE decoder for latent-to-image conversion
        prompts: List of text prompts
        device: Device to generate on
        guidance_scale: CFG guidance scale
        latent_size: Latent spatial size
        latent_channels: Latent channels

    Returns:
        Generated images tensor (B, 3, H, W) in [0, 1]
    """
    model.eval()

    text_embeds = clip_encoder.encode(prompts)
    text_embeds = text_embeds.to(device)

    original_scale = diffusion.guidance_scale
    diffusion.guidance_scale = guidance_scale

    # Sample in latent space using Euler ODE solver, decode to image
    images = diffusion.sample(
        model=model,
        shape=(len(prompts), latent_channels, latent_size, latent_size),
        text_embeds=text_embeds,
        num_steps=50,
        use_cfg=True,
        vae_decoder=vae_decoder,
    )

    diffusion.guidance_scale = original_scale
    return images


def train_diffusion(config: dict[str, Any], use_wandb: bool = False) -> None:
    """Main diffusion training function (latent-space).

    Args:
        config: Training configuration dictionary
        use_wandb: Enable wandb logging
    """
    from src.data.loader import create_dataloader, get_dataset
    from src.models.factory import DiT
    from src.utils.common import get_device, set_seed

    print("=" * 60)
    print("STAGE 2: DIFFUSION TRAINING (Latent Space)")
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
                name=config.get("wandb_run_name", "diffusion-training"),
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

    # Load pre-trained VAE
    vae_checkpoint = config.get("vae_checkpoint", "checkpoints/vae.pt")
    if not Path(vae_checkpoint).exists():
        print(f"Error: VAE checkpoint not found: {vae_checkpoint}")
        print("Please train VAE first using --train-vae")
        return

    print(f"Loading VAE from {vae_checkpoint}...")
    vae = create_vae(
        image_size=config["image_size"],
        z_channels=config.get("latent_channels", config.get("in_channels", 16)),
        ch=config.get("vae_ch", 64),
        ch_mult=tuple(config.get("vae_ch_mult", [1, 2, 4, 4])),
    )
    vae_state = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state["model_state_dict"])
    vae = vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Set scaling factor for latent normalization
    scaling_factor_config = config.get("scaling_factor", 1.0)
    if scaling_factor_config == "auto":
        print("Computing scaling factor from dataset...")
        scaling_factor = vae.compute_scaling_factor(dataloader, num_batches=100)
        vae.set_scaling_factor(scaling_factor)
    elif isinstance(scaling_factor_config, (int, float)):
        vae.set_scaling_factor(float(scaling_factor_config))
    print(f"VAE loaded and frozen (scaling_factor={vae.scaling_factor:.4f})")

    # Load CLIP encoder
    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Compute unconditional embedding
    print("Computing unconditional embedding...")
    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

    # Initialize DiT model for latent space
    latent_size = config.get("latent_size", config["image_size"] // 8)
    in_channels = config.get("in_channels", 16)

    print(f"Initializing DiT-{config['model_size']} for latent space...")
    print(f"  Latent size: {latent_size}x{latent_size}")
    print(f"  Latent channels: {in_channels}")

    model = DiT(
        in_channels=in_channels,
        image_size=latent_size,
        patch_size=config["patch_size"],
        model_size=config["model_size"],
        clip_embed_dim=clip_encoder.embedding_dim,
        model_type=config.get("model_type", "dit"),
        qk_rmsnorm=config.get("qk_rmsnorm", True),
        register_tokens=config.get("register_tokens", 0),
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"DiT parameters: {num_params / 1_000_000:.2f}M")

    # Initialize EMA
    ema = None
    if config["use_ema"]:
        ema = EMA(model, decay=config["ema_decay"])
        ema.to(device)
        print(f"EMA enabled with decay={config['ema_decay']}")

    # Initialize Rectified Flow diffusion
    diffusion = Diffusion(
        num_timesteps=config["num_timesteps"],
        guidance_scale=config["guidance_scale"],
        cfg_probability=config.get("cfg_prob", config.get("initial_cfg_prob", 0.1)),
        uncond_embed=uncond_embed,
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
    # Handle streaming datasets that don't have length
    try:
        num_steps_per_epoch = len(dataloader)
    except TypeError:
        # IterableDataset doesn't have __len__, use config or default
        num_steps_per_epoch = config.get("steps_per_epoch", 1000)
        print(f"Streaming dataset: using {num_steps_per_epoch} steps per epoch")
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
    resume = config.get("resume", False)
    checkpoint_path = Path(config["checkpoint_path"])

    if resume and checkpoint_path.exists():
        print(f"\nResuming training from checkpoint: {checkpoint_path}")
        checkpoint_info = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
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

    # CFG warmup
    initial_cfg = config.get("initial_cfg_prob", 0.0)
    final_cfg = config.get("final_cfg_prob", config.get("cfg_prob", 0.1))
    cfg_warmup = config.get("cfg_warmup_epochs", 0)

    diffusion.cfg_probability = initial_cfg
    print(f"Initial CFG probability: {initial_cfg}")

    # Training loop
    if start_epoch > 0:
        print(f"\nResuming diffusion training from epoch {start_epoch + 1}/{config['epochs']}...")
    else:
        print(f"\nStarting diffusion training for {config['epochs']} epochs...")

    for epoch in range(start_epoch, config["epochs"]):
        # CFG warmup
        if cfg_warmup > 0 and epoch < cfg_warmup:
            progress = epoch / cfg_warmup
            current_cfg = initial_cfg + (final_cfg - initial_cfg) * progress
            diffusion.cfg_probability = current_cfg
            if epoch % 10 == 0:
                print(f"  CFG warmup: {current_cfg:.3f} (epoch {epoch}/{cfg_warmup})")
        elif cfg_warmup > 0:
            diffusion.cfg_probability = final_cfg

        avg_loss, global_step = train_one_epoch(
            model=model,
            diffusion=diffusion,
            dataloader=dataloader,
            clip_encoder=clip_encoder,
            vae_encoder=vae,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
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
                    "epoch/avg_loss": avg_loss,
                    "epoch/epoch": epoch + 1,
                    "epoch/cfg_probability": diffusion.cfg_probability,
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
            ema=ema,
            global_step=global_step,
            scaling_factor=vae.scaling_factor,
        )
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Save periodic checkpoint every 10 epochs (weights only for minimal size)
        checkpoint_interval = config.get("checkpoint_interval", 10)
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
            periodic_path = checkpoint_dir / f"diffusion_epoch_{epoch + 1}.pt"
            periodic_checkpoint = {"model_state_dict": model.state_dict()}
            if ema is not None:
                periodic_checkpoint["ema_state_dict"] = ema.state_dict()
            torch.save(periodic_checkpoint, periodic_path)
            print(f"Saved periodic checkpoint: {periodic_path}")

        # Generate validation samples
        if (epoch + 1) % config.get("validation_interval", 10) == 0:
            print("\nGenerating validation samples...")
            samples = generate_samples(
                model=model,
                diffusion=diffusion,
                clip_encoder=clip_encoder,
                vae_decoder=vae,
                prompts=config["validation_prompts"],
                device=device,
                guidance_scale=config["guidance_scale"],
                latent_size=latent_size,
                latent_channels=in_channels,
            )

            sample_dir = Path(config.get("sample_dir", "samples")) / f"epoch_{epoch + 1}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            for i, prompt in enumerate(config["validation_prompts"]):
                img = samples[i]
                img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                safe_prompt = prompt.replace(" ", "_")[:20]
                img.save(sample_dir / f"{i:02d}_{safe_prompt}.png")

            print(f"Saved samples to {sample_dir}")

    print("\n" + "=" * 60)
    print("Diffusion Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print("=" * 60)

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
