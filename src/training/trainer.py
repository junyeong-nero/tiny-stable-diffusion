"""Training utilities for text-to-emoji."""

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
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.checkpoint import save_checkpoint
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
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    ema: EMA | None = None,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_wandb: bool = False,
    global_step: int = 0,
) -> tuple[float, int]:
    """Train for one epoch.

    Args:
        model: Model to train
        diffusion: Diffusion process
        dataloader: Training data loader
        clip_encoder: CLIP text encoder
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
    epoch_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        images = batch["image"].to(device)
        captions = batch["caption"]

        with torch.no_grad():
            text_embeds = clip_encoder.encode(captions)
            text_embeds = text_embeds.to(device)

        timesteps = torch.randint(
            0,
            diffusion.num_timesteps,
            (images.shape[0],),
            device=device,
        )

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss = diffusion.training_loss(model, images, timesteps, text_embeds)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = diffusion.training_loss(model, images, timesteps, text_embeds)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update()

        loss_value = loss.item()
        epoch_loss += loss_value
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

    return epoch_loss / len(dataloader), global_step


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    diffusion: Diffusion,
    clip_encoder: CLIPTextEncoder,
    prompts: list[str],
    device: torch.device,
    guidance_scale: float = 7.5,
    image_size: int = 32,
) -> torch.Tensor:
    """Generate validation samples.

    Args:
        model: Model to use for generation
        diffusion: Diffusion process
        clip_encoder: CLIP text encoder
        prompts: List of text prompts
        device: Device to generate on
        guidance_scale: CFG guidance scale
        image_size: Output image size

    Returns:
        Generated images tensor
    """
    model.eval()

    text_embeds = clip_encoder.encode(prompts)
    text_embeds = text_embeds.to(device)

    original_scale = diffusion.guidance_scale
    diffusion.guidance_scale = guidance_scale

    images = diffusion.sample(
        model=model,
        shape=(len(prompts), 3, image_size, image_size),
        text_embeds=text_embeds,
        num_steps=50,
        use_ddim=True,
        use_cfg=True,
    )

    diffusion.guidance_scale = original_scale
    return images


def train(config: dict[str, Any], use_wandb: bool = False) -> None:
    """Main training function.

    Args:
        config: Training configuration dictionary
        use_wandb: Enable wandb logging
    """
    from src.data.loader import create_dataloader, get_dataset
    from src.models.factory import DiT
    from src.utils.common import get_device, set_seed

    training_stage = config.get("training_stage", "pretrain")
    if training_stage == "pretrain":
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

    # Initialize wandb
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Disabling wandb logging.")
            use_wandb = False
        else:
            wandb.init(
                project=config.get("wandb_project", "text-to-emoji"),
                name=config.get("wandb_run_name", None),
                config=config,
            )
            print("Wandb logging enabled")

    set_seed(config["seed"])
    device = get_device(config["device"])
    print(f"Using device: {device}")

    # Load dataset
    dataset = get_dataset(config)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    dataloader = create_dataloader(
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

    # Compute unconditional embedding
    print("Computing unconditional embedding...")
    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

    # Initialize model
    print(f"Initializing DiT-{config['model_size']}...")
    model = DiT(
        in_channels=3,
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        model_size=config["model_size"],
        clip_embed_dim=clip_encoder.embedding_dim,
        model_type=config.get("model_type", "dit"),
        qk_rmsnorm=config.get("qk_rmsnorm", True),
        register_tokens=config.get("register_tokens", 0),
    )
    model = model.to(device)

    # Load pretrained checkpoint if available
    pretrain_checkpoint = config.get("pretrain_checkpoint")
    if pretrain_checkpoint and Path(pretrain_checkpoint).exists():
        print(f"Loading pretrained checkpoint: {pretrain_checkpoint}")
        checkpoint = torch.load(pretrain_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Pretrained weights loaded")

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

    # Mixed precision
    use_amp = config["mixed_precision"] and device.type == "cuda"
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
    print(f"\nStarting training for {config['epochs']} epochs...")
    best_loss = float("inf")
    global_step = 0

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

        avg_loss, global_step = train_one_epoch(
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
                ema=ema,
            )

        # Generate validation samples
        if (epoch + 1) % config["validation_interval"] == 0:
            print("\nGenerating validation samples...")
            samples = generate_samples(
                model=model,
                diffusion=diffusion,
                clip_encoder=clip_encoder,
                prompts=config["validation_prompts"],
                device=device,
                guidance_scale=config["guidance_scale"],
                image_size=config["image_size"],
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

            print(f"Saved samples to {sample_dir}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    print("=" * 60)

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
