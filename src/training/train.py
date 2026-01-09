"""Training script for PixMoji-Diffusion."""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    get_parser,
)
from src.data.dataset import EmojiDataset
from src.models.diffusion import Diffusion
from src.models.dit import DiT
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.ema import EMA


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    """Get torch device."""
    if device == "cuda":
        return torch.device("cuda")
    elif device == "mps":
        return torch.device("mps")
    else:
        return torch.device("cpu")


def generate_validation_samples(
    model: nn.Module,
    diffusion: Diffusion,
    clip_encoder: CLIPTextEncoder,
    prompts: list[str],
    device: torch.device,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
) -> torch.Tensor:
    """Generate validation samples from prompts.

    Args:
        model: DiT model.
        diffusion: Diffusion process.
        clip_encoder: CLIP text encoder.
        prompts: List of text prompts.
        device: Device to generate on.
        num_steps: Number of sampling steps.
        guidance_scale: Classifier-free guidance scale.

    Returns:
        Generated images tensor (B, C, H, W) in [0, 1] range.
    """
    model.eval()
    with torch.no_grad():
        text_embeds = clip_encoder.encode(prompts)

        # Set guidance scale temporarily
        original_scale = diffusion.guidance_scale
        diffusion.guidance_scale = guidance_scale

        images = diffusion.sample(
            model=model,
            shape=(len(prompts), 3, 32, 32),
            text_embeds=text_embeds,
            num_steps=num_steps,
            use_ddim=True,
            use_cfg=True,
        )

        diffusion.guidance_scale = original_scale

    model.train()
    return images


def save_validation_images(
    images: torch.Tensor,
    prompts: list[str],
    output_dir: Path,
    epoch: int,
    suffix: str = "",
) -> None:
    """Save validation images to disk.

    Args:
        images: Generated images (B, C, H, W) in [0, 1] range.
        prompts: List of prompts used.
        output_dir: Directory to save images.
        epoch: Current epoch number.
        suffix: Optional suffix for filename (e.g., '_ema').
    """
    from PIL import Image
    import numpy as np

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, prompt) in enumerate(zip(images, prompts)):
        # Convert to PIL Image
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        img_np = img_np.transpose(1, 2, 0)  # CHW -> HWC
        pil_img = Image.fromarray(img_np)

        # Upscale with nearest neighbor for pixel art
        pil_img = pil_img.resize((256, 256), Image.NEAREST)

        # Save with sanitized prompt name
        safe_prompt = prompt.replace(" ", "_")[:30]
        filename = f"epoch{epoch:03d}_{i}_{safe_prompt}{suffix}.png"
        pil_img.save(samples_dir / filename)


def main() -> None:
    """Main training function."""
    parser = get_parser()
    parser.add_argument(
        "--data-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "local"],
        help="Dataset source: huggingface or local",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="junyeong-nero/emoji-32",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (train, validation, test)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use streaming mode for large datasets",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use Exponential Moving Average for model weights",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="EMA decay rate",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=5,
        help="Generate validation samples every N epochs",
    )
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed or 42)

    # Get device
    device = get_device(args.device or "auto")
    print(f"Using device: {device}")

    # Load configurations
    model_config = ModelConfig(
        model_size=getattr(args, "model_size", "S") or "S",
        patch_size=getattr(args, "patch_size", 2) or 2,
    )
    diffusion_config = DiffusionConfig()
    training_config = TrainingConfig(
        epochs=getattr(args, "epochs", 100) or 100,
        batch_size=getattr(args, "batch_size", 64) or 64,
        learning_rate=getattr(args, "learning_rate", 5e-4) or 5e-4,
    )
    data_config = DataConfig(
        source=getattr(args, "data_source", "huggingface") or "huggingface",
        dataset_name=getattr(args, "dataset_name", "junyeong-nero/emoji-32")
        or "junyeong-nero/emoji-32",
        split=getattr(args, "split", "train") or "train",
        streaming=getattr(args, "streaming", False) or False,
    )

    # Create output directory
    output_dir = Path(getattr(args, "output_dir", "checkpoints") or "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if data_config.source == "huggingface":
        print(f"Loading Hugging Face dataset: {data_config.dataset_name}")
        dataset = EmojiDataset(
            dataset_name=data_config.dataset_name,
            split=data_config.split,
            streaming=data_config.streaming,
        )
    else:
        print("Loading local dataset from: data/")
        from src.data.dataset import LocalEmojiDataset

        dataset = LocalEmojiDataset(
            data_dir="data",
        )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )

    # Load CLIP encoder
    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Initialize model
    print(f"Initializing DiT-{model_config.model_size}...")
    model = DiT(
        in_channels=model_config.in_channels,
        image_size=model_config.image_size,
        patch_size=model_config.patch_size,
        model_size=model_config.model_size,
        clip_embed_dim=clip_encoder.embedding_dim,
    )
    model = model.to(device)

    # Print model info
    model_info = model.get_model_size_info()
    print(f"Model parameters: {model_info['num_parameters']:,}")

    # Initialize EMA
    ema = None
    if getattr(args, "use_ema", True):
        ema_decay = getattr(args, "ema_decay", 0.9999)
        ema = EMA(model, decay=ema_decay)
        ema.to(device)
        print(f"EMA enabled with decay={ema_decay}")

    # Initialize wandb
    project_config = ProjectConfig(
        name=getattr(args, "name", None) or "pixmoji-diffusion",
        experiment_name=getattr(args, "name", None) or f"dit-{model_config.model_size}",
        seed=args.seed or 42,
        use_wandb=getattr(args, "wandb", False) or False,
    )

    wandb_run = None
    if project_config.use_wandb:
        try:
            import wandb

            wandb.login()
            wandb_run = wandb.init(
                project=project_config.name,
                name=project_config.experiment_name,
                config={
                    "model_size": model_config.model_size,
                    "patch_size": model_config.patch_size,
                    "image_size": model_config.image_size,
                    "epochs": training_config.epochs,
                    "batch_size": training_config.batch_size,
                    "learning_rate": training_config.learning_rate,
                    "num_timesteps": diffusion_config.num_timesteps,
                    "beta_schedule": diffusion_config.beta_schedule,
                    "guidance_scale": diffusion_config.guidance_scale,
                    "cfg_probability": diffusion_config.cfg_probability,
                    "dataset": data_config.dataset_name,
                    "seed": project_config.seed,
                    "use_ema": getattr(args, "use_ema", True),
                    "ema_decay": getattr(args, "ema_decay", 0.9999),
                },
            )
            print(f"Initialized W&B: {wandb.run.url}")
        except ImportError:
            print("wandb not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")

    # Initialize diffusion
    diffusion = Diffusion(
        num_timesteps=diffusion_config.num_timesteps,
        beta_schedule=diffusion_config.beta_schedule,
        guidance_scale=diffusion_config.guidance_scale,
        cfg_probability=diffusion_config.cfg_probability,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=training_config.betas,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
    )

    # Learning rate scheduler with warmup
    num_steps_per_epoch = len(dataloader)
    total_steps = training_config.epochs * num_steps_per_epoch
    warmup_steps = training_config.warmup_steps

    def lr_lambda(step: int) -> float:
        """Learning rate schedule: warmup + cosine decay."""
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    # Validation prompts
    validation_prompts = [
        "rocket",
        "cat",
        "robot",
        "star",
    ]
    validation_interval = getattr(args, "validation_interval", 5)

    # Track best loss
    best_loss = float("inf")

    # Training loop
    print(f"Starting training for {training_config.epochs} epochs...")
    for epoch in range(training_config.epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{training_config.epochs}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            captions = batch["caption"]

            # Get text embeddings
            with torch.no_grad():
                text_embeds = clip_encoder.encode(captions)

            # Sample random timesteps for noise prediction
            timesteps = torch.randint(
                0,
                diffusion.num_timesteps,
                (images.shape[0],),
                device=device,
            )

            # Calculate loss
            loss = diffusion.training_loss(model, images, timesteps, text_embeds)

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            if training_config.gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training_config.gradient_clip_val,
                )

            optimizer.step()
            scheduler.step()

            # Update EMA
            if ema is not None:
                ema.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")

        # Track best model
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Log to wandb
        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "train_loss": avg_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                }
            )

        # Generate validation samples
        if (epoch + 1) % validation_interval == 0:
            print("Generating validation samples...")

            # Generate with training model
            val_images = generate_validation_samples(
                model=model,
                diffusion=diffusion,
                clip_encoder=clip_encoder,
                prompts=validation_prompts,
                device=device,
            )
            save_validation_images(
                val_images, validation_prompts, output_dir, epoch + 1
            )

            # Generate with EMA model if available
            if ema is not None:
                ema.apply_shadow()
                ema_images = generate_validation_samples(
                    model=model,
                    diffusion=diffusion,
                    clip_encoder=clip_encoder,
                    prompts=validation_prompts,
                    device=device,
                )
                save_validation_images(
                    ema_images, validation_prompts, output_dir, epoch + 1, suffix="_ema"
                )
                ema.restore()

            # Log images to wandb
            if wandb_run is not None:
                import wandb

                wandb.log({
                    "validation_samples": [
                        wandb.Image(img.cpu(), caption=prompt)
                        for img, prompt in zip(val_images, validation_prompts)
                    ],
                    "epoch": epoch + 1,
                })

            print(f"Validation samples saved to {output_dir / 'samples'}")

        # Save checkpoint
        if (epoch + 1) % training_config.checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "model_config": {
                    "model_size": model_config.model_size,
                    "patch_size": model_config.patch_size,
                    "image_size": model_config.image_size,
                },
            }
            if ema is not None:
                checkpoint_data["ema_state_dict"] = ema.state_dict()

            torch.save(checkpoint_data, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best and training_config.save_best:
            best_path = output_dir / "model_best.pt"
            best_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
                "model_config": {
                    "model_size": model_config.model_size,
                    "patch_size": model_config.patch_size,
                    "image_size": model_config.image_size,
                },
            }
            if ema is not None:
                best_data["ema_state_dict"] = ema.state_dict()

            torch.save(best_data, best_path)
            print(f"Saved best model (loss={avg_loss:.4f}): {best_path}")

    # Save final model
    final_path = output_dir / "model_final.pt"
    final_data = {
        "epoch": training_config.epochs,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_size": model_config.model_size,
            "patch_size": model_config.patch_size,
            "image_size": model_config.image_size,
        },
    }
    if ema is not None:
        final_data["ema_state_dict"] = ema.state_dict()

    torch.save(final_data, final_path)
    print(f"Saved final model: {final_path}")

    # Finish wandb
    if wandb_run is not None:
        import wandb

        artifact = wandb.Artifact("pixmoji-model", type="model")
        artifact.add_file(str(final_path))
        wandb.log_artifact(artifact)

        wandb.finish()
        print("W&B run finished")


if __name__ == "__main__":
    main()
