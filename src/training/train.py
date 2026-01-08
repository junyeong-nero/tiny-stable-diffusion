"""Training script for PixMoji-Diffusion."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ModelConfig, DiffusionConfig, TrainingConfig, DataConfig, get_parser
from src.data.dataset import EmojiDataset
from src.models.dit import DiT
from src.models.diffusion import Diffusion
from src.text_encoder.clip_encoder import CLIPTextEncoder


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
        learning_rate=getattr(args, "learning_rate", 1e-4) or 1e-4,
    )
    data_config = DataConfig(
        source=getattr(args, "data_source", "huggingface") or "huggingface",
        dataset_name=getattr(args, "dataset_name", "junyeong-nero/emoji-32") or "junyeong-nero/emoji-32",
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
            image_size=data_config.image_size,
            streaming=data_config.streaming,
        )
    else:
        print(f"Loading local dataset from: data/")
        from src.data.dataset import LocalEmojiDataset

        dataset = LocalEmojiDataset(
            data_dir="data",
            image_size=data_config.image_size,
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

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs * len(dataloader),
        eta_min=training_config.min_lr,
    )

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

            # Sample random timesteps
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

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % training_config.checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
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
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = output_dir / "model_final.pt"
    torch.save({
        "epoch": training_config.epochs,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_size": model_config.model_size,
            "patch_size": model_config.patch_size,
            "image_size": model_config.image_size,
        },
    }, final_path)
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
