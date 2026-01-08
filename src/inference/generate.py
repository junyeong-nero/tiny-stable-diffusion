"""Inference script for generating emoji images from text prompts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import get_parser, InferenceConfig
from src.models.dit import DiT
from src.models.diffusion import Diffusion
from src.text_encoder.clip_encoder import CLIPTextEncoder


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Main inference function."""
    parser = get_parser()
    args = parser.parse_args()

    # Get prompt from args
    prompt = getattr(args, "prompt", None)
    if prompt is None:
        print("Error: --prompt argument is required")
        return

    # Set random seed
    seed = getattr(args, "seed", 42) or 42
    set_seed(seed)

    # Load configuration
    inference_config = InferenceConfig(
        checkpoint=getattr(args, "checkpoint", None),
        num_samples=getattr(args, "num_samples", 4) or 4,
        num_steps=getattr(args, "steps", 50) or 50,
        guidance_scale=getattr(args, "guidance_scale", 7.5) or 7.5,
        seed=seed,
    )

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CLIP encoder
    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Encode prompt
    print(f"Encoding prompt: '{prompt}'")
    text_embeds = clip_encoder.encode([prompt] * inference_config.num_samples)
    text_embeds = text_embeds.to(device)

    # Initialize diffusion
    diffusion = Diffusion(
        num_timesteps=1000,
        beta_schedule="cosine",
        guidance_scale=inference_config.guidance_scale,
    )

    # Load model
    checkpoint_path = inference_config.checkpoint
    if checkpoint_path is None:
        # Look for latest checkpoint
        checkpoints_dir = Path("checkpoints")
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            print("Error: No checkpoint found!")
            print("Please train the model first or provide --checkpoint path")
            return
    else:
        checkpoint_path = Path(checkpoint_path)

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get("model_config", {})
    model_size = model_config.get("model_size", "S")
    patch_size = model_config.get("patch_size", 2)
    image_size = model_config.get("image_size", 32)

    # Initialize model
    model = DiT(
        in_channels=3,
        image_size=image_size,
        patch_size=patch_size,
        model_size=model_size,
        clip_embed_dim=clip_encoder.embedding_dim,
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Print model info
    model_info = model.get_model_size_info()
    print(f"Model: DiT-{model_size} with {model_info['num_parameters']:,} parameters")

    # Generate images
    print(f"Generating {inference_config.num_samples} images...")
    shape = (
        inference_config.num_samples,
        3,
        image_size,
        image_size,
    )

    with torch.no_grad():
        images = diffusion.sample(
            model=model,
            shape=shape,
            text_embeds=text_embeds,
            num_steps=inference_config.num_steps,
            use_ddim=getattr(args, "ddim", True),
            use_cfg=True,
            seed=seed,
        )

    # Save images
    output_dir = Path(inference_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torchvision
    from PIL import Image

    # Convert to PIL images and save
    for i, img_tensor in enumerate(images):
        # Denormalize from [-1, 1] to [0, 255]
        img = (img_tensor * 255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8)
        img = Image.fromarray(img.cpu().numpy())

        # Save at original size
        original_path = output_dir / f"{prompt.replace(' ', '_')}_{i}.png"
        img.save(original_path)
        print(f"Saved: {original_path}")

        # Also save upscaled version
        upscale_size = inference_config.upscale_size
        img_upscaled = img.resize((upscale_size, upscale_size), Image.NEAREST)
        upscaled_path = output_dir / f"{prompt.replace(' ', '_')}_{i}_upscaled.png"
        img_upscaled.save(upscaled_path)
        print(f"Saved upscaled: {upscaled_path}")

    print("Done!")


if __name__ == "__main__":
    main()
