#!/usr/bin/env python3
"""Interactive demo for text-to-emoji.

Generate pixel art emojis from text prompts interactively.

Usage:
    python demo.py --checkpoint checkpoints/model_best.pt

    # Then enter prompts interactively:
    > rocket
    > cute robot
    > smiling sun
    > quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from src.models.diffusion import Diffusion
from src.models import DiT
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.config import DiffusionConfig, ModelConfig


def get_device() -> torch.device:
    """Automatically select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[DiT, Diffusion, CLIPTextEncoder, dict]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.

    Returns:
        Tuple of (model, diffusion, clip_encoder, model_config).
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", {})
    model_size = model_config.get("model_size", "S")
    patch_size = model_config.get("patch_size", 2)
    image_size = model_config.get("image_size", 32)
    model_type = model_config.get("model_type", "dit")
    qk_rmsnorm = model_config.get("qk_rmsnorm", True)
    register_tokens = model_config.get("register_tokens", 0)

    print(f"Model: {model_type.upper()}-{model_size}, patch_size={patch_size}, image_size={image_size}")

    # Load CLIP encoder
    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Initialize model
    model = DiT(
        in_channels=3,
        image_size=image_size,
        patch_size=patch_size,
        model_size=model_size,
        clip_embed_dim=clip_encoder.embedding_dim,
        model_type=model_type,
        qk_rmsnorm=qk_rmsnorm,
        register_tokens=register_tokens,
    )

    # Load weights (try EMA first, then regular)
    if "ema_state_dict" in checkpoint:
        print("Loading EMA weights...")
        ema_state = checkpoint["ema_state_dict"]
        shadow_params = ema_state.get("shadow_params", {})
        state_dict = {k: v for k, v in shadow_params.items()}
        model.load_state_dict(state_dict)
    else:
        print("Loading model weights...")
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    # Initialize diffusion
    diffusion_config = DiffusionConfig()
    diffusion = Diffusion(
        num_timesteps=diffusion_config.num_timesteps,
        beta_schedule=diffusion_config.beta_schedule,
    )

    return model, diffusion, clip_encoder, model_config


def generate_image(
    prompt: str,
    model: DiT,
    diffusion: Diffusion,
    clip_encoder: CLIPTextEncoder,
    device: torch.device,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int | None = None,
) -> Image.Image:
    """Generate a single image from a text prompt.

    Args:
        prompt: Text prompt for generation.
        model: DiT model.
        diffusion: Diffusion process.
        clip_encoder: CLIP text encoder.
        device: Device to generate on.
        num_steps: Number of sampling steps.
        guidance_scale: CFG guidance scale.
        seed: Random seed for reproducibility.

    Returns:
        Generated PIL Image (upscaled to 256x256).
    """
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        text_embeds = clip_encoder.encode([prompt])

        diffusion.guidance_scale = guidance_scale

        images = diffusion.sample(
            model=model,
            shape=(1, 3, 32, 32),
            text_embeds=text_embeds,
            num_steps=num_steps,
            use_ddim=True,
            use_cfg=True,
            seed=seed,
        )

    # Convert to PIL Image
    img = images[0].cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    pil_img = Image.fromarray(img)

    # Upscale with nearest neighbor for pixel art
    pil_img = pil_img.resize((256, 256), Image.NEAREST)

    return pil_img


def generate_batch(
    prompts: list[str],
    model: DiT,
    diffusion: Diffusion,
    clip_encoder: CLIPTextEncoder,
    device: torch.device,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int | None = None,
) -> list[Image.Image]:
    """Generate multiple images from a list of prompts.

    Args:
        prompts: List of text prompts.
        model: DiT model.
        diffusion: Diffusion process.
        clip_encoder: CLIP text encoder.
        device: Device to generate on.
        num_steps: Number of sampling steps.
        guidance_scale: CFG guidance scale.
        seed: Random seed for reproducibility.

    Returns:
        List of generated PIL Images.
    """
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        text_embeds = clip_encoder.encode(prompts)

        diffusion.guidance_scale = guidance_scale

        images = diffusion.sample(
            model=model,
            shape=(len(prompts), 3, 32, 32),
            text_embeds=text_embeds,
            num_steps=num_steps,
            use_ddim=True,
            use_cfg=True,
            seed=seed,
        )

    pil_images = []
    for img in images:
        img_np = img.cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img_np = img_np.transpose(1, 2, 0)
        pil_img = Image.fromarray(img_np)
        pil_img = pil_img.resize((256, 256), Image.NEAREST)
        pil_images.append(pil_img)

    return pil_images


def create_grid(images: list[Image.Image], cols: int = 4) -> Image.Image:
    """Create a grid of images.

    Args:
        images: List of PIL Images (same size).
        cols: Number of columns in grid.

    Returns:
        Grid image.
    """
    if not images:
        return Image.new("RGB", (256, 256), (255, 255, 255))

    rows = (len(images) + cols - 1) // cols
    img_w, img_h = images[0].size

    grid = Image.new("RGB", (cols * img_w, rows * img_h), (255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * img_w, row * img_h))

    return grid


def interactive_mode(
    model: DiT,
    diffusion: Diffusion,
    clip_encoder: CLIPTextEncoder,
    device: torch.device,
    output_dir: Path,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
) -> None:
    """Run interactive generation mode.

    Args:
        model: DiT model.
        diffusion: Diffusion process.
        clip_encoder: CLIP text encoder.
        device: Device to generate on.
        output_dir: Directory to save generated images.
        num_steps: Number of sampling steps.
        guidance_scale: CFG guidance scale.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 50)
    print("text-to-emoji Interactive Demo")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Steps: {num_steps}, Guidance: {guidance_scale}")
    print(f"Output: {output_dir}")
    print("\nCommands:")
    print("  <prompt>     - Generate image from prompt")
    print("  batch <p1>, <p2>, ... - Generate multiple images")
    print("  steps <n>    - Set number of sampling steps")
    print("  guidance <g> - Set guidance scale")
    print("  seed <s>     - Set random seed (or 'none' for random)")
    print("  quit/exit    - Exit demo")
    print("=" * 50 + "\n")

    seed = None
    img_count = 0

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Parse commands
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        elif user_input.lower().startswith("steps "):
            try:
                num_steps = int(user_input.split()[1])
                print(f"Set steps to {num_steps}")
            except (ValueError, IndexError):
                print("Usage: steps <number>")
            continue

        elif user_input.lower().startswith("guidance "):
            try:
                guidance_scale = float(user_input.split()[1])
                print(f"Set guidance scale to {guidance_scale}")
            except (ValueError, IndexError):
                print("Usage: guidance <number>")
            continue

        elif user_input.lower().startswith("seed "):
            try:
                seed_str = user_input.split()[1]
                if seed_str.lower() == "none":
                    seed = None
                    print("Set seed to random")
                else:
                    seed = int(seed_str)
                    print(f"Set seed to {seed}")
            except (ValueError, IndexError):
                print("Usage: seed <number|none>")
            continue

        elif user_input.lower().startswith("batch "):
            prompts = [p.strip() for p in user_input[6:].split(",")]
            prompts = [p for p in prompts if p]
            if not prompts:
                print("Usage: batch <prompt1>, <prompt2>, ...")
                continue

            print(f"Generating {len(prompts)} images...")
            try:
                images = generate_batch(
                    prompts=prompts,
                    model=model,
                    diffusion=diffusion,
                    clip_encoder=clip_encoder,
                    device=device,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )

                # Save individual images
                for img, prompt in zip(images, prompts):
                    img_count += 1
                    safe_prompt = prompt.replace(" ", "_")[:20]
                    filename = f"{img_count:04d}_{safe_prompt}.png"
                    img.save(output_dir / filename)

                # Save grid
                grid = create_grid(images)
                grid_filename = f"grid_{img_count:04d}.png"
                grid.save(output_dir / grid_filename)

                print(f"Saved {len(images)} images and grid to {output_dir}")

            except Exception as e:
                print(f"Error generating images: {e}")
            continue

        # Single prompt generation
        prompt = user_input
        print(f"Generating: '{prompt}'...")

        try:
            img = generate_image(
                prompt=prompt,
                model=model,
                diffusion=diffusion,
                clip_encoder=clip_encoder,
                device=device,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

            img_count += 1
            safe_prompt = prompt.replace(" ", "_")[:20]
            filename = f"{img_count:04d}_{safe_prompt}.png"
            filepath = output_dir / filename
            img.save(filepath)

            print(f"Saved: {filepath}")

        except Exception as e:
            print(f"Error generating image: {e}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="text-to-emoji Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode
    python demo.py --checkpoint checkpoints/model_best.pt

    # Generate single image
    python demo.py --checkpoint checkpoints/model_best.pt --prompt "rocket"

    # Generate multiple images from file
    python demo.py --checkpoint checkpoints/model_best.pt --prompts-file prompts.txt

    # Batch generation
    python demo.py --checkpoint checkpoints/model_best.pt --batch "rocket, cat, star"
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for generation (non-interactive)",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Comma-separated prompts for batch generation",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/demo",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Train a model first with: ./scripts/train.sh")
        sys.exit(1)

    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    # Load model
    model, diffusion, clip_encoder, model_config = load_model(str(checkpoint_path), device)

    output_dir = Path(args.output_dir)

    # Handle different modes
    if args.prompt:
        # Single prompt mode
        print(f"Generating: '{args.prompt}'...")
        img = generate_image(
            prompt=args.prompt,
            model=model,
            diffusion=diffusion,
            clip_encoder=clip_encoder,
            device=device,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        safe_prompt = args.prompt.replace(" ", "_")[:20]
        filename = f"{safe_prompt}.png"
        filepath = output_dir / filename
        img.save(filepath)
        print(f"Saved: {filepath}")

    elif args.batch:
        # Batch mode from command line
        prompts = [p.strip() for p in args.batch.split(",")]
        prompts = [p for p in prompts if p]

        print(f"Generating {len(prompts)} images...")
        images = generate_batch(
            prompts=prompts,
            model=model,
            diffusion=diffusion,
            clip_encoder=clip_encoder,
            device=device,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            safe_prompt = prompt.replace(" ", "_")[:20]
            filename = f"{i:02d}_{safe_prompt}.png"
            img.save(output_dir / filename)

        grid = create_grid(images)
        grid.save(output_dir / "grid.png")
        print(f"Saved {len(images)} images and grid to {output_dir}")

    elif args.prompts_file:
        # Batch mode from file
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            print(f"Error: Prompts file not found: {prompts_path}")
            sys.exit(1)

        prompts = [
            line.strip()
            for line in prompts_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

        print(f"Generating {len(prompts)} images from {prompts_path}...")
        images = generate_batch(
            prompts=prompts,
            model=model,
            diffusion=diffusion,
            clip_encoder=clip_encoder,
            device=device,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            safe_prompt = prompt.replace(" ", "_")[:20]
            filename = f"{i:02d}_{safe_prompt}.png"
            img.save(output_dir / filename)

        grid = create_grid(images)
        grid.save(output_dir / "grid.png")
        print(f"Saved {len(images)} images and grid to {output_dir}")

    else:
        # Interactive mode
        interactive_mode(
            model=model,
            diffusion=diffusion,
            clip_encoder=clip_encoder,
            device=device,
            output_dir=output_dir,
            num_steps=args.steps,
            guidance_scale=args.guidance,
        )


if __name__ == "__main__":
    main()
