"""Image generation utilities for text-to-emoji."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.models.diffusion import Diffusion
from src.models.factory import DiT
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.checkpoint import find_latest_checkpoint
from src.utils.common import get_device, set_seed


@torch.no_grad()
def generate(
    prompts: list[str],
    checkpoint: str | Path | None = None,
    num_samples: int = 1,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    device: str = "auto",
) -> list[Image.Image]:
    """Generate images from text prompts.

    Args:
        prompts: List of text prompts
        checkpoint: Path to model checkpoint (auto-detects if None)
        num_samples: Number of samples per prompt
        num_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        device: Device to use ("auto", "cuda", "mps", "cpu")

    Returns:
        List of generated PIL images
    """
    if seed is not None:
        set_seed(seed)

    device = get_device(device)
    print(f"Using device: {device}")

    # Load CLIP encoder
    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Compute unconditional embedding
    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

    # Find checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint("checkpoints")
        if checkpoint is None:
            raise FileNotFoundError("No checkpoint found. Train the model first.")
        print(f"Using latest checkpoint: {checkpoint}")
    else:
        checkpoint = Path(checkpoint)

    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device)

    model_config = ckpt.get("model_config", {})
    model_size = model_config.get("model_size", "S")
    patch_size = model_config.get("patch_size", 2)
    image_size = model_config.get("image_size", 32)
    model_type = model_config.get("model_type", "dit")
    qk_rmsnorm = model_config.get("qk_rmsnorm", True)
    register_tokens = model_config.get("register_tokens", 0)

    print(f"Initializing DiT-{model_size} ({model_type})...")
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
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Initialize diffusion
    diffusion = Diffusion(
        num_timesteps=1000,
        beta_schedule="cosine",
        guidance_scale=guidance_scale,
        uncond_embed=uncond_embed,
    )

    print(f"\nGenerating {num_samples} image(s) for {len(prompts)} prompt(s)...")
    all_images = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i + 1}/{len(prompts)}] '{prompt}'")

        text_embeds = clip_encoder.encode([prompt] * num_samples)
        text_embeds = text_embeds.to(device)

        images = diffusion.sample(
            model=model,
            shape=(num_samples, 3, image_size, image_size),
            text_embeds=text_embeds,
            num_steps=num_steps,
            use_ddim=True,
            use_cfg=True,
        )

        for img in images:
            img = (img + 1) / 2  # Denormalize
            img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            pil_img = Image.fromarray(img)
            all_images.append(pil_img)

    print(f"Generated {len(all_images)} image(s)")
    return all_images


def demo(checkpoint: str | Path | None = None) -> None:
    """Interactive demo mode.

    Args:
        checkpoint: Path to model checkpoint (uses default if None)
    """
    print("=" * 60)
    print("text-to-emoji Interactive Demo")
    print("=" * 60)
    print("\nEnter prompts to generate images. Type 'quit' to exit.\n")

    # Find checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint("checkpoints")
        if checkpoint is None:
            checkpoint = Path("checkpoints/model_best.pt")

    if not Path(checkpoint).exists():
        print(f"Error: Checkpoint not found: {checkpoint}")
        print("Please train the model first: python main.py --train")
        return

    device = get_device("auto")
    print(f"Using device: {device}")

    # Load CLIP encoder
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Compute unconditional embedding
    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    model_config = ckpt.get("model_config", {})

    model = DiT(
        in_channels=3,
        image_size=32,
        patch_size=2,
        model_size=model_config.get("model_size", "S"),
        clip_embed_dim=clip_encoder.embedding_dim,
        model_type=model_config.get("model_type", "dit"),
        qk_rmsnorm=model_config.get("qk_rmsnorm", True),
        register_tokens=model_config.get("register_tokens", 0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    diffusion = Diffusion(
        num_timesteps=1000,
        beta_schedule="cosine",
        guidance_scale=7.5,
        uncond_embed=uncond_embed,
    )

    while True:
        try:
            prompt = input("\nEnter prompt (or 'quit'): ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt:
                continue

            print(f"Generating: '{prompt}'...")
            text_embeds = clip_encoder.encode([prompt])
            text_embeds = text_embeds.to(device)

            images = diffusion.sample(
                model=model,
                shape=(1, 3, 32, 32),
                text_embeds=text_embeds,
                num_steps=50,
                use_ddim=True,
                use_cfg=True,
            )

            img = images[0]
            img = (img + 1) / 2
            img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            pil_img = Image.fromarray(img)

            # Save
            output_path = f"demo_{prompt[:10].replace(' ', '_')}.png"
            pil_img.save(output_path)
            print(f"Saved: {output_path}")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")
