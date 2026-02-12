"""Image generation utilities for tiny-stable-diffusion.

Implements latent-space generation (Stable Diffusion 3 style):
1. Sample noise in latent space
2. Denoise using DiT model
3. Decode latent to image using VAE decoder
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.models.diffusion import Diffusion
from src.models.factory import DiT
from src.models.vae import create_vae
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.checkpoint import find_latest_checkpoint
from src.utils.common import get_device, set_seed


def infer_model_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model configuration from state dict keys and shapes.

    Args:
        state_dict: Model state dictionary

    Returns:
        Dictionary with inferred model configuration
    """
    config = {}

    # Detect model type: MMDiT vs DiT
    is_mmdit = any("mmdit" in k for k in state_dict.keys())
    config["model_type"] = "mmdit" if is_mmdit else "dit"

    # Count blocks
    block_indices = set()
    for k in state_dict.keys():
        if "blocks." in k:
            parts = k.split("blocks.")
            if len(parts) > 1:
                idx = parts[1].split(".")[0]
                if idx.isdigit():
                    block_indices.add(int(idx))
    num_blocks = len(block_indices)

    # Infer hidden dimension from patch_embed
    if "model.patch_embed.proj.weight" in state_dict:
        hidden_dim = state_dict["model.patch_embed.proj.weight"].shape[0]
        in_channels = state_dict["model.patch_embed.proj.weight"].shape[1]
        patch_size = state_dict["model.patch_embed.proj.weight"].shape[2]
        config["in_channels"] = in_channels
        config["patch_size"] = patch_size
    else:
        hidden_dim = None

    # Infer latent_size from pos_embed
    if "model.pos_embed.pos_embed" in state_dict:
        num_patches = state_dict["model.pos_embed.pos_embed"].shape[1]
        # latent_size = sqrt(num_patches) * patch_size
        import math

        patch_size = config.get("patch_size", 2)
        latent_size = int(math.sqrt(num_patches)) * patch_size
        config["latent_size"] = latent_size

    # Map (num_blocks, hidden_dim) to model_size
    # Standard sizes: S=12/384, B=12/768, L=24/1024, XL=28/1152
    if is_mmdit:
        # MMDiT uses different hidden dims due to joint attention
        # For MMDiT-B: hidden_dim is typically 768 for text and different for image
        if num_blocks == 12:
            if hidden_dim and hidden_dim <= 512:
                config["model_size"] = "S"
            else:
                config["model_size"] = "B"
        elif num_blocks == 24:
            config["model_size"] = "L"
        elif num_blocks == 28:
            config["model_size"] = "XL"
        else:
            config["model_size"] = "B"  # Default
    else:
        # DiT standard sizes
        if num_blocks == 12:
            if hidden_dim and hidden_dim <= 400:
                config["model_size"] = "S"
            else:
                config["model_size"] = "B"
        elif num_blocks == 24:
            config["model_size"] = "L"
        elif num_blocks == 28:
            config["model_size"] = "XL"
        else:
            config["model_size"] = "B"

    return config


@torch.no_grad()
def generate(
    prompts: list[str],
    checkpoint: str | Path | None = None,
    vae_checkpoint: str | Path | None = None,
    num_samples: int = 1,
    num_steps: int = 50,
    sampler: str = "euler",
    guidance_scale: float = 7.5,
    seed: int | None = None,
    device: str = "auto",
    scaling_factor: float | None = None,
) -> list[Image.Image]:
    """Generate images from text prompts using latent-space diffusion.

    Args:
        prompts: List of text prompts
        checkpoint: Path to diffusion model checkpoint
        vae_checkpoint: Path to VAE checkpoint
        num_samples: Number of samples per prompt
        num_steps: Number of diffusion steps
        sampler: Sampling method ("euler" or "ddim")
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        device: Device to use ("auto", "cuda", "mps", "cpu")
        scaling_factor: VAE scaling factor (overrides checkpoint value if provided)

    Returns:
        List of generated PIL images
    """
    if seed is not None:
        set_seed(seed)

    runtime_device = get_device(device)
    print(f"Using device: {runtime_device}")

    # Load CLIP encoder
    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(runtime_device)
    clip_encoder.eval()

    # Compute unconditional embedding
    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(runtime_device)

    # Find diffusion checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint("checkpoints", prefix="diffusion")
        if checkpoint is None:
            checkpoint = Path("checkpoints/diffusion.pt")
    checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint}")

    print(f"Loading diffusion checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=runtime_device)

    # Try to get config from checkpoint, or infer from state_dict
    model_config = ckpt.get("model_config", ckpt.get("config", {}))

    # If no config in checkpoint, infer from state_dict
    if not model_config and "model_state_dict" in ckpt:
        print("No config in checkpoint, inferring from state_dict...")
        inferred_config = infer_model_config_from_state_dict(ckpt["model_state_dict"])
        model_config = inferred_config
        print(f"  Inferred config: {model_config}")

    model_size = model_config.get("model_size", "S")
    patch_size = model_config.get("patch_size", 2)
    model_type = model_config.get("model_type", "dit")
    qk_rmsnorm = model_config.get("qk_rmsnorm", True)
    register_tokens = model_config.get("register_tokens", 0)
    latent_size = model_config.get("latent_size", 8)
    in_channels = model_config.get("in_channels", 16)
    image_size = model_config.get("image_size", 64)

    # Find VAE checkpoint
    if vae_checkpoint is None:
        vae_checkpoint = model_config.get("vae_checkpoint", "checkpoints/vae.pt")
    if vae_checkpoint is None:
        raise ValueError("VAE checkpoint path is missing in config and arguments")
    vae_checkpoint = Path(str(vae_checkpoint))

    if not vae_checkpoint.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")

    # Load VAE
    print(f"Loading VAE from {vae_checkpoint}...")
    vae = create_vae(
        image_size=image_size,
        z_channels=in_channels,
        ch=model_config.get("vae_ch", 64),
        ch_mult=tuple(model_config.get("vae_ch_mult", [1, 2, 4, 4])),
    )
    vae_state = torch.load(vae_checkpoint, map_location=runtime_device)
    vae.load_state_dict(vae_state["model_state_dict"])
    vae = vae.to(runtime_device)
    vae.eval()

    # Set scaling factor: argument > checkpoint > config > default
    if scaling_factor is not None:
        vae.set_scaling_factor(float(scaling_factor))
    elif "scaling_factor" in vae_state:
        vae.set_scaling_factor(vae_state["scaling_factor"])
    else:
        config_sf = model_config.get("scaling_factor", 1.0)
        if isinstance(config_sf, (int, float)) and config_sf != "auto":
            vae.set_scaling_factor(float(config_sf))
    print(f"VAE scaling_factor: {vae.scaling_factor:.4f}")

    # Load DiT model
    print(f"Initializing DiT-{model_size} ({model_type}) for latent space...")
    print(f"  Latent size: {latent_size}x{latent_size}")
    print(f"  Latent channels: {in_channels}")

    model = DiT(
        in_channels=in_channels,
        image_size=latent_size,
        patch_size=patch_size,
        model_size=model_size,
        clip_embed_dim=clip_encoder.embedding_dim,
        model_type=model_type,
        qk_rmsnorm=qk_rmsnorm,
        register_tokens=register_tokens,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(runtime_device)
    model.eval()

    # Initialize Rectified Flow diffusion
    diffusion = Diffusion(
        num_timesteps=1000,
        guidance_scale=guidance_scale,
        uncond_embed=uncond_embed,
    )

    print(f"\nGenerating {num_samples} image(s) for {len(prompts)} prompt(s)...")
    all_images = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i + 1}/{len(prompts)}] '{prompt}'")

        text_embeds = clip_encoder.encode([prompt] * num_samples)
        text_embeds = text_embeds.to(runtime_device)

        images = diffusion.sample(
            model=model,
            shape=(num_samples, in_channels, latent_size, latent_size),
            text_embeds=text_embeds,
            num_steps=num_steps,
            sampler=sampler,
            use_cfg=True,
            vae_decoder=vae,
        )

        for img in images:
            # images are already in [0, 1] after sample()
            img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            pil_img = Image.fromarray(img)
            all_images.append(pil_img)

    print(f"Generated {len(all_images)} image(s)")
    return all_images


def demo(
    checkpoint: str | Path | None = None,
    vae_checkpoint: str | Path | None = None,
) -> None:
    """Interactive demo mode.

    Args:
        checkpoint: Path to diffusion checkpoint
        vae_checkpoint: Path to VAE checkpoint
    """
    print("=" * 60)
    print("tiny-stable-diffusion Interactive Demo")
    print("=" * 60)
    print("\nEnter prompts to generate images. Type 'quit' to exit.\n")

    # Find checkpoints
    if checkpoint is None:
        checkpoint = find_latest_checkpoint("checkpoints", prefix="diffusion")
        if checkpoint is None:
            checkpoint = Path("checkpoints/diffusion.pt")

    if not Path(checkpoint).exists():
        print(f"Error: Diffusion checkpoint not found: {checkpoint}")
        print("Please train the model first:")
        print("  1. python main.py --train-vae")
        print("  2. python main.py --train-diffusion")
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

    # Load diffusion checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    model_config = ckpt.get("model_config", ckpt.get("config", {}))

    latent_size = model_config.get("latent_size", 8)
    in_channels = model_config.get("in_channels", 16)
    image_size = model_config.get("image_size", 64)

    # Find VAE checkpoint
    if vae_checkpoint is None:
        vae_checkpoint = model_config.get("vae_checkpoint", "checkpoints/vae.pt")
    if vae_checkpoint is None:
        print("Error: VAE checkpoint path is missing in config and arguments")
        return
    vae_checkpoint = Path(str(vae_checkpoint))

    if not vae_checkpoint.exists():
        print(f"Error: VAE checkpoint not found: {vae_checkpoint}")
        return

    # Load VAE
    vae = create_vae(
        image_size=image_size,
        z_channels=in_channels,
        ch=model_config.get("vae_ch", 64),
        ch_mult=tuple(model_config.get("vae_ch_mult", [1, 2, 4, 4])),
    )
    vae_state = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Set scaling factor from config or checkpoint
    scaling_factor = model_config.get("scaling_factor", 1.0)
    if isinstance(scaling_factor, (int, float)) and scaling_factor != "auto":
        vae.set_scaling_factor(float(scaling_factor))
    elif "scaling_factor" in vae_state:
        vae.set_scaling_factor(vae_state["scaling_factor"])

    # Load DiT model
    model = DiT(
        in_channels=in_channels,
        image_size=latent_size,
        patch_size=model_config.get("patch_size", 2),
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
        guidance_scale=7.5,
        uncond_embed=uncond_embed,
    )

    print(f"Model loaded. Generating {image_size}x{image_size} images.")

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
                shape=(1, in_channels, latent_size, latent_size),
                text_embeds=text_embeds,
                num_steps=50,
                use_cfg=True,
                vae_decoder=vae,
            )

            img = images[0]
            img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            pil_img = Image.fromarray(img)

            # Save
            output_path = f"demo_{prompt[:10].replace(' ', '_')}.png"
            pil_img.save(output_path)
            print(f"Saved: {output_path}")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")
