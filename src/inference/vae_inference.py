"""VAE inference utilities for tiny-stable-diffusion.

Provides VAE reconstruction functionality for testing VAE quality:
1. Load input image
2. Encode to latent space
3. Decode back to image
4. Save reconstruction and comparison
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from src.config import get_config
from src.models.vae import create_vae
from src.training.checkpoint import find_latest_checkpoint
from src.utils.common import get_device


@torch.no_grad()
def reconstruct_vae(
    input_path: str | Path,
    output_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    save_comparison: bool = True,
    device: str = "auto",
) -> Image.Image:
    """Reconstruct an image through VAE encoder/decoder.

    Args:
        input_path: Path to input image
        output_path: Path to save reconstructed image (optional)
        checkpoint: Path to VAE checkpoint (auto-detects if None)
        save_comparison: Whether to save side-by-side comparison
        device: Device to use ("auto", "cuda", "mps", "cpu")

    Returns:
        Reconstructed PIL image
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    device = get_device(device)
    print(f"Using device: {device}")

    # Find VAE checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint("checkpoints", prefix="vae")
        if checkpoint is None:
            checkpoint = Path("checkpoints/vae.pt")
    checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint}")

    # Load VAE config
    config = get_config("vae_train")
    image_size = config.get("image_size", 64)

    # Create and load VAE
    print(f"Loading VAE from {checkpoint}...")
    vae = create_vae(
        image_size=image_size,
        z_channels=config.get("latent_channels", 16),
        ch=config.get("vae_ch", 64),
        ch_mult=tuple(config.get("vae_ch_mult", [1, 2, 4, 4])),
    )

    ckpt = torch.load(checkpoint, map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    epoch = ckpt.get("epoch", "unknown")
    print(f"  Epoch: {epoch}")

    # Load and preprocess input image
    print(f"Loading image: {input_path}")
    img = Image.open(input_path).convert("RGB")
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = transform(img).unsqueeze(0).to(device)

    # Encode and decode
    print("Running VAE reconstruction...")
    mean, logvar = vae.encode(x)
    z = mean  # Use mean for deterministic reconstruction
    print(f"  Latent shape: {z.shape}")

    recon = vae.decode(z)

    # Post-process
    recon = (recon + 1) / 2  # [-1, 1] -> [0, 1]
    recon = recon.clamp(0, 1)
    recon = recon[0].permute(1, 2, 0).cpu().numpy()
    recon = (recon * 255).astype("uint8")
    output_img = Image.fromarray(recon)

    # Save output if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(output_path)
        print(f"Saved reconstruction to: {output_path}")

        # Save side-by-side comparison
        if save_comparison:
            orig_resized = img.resize((image_size, image_size))
            comparison = Image.new("RGB", (image_size * 2, image_size))
            comparison.paste(orig_resized, (0, 0))
            comparison.paste(output_img, (image_size, 0))
            comparison_path = str(output_path).replace(".png", "_comparison.png")
            comparison.save(comparison_path)
            print(f"Saved comparison to: {comparison_path}")

    return output_img
