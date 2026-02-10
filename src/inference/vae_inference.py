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
from tqdm import tqdm

from src.config import get_config
from src.models.vae import create_vae
from src.training.checkpoint import find_latest_checkpoint
from src.utils.common import get_device


def _load_vae(checkpoint: str | Path | None, device: str | torch.device):
    """Load VAE model from checkpoint.

    Returns:
        Tuple of (vae, image_size, device)
    """
    device = get_device(device)

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

    return vae, image_size, device, checkpoint, epoch


def _reconstruct_single(
    vae,
    img: Image.Image,
    image_size: int,
    device,
) -> Image.Image:
    """Reconstruct a single image through VAE."""
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = transform(img)
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Expected tensor output from image transform")
    x = image_tensor.unsqueeze(0).to(device)

    # Encode and decode
    with torch.no_grad():
        mean, _ = vae.encode(x)
        recon = vae.decode(mean)

    # Post-process
    recon = (recon + 1) / 2  # [-1, 1] -> [0, 1]
    recon = recon.clamp(0, 1)
    recon = recon[0].permute(1, 2, 0).cpu().numpy()
    recon = (recon * 255).astype("uint8")

    return Image.fromarray(recon)


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

    # Load VAE
    vae, image_size, device, ckpt_path, epoch = _load_vae(checkpoint, device)
    print(f"Using device: {device}")
    print(f"Loaded VAE from {ckpt_path} (epoch {epoch})")

    # Load and reconstruct
    print(f"Loading image: {input_path}")
    img = Image.open(input_path).convert("RGB")

    print("Running VAE reconstruction...")
    output_img = _reconstruct_single(vae, img, image_size, device)

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


@torch.no_grad()
def reconstruct_vae_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path | None = None,
    pattern: str = "*.png",
    save_comparison: bool = True,
    device: str = "auto",
) -> list[Image.Image]:
    """Batch reconstruct images through VAE encoder/decoder.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save reconstructed images
        checkpoint: Path to VAE checkpoint (auto-detects if None)
        pattern: Glob pattern for input files (default: "*.png")
        save_comparison: Whether to save side-by-side comparisons
        device: Device to use ("auto", "cuda", "mps", "cpu")

    Returns:
        List of reconstructed PIL images
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all input images
    input_files = sorted(input_dir.glob(pattern))
    if not input_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {input_dir}")

    # Load VAE once
    vae, image_size, device, ckpt_path, epoch = _load_vae(checkpoint, device)
    print(f"Using device: {device}")
    print(f"Loaded VAE from {ckpt_path} (epoch {epoch})")
    print(f"Found {len(input_files)} images to process")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images with progress bar
    results = []
    for input_path in tqdm(input_files, desc="Reconstructing"):
        img = Image.open(input_path).convert("RGB")
        output_img = _reconstruct_single(vae, img, image_size, device)
        results.append(output_img)

        # Save output
        output_path = output_dir / input_path.name
        output_img.save(output_path)

        # Save comparison
        if save_comparison:
            orig_resized = img.resize((image_size, image_size))
            comparison = Image.new("RGB", (image_size * 2, image_size))
            comparison.paste(orig_resized, (0, 0))
            comparison.paste(output_img, (image_size, 0))
            comparison_path = output_dir / input_path.name.replace(".png", "_comparison.png")
            comparison.save(comparison_path)

    print(f"Saved {len(results)} reconstructions to {output_dir}")
    return results


def _compute_latent_statistics(
    vae,
    reference_dir: Path,
    image_size: int,
    device,
    max_images: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of latent space from reference images.

    Args:
        vae: VAE model
        reference_dir: Directory containing reference images
        image_size: Image size for preprocessing
        device: Device to use
        max_images: Maximum number of images to use for statistics

    Returns:
        Tuple of (mean, std) tensors with shape (latent_channels,)
    """
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Find reference images
    image_files = list(reference_dir.glob("*.png")) + list(reference_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {reference_dir}")

    image_files = image_files[:max_images]
    print(f"Computing latent statistics from {len(image_files)} reference images...")

    # Encode all images and collect latent vectors
    all_latents = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        image_tensor = transform(img)
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("Expected tensor output from image transform")
        x = image_tensor.unsqueeze(0).to(device)
        mean, _ = vae.encode(x)
        all_latents.append(mean)

    # Concatenate all latents: (N, C, H, W)
    latents = torch.cat(all_latents, dim=0)

    # Compute per-channel mean and std across all spatial locations and images
    # Shape: (N, C, H, W) -> compute stats over (N, H, W) for each channel
    latent_mean = latents.mean(dim=(0, 2, 3))  # (C,)
    latent_std = latents.std(dim=(0, 2, 3))  # (C,)

    return latent_mean, latent_std


@torch.no_grad()
def decode_random_latent(
    output_path: str | Path,
    checkpoint: str | Path | None = None,
    num_samples: int = 16,
    seed: int | None = None,
    latent_scale: float = 1.0,
    reference_dir: str | Path | None = None,
    device: str = "auto",
) -> list[Image.Image]:
    """Generate images from random latent vectors using VAE decoder only.

    Samples random latent vectors from the distribution learned by the VAE encoder.
    The mean and std are computed from reference images to ensure samples are
    within the valid latent space range.

    Args:
        output_path: Path to save generated images (grid or individual)
        checkpoint: Path to VAE checkpoint (auto-detects if None)
        num_samples: Number of images to generate (default: 16)
        seed: Random seed for reproducibility (optional)
        latent_scale: Scale factor for random latent std (default: 1.0)
        reference_dir: Directory with reference images for latent statistics
                      (default: results/original, fallback: samples/original)
        device: Device to use ("auto", "cuda", "mps", "cpu")

    Returns:
        List of generated PIL images
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")

    # Load VAE
    vae, image_size, device, ckpt_path, epoch = _load_vae(checkpoint, device)
    print(f"Using device: {device}")
    print(f"Loaded VAE from {ckpt_path} (epoch {epoch})")

    # Get latent dimensions from config
    config = get_config("vae_train")
    latent_channels = config.get("latent_channels", 16)
    latent_size = image_size // 8  # f8 compression

    # Compute latent statistics from reference images
    if reference_dir is None:
        default_reference_dir = Path("results/original")
        legacy_reference_dir = Path("samples/original")
        if default_reference_dir.exists():
            reference_dir = default_reference_dir
        elif legacy_reference_dir.exists():
            reference_dir = legacy_reference_dir
        else:
            reference_dir = default_reference_dir
    else:
        reference_dir = Path(reference_dir)

    latent_mean, latent_std = _compute_latent_statistics(vae, reference_dir, image_size, device)

    print(f"Latent statistics - Mean range: [{latent_mean.min():.3f}, {latent_mean.max():.3f}]")
    print(f"Latent statistics - Std range: [{latent_std.min():.3f}, {latent_std.max():.3f}]")

    print(f"Generating {num_samples} images from random latent vectors")
    print(f"Latent shape: ({latent_channels}, {latent_size}, {latent_size})")
    print(f"Latent scale: {latent_scale}")

    # Generate random latent vectors from learned distribution
    # z = mean + std * scale * noise
    z = torch.randn(num_samples, latent_channels, latent_size, latent_size, device=device)
    # Reshape mean and std for broadcasting: (C,) -> (1, C, 1, 1)
    latent_mean = latent_mean.view(1, -1, 1, 1)
    latent_std = latent_std.view(1, -1, 1, 1)
    z = latent_mean + latent_std * latent_scale * z

    # Decode latent to images
    print("Decoding latent vectors...")
    images = vae.decode_from_latent(z)

    # Post-process: [-1, 1] -> [0, 1] -> PIL
    images = (images + 1) / 2
    images = images.clamp(0, 1)

    results = []
    for i in range(num_samples):
        img_tensor = images[i].permute(1, 2, 0).cpu().numpy()
        img_array = (img_tensor * 255).astype("uint8")
        results.append(Image.fromarray(img_array))

    # Save individual images if output_path is a directory or ends with /
    if output_path.suffix == "" or str(output_path).endswith("/"):
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(results):
            img_path = output_dir / f"random_latent_{i:03d}.png"
            img.save(img_path)
        print(f"Saved {num_samples} individual images to {output_dir}")
    else:
        # Save as grid
        grid_size = int(num_samples**0.5)
        while grid_size * grid_size < num_samples:
            grid_size += 1

        grid_img = Image.new("RGB", (grid_size * image_size, grid_size * image_size))
        for i, img in enumerate(results):
            x = (i % grid_size) * image_size
            y = (i // grid_size) * image_size
            grid_img.paste(img, (x, y))

        grid_img.save(output_path)
        print(f"Saved grid image to: {output_path}")

    return results
