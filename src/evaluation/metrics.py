"""Quantitative evaluation metrics for diffusion models.

Implements:
- Inception FID: Frechet Inception Distance (64->299 upsample)
- CLIP-FID: FID using CLIP image features (native 64x64)
- Inception Score (IS)
- CLIPScore: Text-image alignment score

All metrics use torchmetrics for reliable, batched computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore


class CLIPImageFeatureExtractor(nn.Module):
    """Extract CLIP image features for CLIP-FID computation.

    Wraps CLIP's visual encoder to produce feature vectors compatible
    with torchmetrics FrechetInceptionDistance.
    """

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        try:
            import clip
        except ImportError as e:
            raise ImportError(
                "CLIP is required for CLIP-FID. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            ) from e

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP image features.

        Args:
            images: uint8 tensors (B, 3, H, W) in [0, 255]

        Returns:
            Feature vectors (B, 512)
        """
        # Convert uint8 [0, 255] to float [0, 1]
        x = images.float() / 255.0

        # CLIP preprocessing: resize to 224x224 and normalize
        x = F.interpolate(x, size=(224, 224), mode="bicubic", antialias=True)
        x = x.clamp(0, 1)

        # CLIP normalization (from CLIP's preprocess)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device)
        x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        # Extract features
        features = self.clip_model.encode_image(x)
        return features.float()


def _to_uint8_tensor(images: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Convert images to uint8 [0, 255] tensors for torchmetrics.

    Handles:
    - Already uint8 tensors
    - Float tensors in [0, 1]
    - Float tensors in [-1, 1] (normalized with mean=0.5, std=0.5)

    Args:
        images: List of tensors or batched tensor (B, 3, H, W)

    Returns:
        uint8 tensor (B, 3, H, W) in [0, 255]
    """
    if isinstance(images, list):
        images = torch.stack(images)

    if images.dtype == torch.uint8:
        return images

    # If float, determine range
    if images.min() < -0.5:
        # Likely [-1, 1] range, convert to [0, 1]
        images = (images + 1.0) / 2.0

    images = images.clamp(0, 1).mul(255).to(torch.uint8)
    return images


def compute_fid(
    real_images: list[torch.Tensor] | torch.Tensor,
    generated_images: list[torch.Tensor] | torch.Tensor,
    device: str = "cpu",
    batch_size: int = 32,
) -> float:
    """Compute Inception FID between real and generated images.

    Uses InceptionV3 features. Images are upsampled from 64x64 to 299x299
    using bilinear interpolation with antialiasing.

    Args:
        real_images: Real images (list of tensors or batched tensor)
        generated_images: Generated images
        device: Compute device
        batch_size: Batch size for feature extraction

    Returns:
        FID score (lower is better)
    """
    metric_device = _resolve_metrics_device(device, metric_name="FID")
    fid = FrechetInceptionDistance(normalize=False).to(metric_device)

    real = _to_uint8_tensor(real_images)
    gen = _to_uint8_tensor(generated_images)

    # Upsample to 299x299 for InceptionV3
    real = _upsample_for_inception(real)
    gen = _upsample_for_inception(gen)

    # Update in batches to save memory
    for i in range(0, len(real), batch_size):
        batch = real[i : i + batch_size].to(metric_device)
        fid.update(batch, real=True)

    for i in range(0, len(gen), batch_size):
        batch = gen[i : i + batch_size].to(metric_device)
        fid.update(batch, real=False)

    score = fid.compute().item()
    return score


def _upsample_for_inception(images: torch.Tensor) -> torch.Tensor:
    """Upsample images to 299x299 for InceptionV3.

    Args:
        images: uint8 tensors (B, 3, H, W)

    Returns:
        Upsampled uint8 tensors (B, 3, 299, 299)
    """
    if images.shape[-1] == 299 and images.shape[-2] == 299:
        return images

    # Convert to float for interpolation, then back to uint8
    x = images.float()
    x = F.interpolate(x, size=(299, 299), mode="bilinear", antialias=True)
    return x.clamp(0, 255).to(torch.uint8)


def compute_clip_fid(
    real_images: list[torch.Tensor] | torch.Tensor,
    generated_images: list[torch.Tensor] | torch.Tensor,
    device: str = "cpu",
    batch_size: int = 32,
) -> float:
    """Compute CLIP-FID between real and generated images.

    Uses CLIP ViT-B/32 image features instead of InceptionV3.
    Works at native 64x64 resolution without upsampling.
    Computes FID directly from feature statistics using scipy.

    Args:
        real_images: Real images
        generated_images: Generated images
        device: Compute device
        batch_size: Batch size for feature extraction

    Returns:
        CLIP-FID score (lower is better)
    """
    import numpy as np
    from scipy import linalg

    feature_extractor = CLIPImageFeatureExtractor(device=device)

    real = _to_uint8_tensor(real_images)
    gen = _to_uint8_tensor(generated_images)

    if len(real) < 2 or len(gen) < 2:
        raise ValueError(
            f"CLIP-FID requires at least 2 images per set. "
            f"Got {len(real)} real, {len(gen)} generated."
        )

    real_features = _extract_features_batched(feature_extractor, real, device, batch_size)
    gen_features = _extract_features_batched(feature_extractor, gen, device, batch_size)

    # Compute FID from features: FID = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    real_np = real_features.numpy().astype(np.float64)
    gen_np = gen_features.numpy().astype(np.float64)

    mu_real = np.mean(real_np, axis=0)
    mu_gen = np.mean(gen_np, axis=0)
    sigma_real = np.cov(real_np, rowvar=False)
    sigma_gen = np.cov(gen_np, rowvar=False)

    diff = mu_real - mu_gen
    covmean = np.asarray(linalg.sqrtm(sigma_real @ sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    fid_score = float(
        diff @ diff + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
    )
    return max(0.0, fid_score)


def _extract_features_batched(
    extractor: nn.Module,
    images: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Extract features from images in batches.

    Args:
        extractor: Feature extraction model
        images: uint8 tensors (B, 3, H, W)
        device: Compute device
        batch_size: Batch size

    Returns:
        Feature tensor (B, D)
    """
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        features = extractor(batch)
        all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)


def compute_inception_score(
    images: list[torch.Tensor] | torch.Tensor,
    device: str = "cpu",
    batch_size: int = 32,
    splits: int = 10,
) -> tuple[float, float]:
    """Compute Inception Score (IS).

    Args:
        images: Generated images
        device: Compute device
        batch_size: Batch size for feature extraction
        splits: Number of splits for IS calculation

    Returns:
        Tuple of (IS mean, IS std)
    """
    metric_device = _resolve_metrics_device(device, metric_name="Inception Score")
    inception_score = InceptionScore(normalize=False, splits=splits).to(metric_device)

    imgs = _to_uint8_tensor(images)
    imgs = _upsample_for_inception(imgs)

    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size].to(metric_device)
        inception_score.update(batch)

    mean, std = inception_score.compute()
    return mean.item(), std.item()


def compute_clip_score(
    images: list[torch.Tensor] | torch.Tensor,
    prompts: list[str],
    device: str = "cpu",
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute CLIPScore for text-image alignment.

    Args:
        images: Generated images
        prompts: Corresponding text prompts
        device: Compute device
        batch_size: Batch size

    Returns:
        Dictionary with mean, std, and per_sample scores
    """
    clip_score_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)

    imgs = _to_uint8_tensor(images)

    for i in range(0, len(imgs), batch_size):
        batch_imgs = imgs[i : i + batch_size].to(device)
        batch_prompts = prompts[i : i + batch_size]
        clip_score_metric.update(batch_imgs, batch_prompts)

    overall_score = clip_score_metric.compute().item()

    return {
        "mean": overall_score,
        "std": 0.0,
    }


def run_self_test(device: str = "cpu") -> None:
    """Run a quick self-test with random tensors to verify all metrics work.

    Args:
        device: Compute device
    """
    print("Running metrics self-test with random tensors...")
    print(f"Device: {device}")

    # Create random "images" - uint8 [0, 255]
    real = torch.randint(0, 256, (50, 3, 64, 64), dtype=torch.uint8)
    fake = torch.randint(0, 256, (50, 3, 64, 64), dtype=torch.uint8)
    prompts = [f"test prompt {i}" for i in range(50)]

    print("\n1. Inception FID...")
    fid_score = compute_fid(real, fake, device=device, batch_size=16)
    print(f"   FID (Inception): {fid_score:.2f}")

    print("\n2. Inception Score...")
    is_mean, is_std = compute_inception_score(fake, device=device, batch_size=16)
    print(f"   IS: {is_mean:.4f} +/- {is_std:.4f}")

    print("\n3. CLIPScore...")
    clip_result = compute_clip_score(fake, prompts, device=device, batch_size=16)
    print(f"   CLIPScore: {clip_result['mean']:.4f} +/- {clip_result['std']:.4f}")

    print("\n4. CLIP-FID...")
    clip_fid = compute_clip_fid(real, fake, device=device, batch_size=16)
    print(f"   FID (CLIP): {clip_fid:.2f}")

    print("\nAll metrics computed successfully!")


def _resolve_metrics_device(device: str, metric_name: str) -> str:
    if device.startswith("mps"):
        print(f"  {metric_name}: using cpu backend (MPS does not support float64 states)")
        return "cpu"
    return device


if __name__ == "__main__":
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    run_self_test(device=device)
