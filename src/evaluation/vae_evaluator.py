"""VAE Reconstruction Quality Evaluator.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio): Higher is better, typically 20-40 dB
- SSIM (Structural Similarity Index): 0-1, higher is better
- MSE (Mean Squared Error): Lower is better
- LPIPS (Learned Perceptual Image Patch Similarity): Lower is better (optional)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from src.config import get_config
from src.models.vae import create_vae
from src.training.checkpoint import find_latest_checkpoint
from src.utils.common import get_device


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    psnr: float = 0.0
    ssim: float = 0.0
    mse: float = 0.0
    lpips: float | None = None
    num_samples: int = 0
    per_sample_metrics: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "mse": self.mse,
            "lpips": self.lpips,
            "num_samples": self.num_samples,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "VAE Reconstruction Quality Metrics",
            "=" * 50,
            f"  PSNR:  {self.psnr:.2f} dB  (higher is better, typical: 20-40 dB)",
            f"  SSIM:  {self.ssim:.4f}    (0-1, higher is better)",
            f"  MSE:   {self.mse:.6f}  (lower is better)",
        ]
        if self.lpips is not None:
            lines.append(f"  LPIPS: {self.lpips:.4f}    (lower is better)")
        lines.extend(["=" * 50, f"  Evaluated on {self.num_samples} samples"])
        return "\n".join(lines)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        img1: Original image tensor [B, C, H, W] in [0, 1]
        img2: Reconstructed image tensor [B, C, H, W] in [0, 1]
        max_val: Maximum pixel value

    Returns:
        PSNR in dB (higher is better)
    """
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val**2 / mse)


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Args:
        img1: Original image tensor [B, C, H, W] in [0, 1]
        img2: Reconstructed image tensor [B, C, H, W] in [0, 1]
        window_size: Size of the Gaussian window
        C1, C2: Constants for stability

    Returns:
        SSIM value (0-1, higher is better)
    """
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor(
        [
            np.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
            for x in range(window_size)
        ]
    )
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(img1.size(1), 1, window_size, window_size)
    window = window.to(img1.device, dtype=img1.dtype)

    # Compute means
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=pad, groups=img2.size(1))

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1**2, window, padding=pad, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=pad, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=img1.size(1)) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean().item()


def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute Mean Squared Error.

    Args:
        img1: Original image tensor [B, C, H, W]
        img2: Reconstructed image tensor [B, C, H, W]

    Returns:
        MSE value (lower is better)
    """
    return F.mse_loss(img1, img2).item()


def _try_load_lpips():
    """Try to load LPIPS model if available."""
    try:
        import lpips

        return lpips.LPIPS(net="alex", verbose=False)
    except ImportError:
        return None


def _load_vae(checkpoint: str | Path | None, device: str):
    """Load VAE model from checkpoint."""
    device = get_device(device)

    if checkpoint is None:
        checkpoint = find_latest_checkpoint("checkpoints", prefix="vae")
        if checkpoint is None:
            checkpoint = Path("checkpoints/vae.pt")
    checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint}")

    config = get_config("vae_train")
    image_size = config.get("image_size", 64)

    vae = create_vae(
        image_size=image_size,
        z_channels=config.get("latent_channels", 16),
        ch=config.get("vae_ch", 64),
        ch_mult=tuple(config.get("vae_ch_mult", [1, 2, 4, 4])),
    )

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    epoch = ckpt.get("epoch", "unknown")

    return vae, image_size, device, checkpoint, epoch


def evaluate_vae(
    input_dir: str | Path,
    checkpoint: str | Path | None = None,
    device: str = "auto",
    use_lpips: bool = True,
    save_results: str | Path | None = None,
    max_samples: int | None = None,
) -> EvaluationMetrics:
    """Evaluate VAE reconstruction quality on a directory of images.

    Args:
        input_dir: Directory containing input images
        checkpoint: Path to VAE checkpoint (auto-detect if None)
        device: Device to use
        use_lpips: Whether to compute LPIPS metric (requires lpips package)
        save_results: Path to save JSON results
        max_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        EvaluationMetrics with averaged metrics
    """
    vae, image_size, device, ckpt_path, epoch = _load_vae(checkpoint, device)

    # Load LPIPS model if requested
    lpips_model = None
    if use_lpips:
        lpips_model = _try_load_lpips()
        if lpips_model is not None:
            lpips_model = lpips_model.to(device)
        else:
            print("Note: LPIPS not available. Install with: pip install lpips")

    # Preprocessing transform
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )
    normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # Find images
    input_dir = Path(input_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    image_files = sorted(image_files)

    if max_samples is not None:
        image_files = image_files[:max_samples]

    if not image_files:
        raise ValueError(f"No images found in {input_dir}")

    print(f"Evaluating VAE: {ckpt_path} (epoch {epoch})")
    print(f"Found {len(image_files)} images in {input_dir}")

    # Collect metrics
    all_psnr = []
    all_ssim = []
    all_mse = []
    all_lpips = []
    per_sample = []

    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Evaluating"):
            try:
                # Load and preprocess
                img = Image.open(img_path).convert("RGB")
                x_01 = transform(img).unsqueeze(0).to(device)  # [0, 1] range
                x_norm = normalize(x_01)  # [-1, 1] range for VAE

                # Encode and decode
                mean, _ = vae.encode(x_norm)
                recon_norm = vae.decode(mean)

                # Convert back to [0, 1]
                recon_01 = (recon_norm + 1) / 2
                recon_01 = recon_01.clamp(0, 1)

                # Compute metrics
                psnr = compute_psnr(x_01, recon_01)
                ssim = compute_ssim(x_01, recon_01)
                mse = compute_mse(x_01, recon_01)

                all_psnr.append(psnr)
                all_ssim.append(ssim)
                all_mse.append(mse)

                sample_metrics = {
                    "file": img_path.name,
                    "psnr": psnr,
                    "ssim": ssim,
                    "mse": mse,
                }

                # LPIPS (needs [-1, 1] range)
                if lpips_model is not None:
                    lpips_val = lpips_model(x_norm, recon_norm).item()
                    all_lpips.append(lpips_val)
                    sample_metrics["lpips"] = lpips_val

                per_sample.append(sample_metrics)

            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                continue

    # Compute averages
    metrics = EvaluationMetrics(
        psnr=np.mean(all_psnr),
        ssim=np.mean(all_ssim),
        mse=np.mean(all_mse),
        lpips=np.mean(all_lpips) if all_lpips else None,
        num_samples=len(all_psnr),
        per_sample_metrics=per_sample,
    )

    # Save results
    if save_results:
        save_path = Path(save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(
                {
                    "checkpoint": str(ckpt_path),
                    "epoch": epoch,
                    "input_dir": str(input_dir),
                    "metrics": metrics.to_dict(),
                    "per_sample": per_sample,
                },
                f,
                indent=2,
            )
        print(f"Results saved to {save_path}")

    return metrics


def evaluate_vae_on_dataset(
    dataset_name: str = "reach-vb/pokemon-blip-captions",
    checkpoint: str | Path | None = None,
    device: str = "auto",
    use_lpips: bool = True,
    save_results: str | Path | None = None,
    max_samples: int = 100,
    split: str = "train",
    image_field: str = "image",
) -> EvaluationMetrics:
    """Evaluate VAE on a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name
        checkpoint: Path to VAE checkpoint
        device: Device to use
        use_lpips: Whether to compute LPIPS
        save_results: Path to save JSON results
        max_samples: Number of samples to evaluate
        split: Dataset split to use
        image_field: Field name containing images

    Returns:
        EvaluationMetrics
    """
    from datasets import load_dataset

    vae, image_size, device, ckpt_path, epoch = _load_vae(checkpoint, device)

    # Load LPIPS
    lpips_model = None
    if use_lpips:
        lpips_model = _try_load_lpips()
        if lpips_model is not None:
            lpips_model = lpips_model.to(device)

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )
    normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    print(f"Evaluating VAE: {ckpt_path} (epoch {epoch})")
    print(f"Dataset: {dataset_name}, max_samples: {max_samples}")

    all_psnr = []
    all_ssim = []
    all_mse = []
    all_lpips = []
    per_sample = []

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, total=max_samples, desc="Evaluating")):
            if i >= max_samples:
                break

            try:
                img = sample[image_field]
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert("RGB")
                else:
                    img = img.convert("RGB")

                x_01 = transform(img).unsqueeze(0).to(device)
                x_norm = normalize(x_01)

                mean, _ = vae.encode(x_norm)
                recon_norm = vae.decode(mean)
                recon_01 = ((recon_norm + 1) / 2).clamp(0, 1)

                psnr = compute_psnr(x_01, recon_01)
                ssim = compute_ssim(x_01, recon_01)
                mse = compute_mse(x_01, recon_01)

                all_psnr.append(psnr)
                all_ssim.append(ssim)
                all_mse.append(mse)

                sample_metrics = {"index": i, "psnr": psnr, "ssim": ssim, "mse": mse}

                if lpips_model is not None:
                    lpips_val = lpips_model(x_norm, recon_norm).item()
                    all_lpips.append(lpips_val)
                    sample_metrics["lpips"] = lpips_val

                per_sample.append(sample_metrics)

            except Exception as e:
                print(f"Warning: Failed to process sample {i}: {e}")
                continue

    metrics = EvaluationMetrics(
        psnr=np.mean(all_psnr),
        ssim=np.mean(all_ssim),
        mse=np.mean(all_mse),
        lpips=np.mean(all_lpips) if all_lpips else None,
        num_samples=len(all_psnr),
        per_sample_metrics=per_sample,
    )

    if save_results:
        save_path = Path(save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(
                {
                    "checkpoint": str(ckpt_path),
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "metrics": metrics.to_dict(),
                    "per_sample": per_sample,
                },
                f,
                indent=2,
            )
        print(f"Results saved to {save_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate VAE reconstruction quality")
    parser.add_argument("--input-dir", type=str, help="Directory with images to evaluate")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--checkpoint", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--no-lpips", action="store_true", help="Disable LPIPS metric")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to evaluate")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--image-field", type=str, default="image", help="Image field name")

    args = parser.parse_args()

    if args.input_dir:
        metrics = evaluate_vae(
            input_dir=args.input_dir,
            checkpoint=args.checkpoint,
            device=args.device,
            use_lpips=not args.no_lpips,
            save_results=args.save,
            max_samples=args.max_samples,
        )
    elif args.dataset:
        metrics = evaluate_vae_on_dataset(
            dataset_name=args.dataset,
            checkpoint=args.checkpoint,
            device=args.device,
            use_lpips=not args.no_lpips,
            save_results=args.save,
            max_samples=args.max_samples,
            split=args.split,
            image_field=args.image_field,
        )
    else:
        parser.print_help()
        exit(1)

    print(metrics)
