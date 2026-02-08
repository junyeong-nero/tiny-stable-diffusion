"""Diffusion model evaluation orchestrator.

Combines metrics (FID, IS, CLIPScore) computation with image generation
to provide a complete evaluation pipeline accessible via CLI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_clip_fid,
    compute_clip_score,
    compute_fid,
    compute_inception_score,
)
from src.inference.generator import generate
from src.utils.common import get_device, set_seed


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics."""

    fid: float | None = None
    clip_fid: float | None = None
    inception_score_mean: float | None = None
    inception_score_std: float | None = None
    clip_score_mean: float | None = None
    clip_score_std: float | None = None
    num_generated: int = 0
    num_real: int = 0
    checkpoint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _load_images_from_directory(
    image_dir: str | Path,
    target_size: int = 64,
    max_images: int | None = None,
) -> list[torch.Tensor]:
    """Load images from a directory and convert to uint8 tensors.

    Args:
        image_dir: Path to image directory
        target_size: Resize images to this size
        max_images: Maximum number of images to load

    Returns:
        List of uint8 tensors (3, H, W) in [0, 255]
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    transform = transforms.Compose(
        [
            transforms.Resize(
                (target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
        ]
    )

    images = []
    for path in tqdm(image_paths, desc="Loading real images"):
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img)  # [0, 1] float
            tensor = (tensor * 255).to(torch.uint8)
            images.append(tensor)
        except Exception:
            continue

    return images


def _load_pairs_from_dataset(
    dataset_name: str,
    num_samples: int = 1000,
    target_size: int = 64,
    image_field: str = "jpg",
    caption_field: str = "txt",
    use_webdataset: bool = False,
) -> tuple[list[torch.Tensor], list[str]]:
    """Load (image, caption) pairs from a dataset.

    Args:
        dataset_name: HuggingFace dataset name
        num_samples: Number of pairs to load
        target_size: Resize images to this size
        image_field: Name of the image field
        caption_field: Name of the caption field
        use_webdataset: Use WebDatasetCaptionDataset for tar-based datasets

    Returns:
        Tuple of (real_images as uint8 tensors, captions as strings)
    """
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
        ]
    )

    images: list[torch.Tensor] = []
    captions: list[str] = []

    print(f"Loading {num_samples} (image, caption) pairs from {dataset_name}...")

    if use_webdataset:
        from src.data.dataset import WebDatasetCaptionDataset

        dataset = WebDatasetCaptionDataset(
            dataset_name=dataset_name,
            transform=eval_transform,
            image_field=image_field,
            caption_field=caption_field,
            target_size=target_size,
        )

        for sample in tqdm(dataset, desc="Loading dataset pairs", total=num_samples):
            tensor = sample["image"]
            # transform already returns [0,1] float; convert to uint8
            tensor = (tensor * 255).to(torch.uint8)
            images.append(tensor)
            captions.append(sample["caption"])
            if len(images) >= num_samples:
                break
    else:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split="train", streaming=True)

        for sample in tqdm(dataset, desc="Loading dataset pairs", total=num_samples):
            image_data = sample.get(image_field)
            if image_data is None:
                continue

            try:
                if isinstance(image_data, Image.Image):
                    img = image_data
                elif isinstance(image_data, bytes):
                    import io

                    img = Image.open(io.BytesIO(image_data))
                else:
                    continue

                img = img.convert("RGB")
                tensor = eval_transform(img)
                tensor = (tensor * 255).to(torch.uint8)
                images.append(tensor)

                caption = sample.get(caption_field, "an image")
                if isinstance(caption, bytes):
                    caption = caption.decode("utf-8")
                elif isinstance(caption, list):
                    import random

                    picked = random.choice(caption) if caption else "an image"
                    caption = picked.decode("utf-8") if isinstance(picked, bytes) else picked
                captions.append(str(caption))

                if len(images) >= num_samples:
                    break
            except Exception:
                continue

    print(f"  Loaded {len(images)} pairs")
    return images, captions


def _generate_images(
    checkpoint: str | Path | None,
    vae_checkpoint: str | Path | None,
    prompts: list[str],
    num_samples_per_prompt: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[str]]:
    """Generate images and return as uint8 tensors with their prompts.

    Returns:
        Tuple of (image tensors, corresponding prompts)
    """
    set_seed(seed)

    all_tensors = []
    all_prompts = []

    pil_images = generate(
        prompts=prompts,
        checkpoint=checkpoint,
        vae_checkpoint=vae_checkpoint,
        num_samples=num_samples_per_prompt,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    to_tensor = transforms.ToTensor()
    for i, img in enumerate(pil_images):
        tensor = to_tensor(img)  # [0, 1]
        tensor = (tensor * 255).to(torch.uint8)
        all_tensors.append(tensor)

        prompt_idx = i // num_samples_per_prompt
        all_prompts.append(prompts[min(prompt_idx, len(prompts) - 1)])

    return all_tensors, all_prompts


def evaluate_diffusion(
    checkpoint: str | Path | None = None,
    vae_checkpoint: str | Path | None = None,
    prompts: list[str] | None = None,
    real_images_dir: str | Path | None = None,
    dataset_name: str | None = None,
    num_eval_samples: int = 1000,
    num_samples_per_prompt: int = 1,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    metrics: list[str] | None = None,
    seed: int = 42,
    device: str = "auto",
    eval_batch_size: int = 32,
    save_results: str | Path | None = None,
    image_field: str = "jpg",
    caption_field: str = "txt",
    use_webdataset: bool = False,
) -> EvaluationResult:
    """Run full evaluation pipeline.

    When dataset_name is specified, loads (image, caption) pairs from the dataset
    and uses captions as prompts for generation, enabling proper FID comparison
    between real and generated images.

    When only real_images_dir is specified, loads images from directory and uses
    prompts from --prompt/--eval-prompts-file/defaults.

    When neither is specified, only computes IS and CLIPScore using default prompts.

    Args:
        checkpoint: Path to diffusion checkpoint
        vae_checkpoint: Path to VAE checkpoint
        prompts: Text prompts for generation (overridden by dataset captions when dataset_name is set)
        real_images_dir: Directory with real images (for FID)
        dataset_name: HuggingFace dataset name for (image, caption) pairs
        num_eval_samples: Number of images to generate for evaluation
        num_samples_per_prompt: Samples per prompt
        num_steps: Diffusion sampling steps
        guidance_scale: CFG guidance scale
        metrics: List of metrics to compute (fid, clip_fid, is, clip_score)
        seed: Random seed
        device: Compute device
        eval_batch_size: Batch size for metric computation
        save_results: Path to save JSON results
        image_field: Name of the image field in dataset
        caption_field: Name of the caption field in dataset
        use_webdataset: Use WebDataset format for loading

    Returns:
        EvaluationResult with computed metrics
    """
    if metrics is None:
        metrics = ["fid", "clip_fid", "is", "clip_score"]

    device_str = str(get_device(device))
    result = EvaluationResult(checkpoint=str(checkpoint or ""))
    needs_real = "fid" in metrics or "clip_fid" in metrics

    real_images: list[torch.Tensor] = []
    gen_images: list[torch.Tensor] = []
    gen_prompts: list[str] = []

    if dataset_name is not None and needs_real:
        # Dataset path: load (image, caption) pairs, use captions as prompts
        real_images, dataset_captions = _load_pairs_from_dataset(
            dataset_name=dataset_name,
            num_samples=num_eval_samples,
            image_field=image_field,
            caption_field=caption_field,
            use_webdataset=use_webdataset,
        )

        print(f"\nGenerating {len(dataset_captions)} images from dataset captions...")

        gen_images, gen_prompts = _generate_images(
            checkpoint=checkpoint,
            vae_checkpoint=vae_checkpoint,
            prompts=dataset_captions,
            num_samples_per_prompt=1,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    else:
        # Non-dataset path: use provided/default prompts
        if prompts is None:
            prompts = [
                "a cat sitting on a couch",
                "a dog running in a park",
                "a person riding a bicycle",
                "a bus on a city street",
                "a pizza on a wooden table",
            ]

        total_prompts = len(prompts)
        num_samples_per_prompt = max(1, num_eval_samples // total_prompts)

        print(
            f"\nGenerating {total_prompts * num_samples_per_prompt} images ({total_prompts} prompts x {num_samples_per_prompt} samples)..."
        )

        gen_images, gen_prompts = _generate_images(
            checkpoint=checkpoint,
            vae_checkpoint=vae_checkpoint,
            prompts=prompts,
            num_samples_per_prompt=num_samples_per_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        # Load real images from directory if needed
        if needs_real:
            if real_images_dir is not None:
                real_images = _load_images_from_directory(
                    real_images_dir,
                    max_images=num_eval_samples,
                )
            else:
                print("Warning: No real images source specified. Skipping FID metrics.")
                metrics = [m for m in metrics if m not in ("fid", "clip_fid")]

    result.num_generated = len(gen_images)
    result.num_real = len(real_images)

    # Compute metrics
    if "fid" in metrics and real_images:
        print("\nComputing Inception FID...")
        result.fid = compute_fid(
            real_images, gen_images, device=device_str, batch_size=eval_batch_size
        )
        print(f"  FID (Inception): {result.fid:.2f}")

    if "clip_fid" in metrics and real_images:
        print("\nComputing CLIP-FID...")
        result.clip_fid = compute_clip_fid(
            real_images, gen_images, device=device_str, batch_size=eval_batch_size
        )
        print(f"  FID (CLIP): {result.clip_fid:.2f}")

    if "is" in metrics:
        print("\nComputing Inception Score...")
        is_mean, is_std = compute_inception_score(
            gen_images, device=device_str, batch_size=eval_batch_size
        )
        result.inception_score_mean = is_mean
        result.inception_score_std = is_std
        print(f"  IS: {is_mean:.4f} +/- {is_std:.4f}")

    if "clip_score" in metrics:
        print("\nComputing CLIPScore...")
        clip_result = compute_clip_score(
            gen_images, gen_prompts, device=device_str, batch_size=eval_batch_size
        )
        result.clip_score_mean = clip_result["mean"]
        result.clip_score_std = clip_result["std"]
        print(f"  CLIPScore: {result.clip_score_mean:.4f} +/- {result.clip_score_std:.4f}")

    # Save results
    if save_results is not None:
        save_path = Path(save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {save_path}")

    return result


def format_evaluation_results(result: EvaluationResult) -> str:
    """Format evaluation results into a readable summary.

    Args:
        result: EvaluationResult to format

    Returns:
        Formatted string
    """
    lines = [
        "",
        "\u2550" * 58,
        "  Diffusion Model Evaluation Results",
        "\u2550" * 58,
    ]

    if result.fid is not None:
        lines.append(f"  FID (Inception):  {result.fid:>8.1f}    (lower is better)")

    if result.clip_fid is not None:
        lines.append(f"  FID (CLIP):       {result.clip_fid:>8.1f}    (lower is better)")

    if result.inception_score_mean is not None:
        lines.append(
            f"  IS:            {result.inception_score_mean:>5.2f} \u00b1 {result.inception_score_std:.2f}"
            f"  (higher is better)"
        )

    if result.clip_score_mean is not None:
        lines.append(
            f"  CLIPScore:     {result.clip_score_mean:>5.3f} \u00b1 {result.clip_score_std:.2f}"
            f"  (higher is better)"
        )

    lines.extend(
        [
            "\u2550" * 58,
            f"  Generated: {result.num_generated} samples | Real: {result.num_real} samples",
            f"  Checkpoint: {result.checkpoint}",
        ]
    )

    return "\n".join(lines)
