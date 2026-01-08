#!/usr/bin/env python3
"""Validate the emoji-64 dataset.

Checks:
- Image dimensions (should be 64x64)
- Color mode (should be RGB)
- File integrity (no corrupted images)
- Caption presence and format
- Statistics (total samples, unique captions)

Usage:
    python scripts/validate_dataset.py [--source huggingface] [--dataset-name junyeong-nero/emoji-64]
    python scripts/validate_dataset.py --local data/
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow library not found.")
    print("Install with: pip install pillow")
    sys.exit(1)


def validate_huggingface_dataset(
    dataset_name: str = "junyeong-nero/emoji-64",
    split: str = "train",
    max_samples: int | None = None,
) -> dict:
    """Validate dataset from Hugging Face.

    Args:
        dataset_name: Hugging Face dataset name
        split: Dataset split
        max_samples: Maximum samples to validate

    Returns:
        Dictionary with validation results
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Install with: pip install datasets")
        sys.exit(1)

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=True,
    )
    print(f"Total samples in dataset: {len(dataset)}")

    # Validation results
    results = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "size_distribution": Counter(),
        "mode_distribution": Counter(),
        "caption_distribution": Counter(),
    }

    valid_samples = []
    invalid_samples = []

    print("Validating samples...")
    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break

        results["total"] += 1

        try:
            image = sample["image"]
            caption = sample.get("text", sample.get("caption", ""))

            # Check image
            if not isinstance(image, Image.Image):
                raise ValueError(f"Sample {idx}: image is not a PIL Image")

            # Check dimensions (64x64 for emoji-64)
            width, height = image.size
            results["size_distribution"][f"{width}x{height}"] += 1

            if width != 64 or height != 64:
                raise ValueError(
                    f"Sample {idx}: Expected 64x64, got {width}x{height}"
                )

            # Check color mode
            mode = image.mode
            results["mode_distribution"][mode] += 1

            if mode not in ["RGB", "RGBA", "L", "P"]:
                raise ValueError(f"Sample {idx}: Unsupported mode: {mode}")

            # Check caption
            if not caption:
                raise ValueError(f"Sample {idx}: Missing caption")

            results["caption_distribution"][len(caption.split())] += 1

            valid_samples.append({"idx": idx, "caption": caption})

        except Exception as e:
            results["invalid"] += 1
            results["errors"].append(str(e))
            invalid_samples.append({"idx": idx, "error": str(e)})

        if (results["total"]) % 500 == 0:
            print(f"  Processed {results['total']} samples...")

    results["valid"] = results["total"] - results["invalid"]

    return results


def validate_local_directory(
    data_dir: str = "data",
    target_size: tuple[int, int] = (64, 64),
) -> dict:
    """Validate local dataset directory.

    Args:
        data_dir: Path to local dataset directory
        target_size: Expected image size (width, height)

    Returns:
        Dictionary with validation results
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Directory not found: {data_path}")
        sys.exit(1)

    print(f"Validating local dataset: {data_path}")

    # Find all image files
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    image_files = []
    for ext in extensions:
        image_files.extend(data_path.rglob(f"*{ext}"))

    if not image_files:
        print("Error: No image files found!")
        sys.exit(1)

    print(f"Found {len(image_files)} image files")

    # Validation results
    results = {
        "total": len(image_files),
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "size_distribution": Counter(),
        "mode_distribution": Counter(),
        "caption_distribution": Counter(),
        "corrupted": [],
        "size_distribution_bytes": Counter(),
    }

    valid_samples = []
    invalid_samples = []

    for idx, filepath in enumerate(sorted(image_files)):
        try:
            # Load image
            image = Image.open(filepath)
            image.load()

            # Get file size
            file_size = filepath.stat().st_size
            results["size_distribution_bytes"][file_size // 1024] += 1

            # Check dimensions
            width, height = image.size
            results["size_distribution"][f"{width}x{height}"] += 1

            if (width, height) != target_size:
                raise ValueError(
                    f"Expected {target_size[0]}x{target_size[1]}, got {width}x{height}"
                )

            # Check color mode
            mode = image.mode
            results["mode_distribution"][mode] += 1

            if mode not in ["RGB", "RGBA", "L", "P"]:
                raise ValueError(f"Unsupported mode: {mode}")

            # Check caption (from filename)
            caption = extract_caption(filepath.name)
            if caption:
                results["caption_distribution"][len(caption.split())] += 1

            valid_samples.append({
                "path": str(filepath),
                "caption": caption,
            })
            results["valid"] += 1

        except Exception as e:
            results["invalid"] += 1
            results["errors"].append(f"{filepath}: {e}")
            invalid_samples.append({
                "path": str(filepath),
                "error": str(e),
            })

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)} files...")

    return results


def extract_caption(filename: str) -> str:
    """Extract caption from filename."""
    name = Path(filename).stem
    import re

    # Skip if codepoint
    if re.match(r"^[0-9a-fA-F]+$", name):
        return name

    # Clean filename
    parts = name.split("_")
    caption = " ".join(parts)
    caption = caption.replace("-", " ")
    caption = re.sub(r"\s+", " ", caption).strip()

    return caption


def print_results(results: dict, source: str = "dataset") -> None:
    """Print validation results."""
    print("\n" + "=" * 60)
    print(f"Validation Results ({source})")
    print("=" * 60)

    print(f"\nTotal samples: {results['total']}")
    print(f"Valid: {results['valid']} ({results['valid']/max(results['total'],1)*100:.1f}%)")
    print(f"Invalid: {results['invalid']} ({results['invalid']/max(results['total'],1)*100:.1f}%)")

    print(f"\nImage Size Distribution:")
    for size, count in sorted(results["size_distribution"].items()):
        print(f"  {size}: {count} images")

    print(f"\nColor Mode Distribution:")
    for mode, count in sorted(results["mode_distribution"].items()):
        print(f"  {mode}: {count} images")

    if results["caption_distribution"]:
        print(f"\nCaption Word Count Distribution:")
        for words, count in sorted(results["caption_distribution"].items()):
            print(f"  {words} words: {count} samples")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])} total):")
        for error in results["errors"][:10]:
            print(f"  - {error}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")

    print("=" * 60)

    # Return success status
    if results["invalid"] == 0:
        print("✓ Dataset validation PASSED!")
        return True
    else:
        print(f"✗ Dataset validation FAILED ({results['invalid']} errors)")
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate emoji dataset")
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "local"],
        help="Dataset source",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="junyeong-nero/emoji-64",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--local",
        type=str,
        default="data",
        help="Local dataset directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to validate",
    )
    parser.add_argument(
        "--target-size",
        type=str,
        default="64x64",
        help="Expected image size (WxH)",
    )

    args = parser.parse_args()

    # Parse target size
    try:
        target_w, target_h = map(int, args.target_size.split("x"))
        target_size = (target_w, target_h)
    except ValueError:
        print(f"Error: Invalid target size format: {args.target_size}")
        print("Use format: WxH (e.g., 64x64)")
        sys.exit(1)

    if args.source == "huggingface":
        results = validate_huggingface_dataset(
            dataset_name=args.dataset_name,
            split=args.split,
            max_samples=args.max_samples,
        )
        source_name = f"Hugging Face: {args.dataset_name}"
    else:
        results = validate_local_directory(
            data_dir=args.local,
            target_size=target_size,
        )
        source_name = f"Local: {args.local}"

    success = print_results(results, source_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
