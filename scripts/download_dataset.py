#!/usr/bin/env python3
"""Download and prepare the emoji-64 dataset from Hugging Face.

This script downloads the junyeong-nero/emoji-64 dataset and optionally
caches it locally for faster training.

Usage:
    python scripts/download_dataset.py [--output-dir data/] [--split train]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found.")
    print("Install with: pip install datasets")
    sys.exit(1)


def download_dataset(
    dataset_name: str = "junyeong-nero/emoji-64",
    output_dir: str = "data",
    split: str = "train",
    max_samples: int | None = None,
) -> None:
    """Download dataset from Hugging Face and save locally.

    Args:
        dataset_name: Hugging Face dataset name
        output_dir: Local directory to save images
        split: Dataset split to download
        max_samples: Maximum number of samples to download (None for all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_path}")
    print(f"Split: {split}")

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(
            dataset_name,
            split=split,
            trust_remote_code=True,
        )

        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(dataset)}")

        # Count saved files
        saved_count = 0
        skipped_count = 0

        # Save images locally
        from PIL import Image

        for idx, sample in enumerate(dataset):
            if max_samples is not None and idx >= max_samples:
                break

            image = sample["image"]
            caption = sample.get("text", sample.get("caption", f"emoji_{idx}"))

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Create safe filename from caption
            safe_caption = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in caption.lower()
            )
            if not safe_caption:
                safe_caption = f"emoji_{idx}"

            filename = f"{idx:05d}_{safe_caption}.png"
            filepath = output_path / filename

            # Skip if already exists
            if filepath.exists():
                skipped_count += 1
                continue

            # Save image
            image.save(filepath)
            saved_count += 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)} images...")

        print(f"\nDownload complete!")
        print(f"  Saved: {saved_count} images")
        print(f"  Skipped (already exists): {skipped_count} images")
        print(f"  Total: {saved_count + skipped_count} images")

        # Save metadata
        metadata_path = output_path / "metadata.csv"
        with open(metadata_path, "w") as f:
            f.write("filename,caption\n")
            for idx, sample in enumerate(dataset):
                if max_samples is not None and idx >= max_samples:
                    break
                caption = sample.get("text", sample.get("caption", f"emoji_{idx}"))
                safe_caption = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in caption.lower()
                )
                if not safe_caption:
                    safe_caption = f"emoji_{idx}"
                filename = f"{idx:05d}_{safe_caption}.png"
                f.write(f'"{filename}","{caption}"\n')

        print(f"  Metadata saved: {metadata_path}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


def check_cache_size(cache_dir: str = "~/.cache/pixmoji") -> None:
    """Check the size of the cached dataset."""
    cache_path = Path(cache_dir).expanduser()

    if not cache_path.exists():
        print(f"Cache directory does not exist: {cache_path}")
        return

    total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
    print(f"Cache directory: {cache_path}")
    print(f"Cache size: {total_size / (1024 * 1024):.2f} MB")


def clear_cache(cache_dir: str = "~/.cache/pixmoji") -> None:
    """Clear the Hugging Face dataset cache."""
    cache_path = Path(cache_dir).expanduser()

    if not cache_path.exists():
        print(f"Cache directory does not exist: {cache_path}")
        return

    import shutil

    shutil.rmtree(cache_path)
    print(f"Cache cleared: {cache_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and prepare emoji-64 dataset from Hugging Face"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="junyeong-nero/emoji-64",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for downloaded images",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to download",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download (default: all)",
    )
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="Check cache size",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear Hugging Face dataset cache",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.cache/pixmoji",
        help="Cache directory for Hugging Face datasets",
    )

    args = parser.parse_args()

    if args.check_cache:
        check_cache_size(args.cache_dir)
        return

    if args.clear_cache:
        clear_cache(args.cache_dir)
        return

    download_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
