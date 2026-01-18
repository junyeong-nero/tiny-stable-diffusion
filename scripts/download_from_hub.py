#!/usr/bin/env python3
"""Download trained checkpoints from HuggingFace Hub.

Usage:
    # Download VAE checkpoint
    python scripts/download_from_hub.py --repo-id username/tiny-sd-vae --model-type vae

    # Download Diffusion checkpoint
    python scripts/download_from_hub.py --repo-id username/tiny-sd-diffusion --model-type diffusion

    # Download both from a combined repo
    python scripts/download_from_hub.py --repo-id username/tiny-sd-models --model-type all

    # Download to custom directory
    python scripts/download_from_hub.py --repo-id username/tiny-sd-vae --model-type vae --output-dir ./models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.hf_upload import check_hf_hub_available, download_from_hub


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download trained checkpoints from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download VAE
    python scripts/download_from_hub.py --repo-id myuser/tiny-sd-vae --model-type vae

    # Download Diffusion
    python scripts/download_from_hub.py --repo-id myuser/tiny-sd-diffusion --model-type diffusion

    # Download both from combined repo
    python scripts/download_from_hub.py --repo-id myuser/tiny-sd-models --model-type all

Environment:
    HF_TOKEN: HuggingFace API token for private repos (or use --token)
        """,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vae", "diffusion", "all"],
        required=True,
        help="Type of model to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save downloaded checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token for private repos (defaults to HF_TOKEN env var)",
    )

    args = parser.parse_args()

    if not check_hf_hub_available():
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == "all":
        # Download both VAE and diffusion
        print("Downloading VAE and Diffusion checkpoints...")
        for model_type in ["vae", "diffusion"]:
            _download_model(
                repo_id=args.repo_id,
                model_type=model_type,
                output_dir=output_dir,
                token=args.token,
            )
    else:
        _download_model(
            repo_id=args.repo_id,
            model_type=args.model_type,
            output_dir=output_dir,
            token=args.token,
        )


def _download_model(
    repo_id: str,
    model_type: str,
    output_dir: Path,
    token: str | None,
) -> None:
    """Download a single model from HuggingFace Hub."""
    print(f"\nDownloading {model_type} checkpoint from {repo_id}...")

    try:
        local_path = download_from_hub(
            repo_id=repo_id,
            model_type=model_type,
            local_dir=str(output_dir),
            token=token,
        )
        print(f"Success! Checkpoint saved to: {local_path}")
    except Exception as e:
        print(f"Error downloading {model_type}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
