#!/usr/bin/env python3
"""Upload trained checkpoints to HuggingFace Hub.

Usage:
    # Upload VAE checkpoint
    python scripts/upload_to_hub.py --model-type vae --repo-id username/tiny-sd-vae

    # Upload Diffusion checkpoint
    python scripts/upload_to_hub.py --model-type diffusion --repo-id username/tiny-sd-diffusion

    # Upload both to a single repo
    python scripts/upload_to_hub.py --model-type all --repo-id username/tiny-sd-models

    # Upload with custom checkpoint path
    python scripts/upload_to_hub.py --model-type vae --checkpoint checkpoints/vae.pt --repo-id username/my-vae

    # Create private repository
    python scripts/upload_to_hub.py --model-type all --repo-id username/tiny-sd-private --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.hf_upload import check_hf_hub_available, push_to_hub


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload trained checkpoints to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload VAE
    python scripts/upload_to_hub.py --model-type vae --repo-id myuser/tiny-sd-vae

    # Upload Diffusion
    python scripts/upload_to_hub.py --model-type diffusion --repo-id myuser/tiny-sd-diffusion

    # Upload both models to single repo
    python scripts/upload_to_hub.py --model-type all --repo-id myuser/tiny-sd-models

Environment:
    HF_TOKEN: HuggingFace API token (or use --token)
        """,
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vae", "diffusion", "all"],
        required=True,
        help="Type of model to upload",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: checkpoints/{model_type}.pt)",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="checkpoints/vae.pt",
        help="Path to VAE checkpoint (for --model-type all)",
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        type=str,
        default="checkpoints/diffusion.pt",
        help="Path to diffusion checkpoint (for --model-type all)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (defaults to HF_TOKEN env var)",
    )

    args = parser.parse_args()

    if not check_hf_hub_available():
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    if args.model_type == "all":
        # Upload both VAE and diffusion to the same repo
        _upload_model(
            checkpoint_path=args.vae_checkpoint,
            repo_id=args.repo_id,
            model_type="vae",
            private=args.private,
            token=args.token,
        )
        _upload_model(
            checkpoint_path=args.diffusion_checkpoint,
            repo_id=args.repo_id,
            model_type="diffusion",
            private=args.private,
            token=args.token,
        )
    else:
        checkpoint_path = args.checkpoint or f"checkpoints/{args.model_type}.pt"
        _upload_model(
            checkpoint_path=checkpoint_path,
            repo_id=args.repo_id,
            model_type=args.model_type,
            private=args.private,
            token=args.token,
        )


def _upload_model(
    checkpoint_path: str,
    repo_id: str,
    model_type: str,
    private: bool,
    token: str | None,
) -> None:
    """Upload a single model to HuggingFace Hub."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\nUploading {model_type} checkpoint: {checkpoint_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")

    try:
        url = push_to_hub(
            checkpoint_path=str(checkpoint_path),
            repo_id=repo_id,
            model_type=model_type,
            token=token,
            private=private,
            commit_message=f"Upload {model_type} checkpoint",
        )
        print(f"\nSuccess! Model available at: {url}")
    except Exception as e:
        print(f"Error uploading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
