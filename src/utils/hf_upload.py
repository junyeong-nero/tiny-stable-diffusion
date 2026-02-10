"""Hugging Face Hub upload utilities for trained models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

try:
    from huggingface_hub import HfApi, create_repo

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def check_hf_hub_available() -> bool:
    """Check if huggingface_hub is available."""
    return HF_HUB_AVAILABLE


def push_to_hub(
    checkpoint_path: str | Path,
    repo_id: str,
    model_type: str = "vae",
    config: dict[str, Any] | None = None,
    token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """Push a trained model checkpoint to Hugging Face Hub.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        model_type: Type of model - "vae" or "diffusion"
        config: Training configuration to save alongside the model
        token: HuggingFace API token (uses HF_TOKEN env var if not provided)
        private: Whether to create a private repository
        commit_message: Custom commit message

    Returns:
        URL of the uploaded model

    Raises:
        ImportError: If huggingface_hub is not installed
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Get token from environment if not provided
    token = token or os.environ.get("HF_TOKEN")

    api = HfApi(token=token)

    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Warning: Could not create repository: {e}")

    # Prepare commit message
    if commit_message is None:
        commit_message = f"Upload {model_type} checkpoint"

    # Create a temporary directory for organized uploads
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Copy checkpoint with a standardized name
        checkpoint_name = f"{model_type}.pt"
        target_checkpoint = tmp_path / checkpoint_name

        # Load and re-save checkpoint (ensures compatibility)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        torch.save(checkpoint, target_checkpoint)

        # Save model config as JSON
        if config is not None:
            config_path = tmp_path / "config.json"
            # Filter out non-serializable items
            serializable_config = _make_serializable(config)
            with open(config_path, "w") as f:
                json.dump(serializable_config, f, indent=2)

        # Create model card
        model_card = _create_model_card(repo_id, model_type, config)
        readme_path = tmp_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(model_card)

        # Upload the folder
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            commit_message=commit_message,
        )

    print(f"Successfully uploaded to: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


def _make_serializable(obj: Any) -> Any:
    """Convert an object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, "item"):  # numpy/torch scalar
        return obj.item()
    elif hasattr(obj, "tolist"):  # numpy/torch array
        return obj.tolist()
    else:
        return str(obj)


def _create_model_card(
    repo_id: str,
    model_type: str,
    config: dict[str, Any] | None = None,
) -> str:
    """Create a model card for the uploaded model."""
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

    tags = f"""  - tiny-stable-diffusion
  - {model_type}
  - image-generation
  - diffusion"""

    card = f"""---
license: mit
tags:
{tags}
library_name: pytorch
---

# {model_name}

This is a **{model_type.upper()}** model trained with [tiny-stable-diffusion](https://github.com/your-username/tiny-stable-diffusion).

## Model Description

"""

    if model_type == "vae":
        card += """This is a Variational Autoencoder (VAE) trained to compress images into a latent space.
The VAE follows the SD3 architecture with 16 latent channels and f8 compression ratio.

### Architecture
- **Type**: AutoencoderKL
- **Latent Channels**: 16
- **Compression**: f8 (64x64 → 8x8)
"""
    else:
        card += """This is a Diffusion Transformer (DiT/MMDiT) trained for text-to-image generation in latent space.

### Architecture
- **Type**: DiT or MMDiT (Multi-Modal Diffusion Transformer)
- **Conditioning**: CLIP text embeddings
"""

    if config:
        card += "\n## Training Configuration\n\n```json\n"
        card += json.dumps(_make_serializable(config), indent=2)
        card += "\n```\n"

    card += """
## Usage

```python
import torch
from src.models.vae import create_vae  # or appropriate model import

# Load checkpoint
checkpoint = torch.load("model.pt", map_location="cpu")

# Create model and load weights
model = create_model(...)  # Use config from checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
```
"""

    card += """
## License

MIT License
"""
    return card


def download_from_hub(
    repo_id: str,
    model_type: str = "vae",
    local_dir: str | Path = "checkpoints",
    token: str | None = None,
) -> Path:
    """Download a model from Hugging Face Hub.

    Args:
        repo_id: HuggingFace repository ID
        model_type: Type of model - "vae" or "diffusion"
        local_dir: Local directory to save the model
        token: HuggingFace API token

    Returns:
        Path to the downloaded checkpoint file
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        )

    from huggingface_hub import hf_hub_download

    token = token or os.environ.get("HF_TOKEN")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = f"{model_type}.pt"

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_name,
        local_dir=str(local_dir),
        token=token,
    )

    print(f"Downloaded checkpoint to: {local_path}")
    return Path(local_path)
