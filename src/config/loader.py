"""Configuration loading utilities for tiny-stable-diffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yaml import safe_load


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from config.yaml file.

    Args:
        config_path: Path to config file. If None, uses default config.yaml in project root.

    Returns:
        Configuration dictionary with training_stage, vae_train, diffusion_train, and common keys.
    """
    if config_path is None:
        # Find project root by looking for config.yaml
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = safe_load(f)

    return config


def get_config(stage: str, config_path: Path | str | None = None) -> dict[str, Any]:
    """Get merged configuration for a specific training stage.

    Args:
        stage: One of "vae_train", "diffusion_train"
        config_path: Optional path to config file

    Returns:
        Merged configuration dictionary with top-level, common + stage-specific settings.
    """
    config = load_config(config_path)

    # Get top-level settings (like model_type, training_stage)
    excluded_keys = ("common", "vae_train", "diffusion_train")
    top_level = {k: v for k, v in config.items() if k not in excluded_keys}

    common = config.get("common", {})
    stage_config = config.get(stage, {})

    return {**top_level, **common, **stage_config}


def get_training_stage(config_path: Path | str | None = None) -> str:
    """Get the current training stage from config.

    Args:
        config_path: Optional path to config file

    Returns:
        Training stage string ("vae_train" or "diffusion_train")
    """
    config = load_config(config_path)
    return config.get("training_stage", "vae_train")
