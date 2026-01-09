"""Checkpoint save and load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.training.ema import EMA


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    loss: float,
    path: str | Path,
    config: dict[str, Any],
    ema: EMA | None = None,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state (optional)
        epoch: Current epoch
        loss: Current loss value
        path: Path to save checkpoint
        config: Training configuration
        ema: EMA state (optional)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "model_config": {
            "model_size": config.get("model_size", "S"),
            "patch_size": config.get("patch_size", 2),
            "image_size": config.get("image_size", 32),
            "model_type": config.get("model_type", "dit"),
            "qk_rmsnorm": config.get("qk_rmsnorm", True),
            "register_tokens": config.get("register_tokens", 0),
        },
    }

    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: EMA | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        ema: EMA to load state into (optional)
        device: Device to load tensors to

    Returns:
        Checkpoint dictionary with epoch, loss, and model_config
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "model_config": checkpoint.get("model_config", {}),
    }


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to most recent checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: p.stat().st_mtime)
