"""Centralized wandb logging utilities for tiny-stable-diffusion."""

from __future__ import annotations

from typing import Any

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class WandbLogger:
    """Centralized wandb logger for training.

    Handles wandb initialization, logging, and cleanup with resume support.

    Example:
        logger = WandbLogger(
            enabled=True,
            project="tiny-stable-diffusion",
            run_name="vae-training",
            config=config,
            run_id=checkpoint.get("wandb_run_id"),  # For resume
        )

        # In training loop
        logger.log({"train/loss": loss}, step=global_step)

        # At end of training
        logger.finish()
    """

    def __init__(
        self,
        enabled: bool = False,
        project: str = "tiny-stable-diffusion",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize wandb logger.

        Args:
            enabled: Whether to enable wandb logging
            project: Wandb project name
            run_name: Wandb run name
            config: Training configuration to log
            run_id: Existing run ID for resume (optional)
        """
        self._enabled = enabled and WANDB_AVAILABLE
        self._run_id: str | None = None

        if enabled and not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Disabling wandb logging.")

        if self._enabled:
            wandb.init(
                project=project,
                name=run_name,
                config=config,
                id=run_id,
                resume="allow" if run_id else None,
            )
            self._run_id = wandb.run.id
            resume_info = f" (resumed: {run_id})" if run_id else ""
            print(f"Wandb logging enabled (run_id: {self._run_id}){resume_info}")

    @property
    def enabled(self) -> bool:
        """Whether wandb logging is enabled."""
        return self._enabled

    @property
    def run_id(self) -> str | None:
        """Current wandb run ID for checkpoint saving."""
        return self._run_id

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names to values
            step: Global step number (optional)
        """
        if self._enabled:
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Finish wandb run."""
        if self._enabled:
            wandb.finish()
