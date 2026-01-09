"""Exponential Moving Average (EMA) for model weights.

EMA maintains a smoothed version of model weights during training,
which typically produces better and more stable results for inference.
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average for model weights.

    Maintains a shadow copy of model parameters that are updated
    with exponential moving average during training.

    Args:
        model: The model to track.
        decay: EMA decay rate. Higher values = slower updates.
            Typical values: 0.999, 0.9999.
        update_after_step: Start EMA updates after this many steps.
        update_every: Update EMA every N steps.

    Example:
        >>> model = MyModel()
        >>> ema = EMA(model, decay=0.9999)
        >>> for batch in dataloader:
        ...     loss = train_step(model, batch)
        ...     ema.update()
        >>> # Use EMA weights for inference
        >>> ema.apply_shadow()
        >>> output = model(input)
        >>> ema.restore()  # Restore training weights
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
    ) -> None:
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every

        self.step = 0
        self.shadow_params: dict[str, torch.Tensor] = {}
        self.backup_params: dict[str, torch.Tensor] = {}

        # Initialize shadow parameters
        self._init_shadow_params()

    def _init_shadow_params(self) -> None:
        """Initialize shadow parameters as copies of model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    def update(self) -> None:
        """Update EMA parameters.

        Should be called after each optimizer step.
        """
        self.step += 1

        if self.step < self.update_after_step:
            return

        if self.step % self.update_every != 0:
            return

        # Compute effective decay (warmup for early steps)
        decay = min(self.decay, (1 + self.step) / (10 + self.step))

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    # EMA update: shadow = decay * shadow + (1 - decay) * param
                    self.shadow_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def apply_shadow(self) -> None:
        """Apply EMA weights to model (for inference).

        Backs up current weights before applying.
        Call restore() to revert to training weights.
        """
        self.backup_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])

    def restore(self) -> None:
        """Restore original model weights after apply_shadow()."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params = {}

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            "step": self.step,
            "decay": self.decay,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.step = state_dict["step"]
        self.decay = state_dict["decay"]
        self.shadow_params = state_dict["shadow_params"]

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA weights to another model instance."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name])

    def to(self, device: torch.device) -> "EMA":
        """Move EMA parameters to device."""
        for name in self.shadow_params:
            self.shadow_params[name] = self.shadow_params[name].to(device)
        return self

    def __repr__(self) -> str:
        return f"EMA(decay={self.decay}, step={self.step})"
