"""DiT model factory - unified interface for VanillaDiT and MMDiT."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.models.mmdit import MMDIT_AVAILABLE, MMDiT
from src.models.vanilla_dit import VanillaDiT


class DiT(nn.Module):
    """Unified DiT model supporting both Vanilla DiT and MMDiT.

    Factory class that returns the appropriate model based on model_type.

    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        image_size: Input image size (default: 32)
        patch_size: Patch size for patch embedding (default: 2)
        model_size: Model size - S, B, L, or XL
        clip_embed_dim: CLIP text embedding dimension
        attn_dropout: Attention dropout probability
        mlp_dropout: MLP dropout probability
        mlp_ratio: MLP hidden dimension ratio
        model_type: "dit" for Vanilla DiT, "mmdit" for Multi-Modal DiT
        qk_rmsnorm: Use RMSNorm for QK attention (MMDiT only)
        register_tokens: Number of register tokens (MMDiT only)
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 2,
        model_size: Literal["S", "B", "L", "XL"] = "S",
        clip_embed_dim: int = 512,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        model_type: Literal["dit", "mmdit"] = "dit",
        qk_rmsnorm: bool = True,
        register_tokens: int = 0,
    ) -> None:
        super().__init__()

        self.model_type = model_type

        if model_type == "mmdit":
            self.model = MMDiT(
                in_channels=in_channels,
                image_size=image_size,
                patch_size=patch_size,
                model_size=model_size,
                clip_embed_dim=clip_embed_dim,
                qk_rmsnorm=qk_rmsnorm,
                register_tokens=register_tokens,
            )
        else:
            self.model = VanillaDiT(
                in_channels=in_channels,
                image_size=image_size,
                patch_size=patch_size,
                model_size=model_size,
                clip_embed_dim=clip_embed_dim,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                mlp_ratio=mlp_ratio,
            )

        # Copy attributes for compatibility
        self.in_channels = self.model.in_channels
        self.image_size = self.model.image_size
        self.patch_size = self.model.patch_size
        self.model_size = self.model.model_size
        self.hidden_size = self.model.hidden_size
        self.num_layers = self.model.num_layers
        self.num_heads = self.model.num_heads
        self.clip_embed_dim = self.model.clip_embed_dim

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy image (B, C, H, W)
            timestep: Diffusion timestep (B,) or timestep embedding
            text_embeds: Text embeddings from CLIP (B, D_clip)

        Returns:
            Predicted noise (B, C, H, W)
        """
        return self.model(x, timestep, text_embeds)

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype from first parameter."""
        return self.model.dtype

    def parameters_count(self) -> int:
        """Count total parameters."""
        return self.model.parameters_count()

    def get_model_size_info(self) -> dict:
        """Get model size information."""
        ESTIMATED_SIZES = {
            "S": "~30M (DiT) / ~132M (MMDiT)",
            "B": "~130M (DiT) / ~300M (MMDiT)",
            "L": "~300M (DiT) / ~675M (MMDiT)",
            "XL": "~675M (DiT) / ~780M (MMDiT)",
        }
        return {
            "model_size": self.model_size,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_parameters": self.parameters_count(),
            "estimated_size": ESTIMATED_SIZES.get(self.model_size, "Unknown"),
        }


def create_dit_model(
    model_type: Literal["dit", "mmdit"] = "dit",
    **kwargs,
) -> VanillaDiT | MMDiT:
    """Factory function to create DiT or MMDiT model.

    Args:
        model_type: "dit" for Vanilla DiT, "mmdit" for Multi-Modal DiT
        **kwargs: Additional arguments passed to model constructor

    Returns:
        VanillaDiT or MMDiT model instance
    """
    if model_type == "mmdit":
        if not MMDIT_AVAILABLE:
            raise ImportError("mmdit is not installed. Install with: pip install mmdit")
        return MMDiT(**kwargs)
    return VanillaDiT(**kwargs)
