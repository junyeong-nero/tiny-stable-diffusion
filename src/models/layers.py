"""Common layer components for DiT models.

This module contains the building blocks shared by VanillaDiT and MMDiT.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings.

    Converts a (B, C, H, W) image into (B, N, D) patch tokens
    where N = (H/patch_size) * (W/patch_size).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 384,
        patch_size: int = 2,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Patch embeddings (B, N, D) where N = num_patches
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class PositionEmbed(nn.Module):
    """Learned positional embeddings for patch tokens."""

    def __init__(self, num_patches: int, hidden_size: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input."""
        return x + self.pos_embed


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive modulation (shift and scale) to normalized input."""
    return x * (1 + scale) + shift


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero for diffusion timestep conditioning.

    From DiT paper: "We extend AdaIN by zero-initializing the
    modulation parameters in each block... AdaLN-Zero is more effective."

    Each block receives 6 parameters:
    - shift_msa, scale_msa, gate_msa (for self-attention)
    - shift_mlp, scale_mlp, gate_mlp (for MLP)
    """

    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_layers * hidden_size * 6),
        )

    def forward(
        self, temb: torch.Tensor
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Project timestep embedding to per-block parameters.

        Args:
            temb: Time embedding (B, D)

        Returns:
            Lists of (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        temb = self.ada_lin(temb)  # (B, num_layers * D * 6)
        temb = temb.view(temb.size(0), self.num_layers, -1)  # (B, num_layers, D*6)

        shifts_msa, scales_msa, gates_msa = [], [], []
        shifts_mlp, scales_mlp, gates_mlp = [], [], []

        for i in range(self.num_layers):
            params = temb[:, i, :]  # (B, D*6)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
            shifts_msa.append(shift_msa[:, None, :])  # (B, 1, D)
            scales_msa.append(scale_msa[:, None, :])
            gates_msa.append(gate_msa[:, None, :])
            shifts_mlp.append(shift_mlp[:, None, :])
            scales_mlp.append(scale_mlp[:, None, :])
            gates_mlp.append(gate_mlp[:, None, :])

        return shifts_msa, scales_msa, gates_msa, shifts_mlp, scales_mlp, gates_mlp


class DiTBlock(nn.Module):
    """DiT Transformer Block with Cross-Attention.

    Architecture:
        x -> LayerNorm -> Self-Attention -> Add
        x -> LayerNorm -> Cross-Attention (with text) -> Add
        x -> LayerNorm -> MLP -> Add

    AdaLN-Zero modulation is applied to self-attention and MLP branches.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Self-Attention on image tokens
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attn_dropout, batch_first=True
        )

        # Cross-Attention for text conditioning
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attn_dropout, batch_first=True
        )

        # MLP
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(mlp_dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        text_embeds: torch.Tensor | None = None,
        shift_msa: torch.Tensor | None = None,
        scale_msa: torch.Tensor | None = None,
        gate_msa: torch.Tensor | None = None,
        shift_mlp: torch.Tensor | None = None,
        scale_mlp: torch.Tensor | None = None,
        gate_mlp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with AdaLN-Zero conditioning."""
        # Self-Attention with AdaLN-Zero modulation
        x_norm = self.norm1(x)
        if shift_msa is not None and scale_msa is not None:
            x_norm = modulate(x_norm, shift_msa, scale_msa)

        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        if gate_msa is not None:
            x_attn = gate_msa * x_attn
        x = x + x_attn

        # Cross-Attention with text conditioning
        if text_embeds is not None:
            x_norm = self.norm2(x)
            x_cross, _ = self.cross_attn(x_norm, text_embeds, text_embeds)
            x = x + x_cross

        # MLP with AdaLN-Zero modulation
        x_norm = self.norm3(x)
        if shift_mlp is not None and scale_mlp is not None:
            x_norm = modulate(x_norm, shift_mlp, scale_mlp)

        x_mlp = self.mlp(x_norm)
        if gate_mlp is not None:
            x_mlp = gate_mlp * x_mlp
        x = x + x_mlp

        return x


class FinalLayer(nn.Module):
    """Final layer to decode patch embeddings back to image.

    Includes AdaLN modulation for timestep conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode patch tokens to image.

        Args:
            x: Patch embeddings (B, N, D)
            c: Conditioning embedding (B, D) - timestep embedding

        Returns:
            Image tensor (B, C, H, W)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift.unsqueeze(1), scale.unsqueeze(1))
        x = self.linear(x)  # (B, N, patch_size^2 * C)

        B, N, _ = x.shape
        h = w = int(N**0.5)

        # (B, N, p*p*C) -> (B, h, w, p, p, C) -> (B, C, h*p, w*p)
        x = x.view(B, h, w, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.contiguous().view(B, self.out_channels, h * self.patch_size, w * self.patch_size)

        return x


# Model size configurations
MODEL_CONFIGS = {
    "S": {"layers": 12, "heads": 6, "hidden": 384},
    "B": {"layers": 12, "heads": 12, "hidden": 768},
    "L": {"layers": 24, "heads": 16, "hidden": 1024},
    "XL": {"layers": 28, "heads": 16, "hidden": 1152},
}
