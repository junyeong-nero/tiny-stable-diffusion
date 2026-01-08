"""Diffusion Transformer (DiT) architecture.

Implements the DiT model from "Scalable Diffusion Models with Transformers"
(Google Research, 2023): https://arxiv.org/abs/2212.09748
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Calculate number of patches
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        # Conv2D projection: (C, H, W) -> (D, H/p, W/p)
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

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


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero for diffusion timestep conditioning.

    From DiT paper: "We extend AdaIN by zero-initializing the
    modulation parameters in each block... AdaLN-Zero is more effective."
    """

    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Project timestep to per-block scale and shift parameters
        # Each layer gets its own adaptive parameters
        self.ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_layers * hidden_size * 3),  # scale, shift, gate
        )

    def forward(self, temb: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Project timestep embedding to per-block parameters.

        Args:
            temb: Time embedding (B, D)

        Returns:
            Lists of scale, shift, and gate parameters for each block
        """
        temb = self.ada_lin(temb)  # (B, num_layers * D * 3)
        temb = temb.view(temb.size(0), self.num_layers, -1)  # (B, num_layers, D*3)

        # Split into scale, shift, gate for each layer
        scales = []
        shifts = []
        gates = []

        for i in range(self.num_layers):
            params = temb[:, i, :]  # (B, D*3)
            scale, shift, gate = params.chunk(3, dim=-1)
            scales.append(scale[:, None, :])  # (B, 1, D)
            shifts.append(shift[:, None, :])
            gates.append(gate[:, None, :])

        return scales, shifts, gates


class DiTBlock(nn.Module):
    """DiT Transformer Block with Cross-Attention.

    Architecture:
        x -> LayerNorm -> Attention -> Add
        where Attention can be:
        - Self-Attention (on image tokens)
        - Cross-Attention (between image and text tokens)
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
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attn_dropout, batch_first=True
        )

        # Cross-Attention for text conditioning
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attn_dropout, batch_first=True
        )

        # MLP
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
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
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional AdaLN-Zero conditioning.

        Args:
            x: Input tokens (B, N, D)
            text_embeds: Text conditioning (B, L, D), optional
            scale: AdaLN scale parameter (B, 1, D)
            shift: AdaLN shift parameter (B, 1, D)
            gate: AdaLN gate parameter (B, 1, D)

        Returns:
            Processed tokens (B, N, D)
        """
        # Self-Attention with optional AdaLN-Zero
        if scale is not None and shift is not None:
            x_norm = self.norm1(x) * (1 + scale) + shift
        else:
            x_norm = self.norm1(x)

        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + x_attn

        # Cross-Attention with text conditioning
        if text_embeds is not None:
            x_norm = self.norm2(x)
            x_cross, _ = self.cross_attn(x_norm, text_embeds, text_embeds)
            x = x + x_cross

        # MLP with optional AdaLN-Zero
        x_norm = self.norm3(x)
        x_mlp = self.mlp(x_norm)
        if gate is not None:
            x_mlp = x_mlp * gate
        x = x + x_mlp

        return x


class FinalLayer(nn.Module):
    """Final layer to decode patch embeddings back to image."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode patch tokens to image.

        Args:
            x: Patch embeddings (B, N, D)

        Returns:
            Image tensor (B, C, H, W)
        """
        x = self.norm(x)
        x = self.linear(x)  # (B, N, patch_size^2 * C)

        # Reshape to image
        B, N, _ = x.shape
        x = x.view(B, self.patch_size, self.patch_size, self.out_channels, N)
        x = x.permute(0, 3, 4, 1, 2)  # (B, C, N, p, p)
        x = x.flatten(2, 3)  # (B, C, N*p, p)
        H = W = int(N**0.5) * self.patch_size
        x = x.view(B, self.out_channels, H, W)

        return x


class DiT(nn.Module):
    """Diffusion Transformer (DiT) Model.

    Architecture:
        Input Image -> Patch Embed -> Position Embed -> DiT Blocks -> Final Layer -> Output Image

    Conditioning:
        - Timestep: Via AdaLN-Zero in each block
        - Text: Via Cross-Attention in each block
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
    ) -> None:
        super().__init__()

        # Load model size configuration
        MODEL_CONFIGS = {
            "S": {"layers": 12, "heads": 6, "hidden": 384},
            "B": {"layers": 12, "heads": 12, "hidden": 768},
            "L": {"layers": 24, "heads": 16, "hidden": 1024},
            "XL": {"layers": 28, "heads": 16, "hidden": 1152},
        }

        config = MODEL_CONFIGS[model_size]

        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = config["hidden"]
        self.num_layers = config["layers"]
        self.num_heads = config["heads"]
        self.clip_embed_dim = clip_embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels, self.hidden_size, patch_size, image_size
        )
        self.num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = PositionEmbed(self.num_patches, self.hidden_size)

        # Text projection (CLIP -> hidden_size)
        self.text_proj = nn.Linear(clip_embed_dim, self.hidden_size)

        # Timestep embedding
        self.timestep_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # AdaLN-Zero for timestep conditioning
        self.ada_ln_zero = AdaLNZero(self.hidden_size, self.num_layers)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio,
                attn_dropout,
                mlp_dropout,
            )
            for _ in range(self.num_layers)
        ])

        # Final layer
        self.final_layer = FinalLayer(self.hidden_size, patch_size, in_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following DiT paper."""
        # Initialize patch embed projection
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Initialize pos embed
        nn.init.normal_(self.pos_embed.pos_embed, std=0.02)

        # Initialize timestep embed
        for layer in self.timestep_embed:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # Initialize text projection
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.constant_(self.text_proj.bias, 0)

        # Initialize DiT blocks
        for block in self.blocks:
            if isinstance(block.attn, nn.MultiheadAttention):
                nn.init.xavier_uniform_(block.attn.in_proj_weight)
                nn.init.constant_(block.attn.in_proj_bias, 0)
            if isinstance(block.cross_attn, nn.MultiheadAttention):
                nn.init.xavier_uniform_(block.cross_attn.in_proj_weight)
                nn.init.constant_(block.cross_attn.in_proj_bias, 0)
            for layer in block.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

        # Initialize final layer
        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.norm.weight, 1.0)
        nn.init.constant_(self.final_layer.norm.bias, 0.0)

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
            text_embeds: Text embeddings from CLIP (B, L, D_clip)

        Returns:
            Predicted noise (B, C, H, W)
        """
        # Project text embeddings to hidden size
        text_embeds = self.text_proj(text_embeds)  # (B, L, D_hidden)

        # Get timestep embedding
        if isinstance(timestep, torch.Tensor) and timestep.dim() == 1:
            timestep = self._get_timestep_embedding(timestep)
        elif not isinstance(timestep, torch.Tensor):
            timestep = self._get_timestep_embedding(
                torch.tensor([timestep], device=x.device)
            )

        timestep = self.timestep_embed(timestep)  # (B, D)

        # Get AdaLN-Zero parameters
        scales, shifts, gates = self.ada_ln_zero(timestep)

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        x = self.pos_embed(x)  # (B, N, D)

        # DiT blocks
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                text_embeds=text_embeds,
                scale=scales[i],
                shift=shifts[i],
                gate=gates[i],
            )

        # Final layer
        x = self.final_layer(x)  # (B, C, H, W)

        return x

    def _get_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        max_period: int = 10000,
    ) -> torch.Tensor:
        """Create sinusoidal timestep embedding.

        Args:
            timesteps: Tensor of timesteps (B,)
            max_period: Maximum period for cosine embedding

        Returns:
            Timestep embedding (B, D)
        """
        half = self.hidden_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding.to(dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype from first parameter."""
        for param in self.parameters():
            return param.dtype
        return torch.float32

    def parameters_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_size_info(self) -> dict:
        """Get model size information."""
        MODEL_CONFIGS = {
            "S": "~30M",
            "B": "~130M",
            "L": "~300M",
            "XL": "~675M",
        }
        return {
            "model_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_parameters": self.parameters_count(),
            "estimated_size": MODEL_CONFIGS.get(
                next((k for k, v in {
                    "S": 384, "B": 768, "L": 1024, "XL": 1152
                }.items() if v == self.hidden_size), "S"),
                "~30M"
            ),
        }


if __name__ == "__main__":
    # Quick test
    model = DiT(model_size="S", patch_size=2, image_size=32)
    info = model.get_model_size_info()
    print(f"DiT-S Parameters: {info['num_parameters']:,}")
    print(f"Hidden Size: {info['model_size']}")
    print(f"Layers: {info['num_layers']}")
    print(f"Heads: {info['num_heads']}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    text = torch.randn(2, 77, 512)  # CLIP max context length

    with torch.no_grad():
        out = model(x, t, text)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
