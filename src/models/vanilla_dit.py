"""Vanilla Diffusion Transformer (DiT) implementation.

From "Scalable Diffusion Models with Transformers" (Google Research, 2023).
https://arxiv.org/abs/2212.09748
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from src.models.layers import (
    MODEL_CONFIGS,
    AdaLNZero,
    DiTBlock,
    FinalLayer,
    PatchEmbed,
    PositionEmbed,
)


class VanillaDiT(nn.Module):
    """Standard Diffusion Transformer with Cross-Attention.

    Architecture:
        Input Image -> Patch Embed -> Position Embed -> DiT Blocks -> Final Layer -> Output

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

        config = MODEL_CONFIGS[model_size]

        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.model_size = model_size
        self.hidden_size = config["hidden"]
        self.num_layers = config["layers"]
        self.num_heads = config["heads"]
        self.clip_embed_dim = clip_embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, self.hidden_size, patch_size, image_size)
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
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio,
                    attn_dropout,
                    mlp_dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(self.hidden_size, patch_size, in_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following DiT paper."""
        # Patch embed projection
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Pos embed
        nn.init.normal_(self.pos_embed.pos_embed, std=0.02)

        # Timestep embed
        for layer in self.timestep_embed:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # Text projection
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.constant_(self.text_proj.bias, 0)

        # Zero-initialize AdaLN-Zero output layer
        nn.init.constant_(self.ada_ln_zero.ada_lin[-1].weight, 0)
        nn.init.constant_(self.ada_ln_zero.ada_lin[-1].bias, 0)

        # DiT blocks
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

        # Final layer
        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def _get_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        max_period: int = 10000,
    ) -> torch.Tensor:
        """Create sinusoidal timestep embedding."""
        half = self.hidden_size // 2
        model_dtype = self.dtype
        model_device = next(self.parameters()).device

        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=model_dtype, device=model_device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding.to(dtype=model_dtype)

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype from first parameter."""
        for param in self.parameters():
            return param.dtype
        return torch.float32

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
        # Add sequence dimension for cross-attention: (B, D) -> (B, 1, D)
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(1)

        # Project text embeddings to hidden size
        text_embeds = self.text_proj(text_embeds)  # (B, 1, D_hidden)

        # Get timestep embedding
        if isinstance(timestep, torch.Tensor) and timestep.dim() == 1:
            timestep = self._get_timestep_embedding(timestep)
        elif not isinstance(timestep, torch.Tensor):
            timestep = self._get_timestep_embedding(torch.tensor([timestep], device=x.device))

        timestep = self.timestep_embed(timestep)  # (B, D)

        # Get AdaLN-Zero parameters
        shifts_msa, scales_msa, gates_msa, shifts_mlp, scales_mlp, gates_mlp = self.ada_ln_zero(
            timestep
        )

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        x = self.pos_embed(x)  # (B, N, D)

        # DiT blocks
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                text_embeds=text_embeds,
                shift_msa=shifts_msa[i],
                scale_msa=scales_msa[i],
                gate_msa=gates_msa[i],
                shift_mlp=shifts_mlp[i],
                scale_mlp=scales_mlp[i],
                gate_mlp=gates_mlp[i],
            )

        # Final layer
        x = self.final_layer(x, timestep)  # (B, C, H, W)

        return x

    def parameters_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
