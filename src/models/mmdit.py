"""Multi-Modal Diffusion Transformer (MMDiT) from Stable Diffusion 3.

From "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
(Esser et al., 2024): https://arxiv.org/abs/2403.03206
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from src.models.layers import MODEL_CONFIGS, FinalLayer, PatchEmbed, PositionEmbed

try:
    from mmdit import MMDiT as MMDitModel

    MMDIT_AVAILABLE = True
except ImportError:
    MMDIT_AVAILABLE = False


class MMDiT(nn.Module):
    """Multi-Modal Diffusion Transformer from Stable Diffusion 3.

    Architecture:
        - Uses joint text-image attention instead of cross-attention
        - Text and image tokens attend to each other in unified attention
        - Supports qk_rmsnorm and register_tokens for improved training
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 2,
        model_size: Literal["S", "B", "L", "XL"] = "S",
        clip_embed_dim: int = 512,
        qk_rmsnorm: bool = True,
        register_tokens: int = 0,
    ) -> None:
        super().__init__()

        if not MMDIT_AVAILABLE:
            raise ImportError("mmdit is not installed. Install with: pip install mmdit")

        config = MODEL_CONFIGS[model_size]

        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.model_size = model_size
        self.hidden_size = config["hidden"]
        self.num_layers = config["layers"]
        self.num_heads = config["heads"]
        self.clip_embed_dim = clip_embed_dim

        # MMDiT model from lucidrains
        self.mmdit = MMDitModel(
            depth=self.num_layers,
            dim_image=self.hidden_size,
            dim_text=clip_embed_dim,
            dim_cond=self.hidden_size,
            num_register_tokens=register_tokens,
            qk_rmsnorm=qk_rmsnorm,
        )

        # Patch embedding for images
        self.patch_embed = PatchEmbed(in_channels, self.hidden_size, patch_size, image_size)
        self.num_patches = self.patch_embed.num_patches

        # Position embedding for image tokens
        self.pos_embed = PositionEmbed(self.num_patches, self.hidden_size)

        self.register_tokens = register_tokens

        # Timestep embedding
        self.timestep_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Final layer to decode patch embeddings to image
        self.final_layer = FinalLayer(self.hidden_size, patch_size, in_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for MMDiT model."""
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        nn.init.constant_(self.patch_embed.proj.bias, 0)
        nn.init.normal_(self.pos_embed.pos_embed, std=0.02)

        for layer in self.timestep_embed:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

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
        # Get timestep embedding
        if isinstance(timestep, torch.Tensor) and timestep.dim() == 1:
            timestep = self._get_timestep_embedding(timestep)
        elif not isinstance(timestep, torch.Tensor):
            timestep = self._get_timestep_embedding(torch.tensor([timestep], device=x.device))

        time_cond = self.timestep_embed(timestep)  # (B, D)

        # Patch embedding for image
        image_tokens = self.patch_embed(x)  # (B, N, D_hidden)
        image_tokens = self.pos_embed(image_tokens)  # (B, N, D_hidden)

        # Prepare text tokens (B, D) -> (B, 1, D)
        if text_embeds.dim() == 2:
            text_tokens = text_embeds.unsqueeze(1)  # (B, 1, D_text)
        else:
            text_tokens = text_embeds

        # MMDiT forward pass with joint text-image attention
        text_out, image_out = self.mmdit(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            text_mask=None,
            time_cond=time_cond,
        )

        # Final layer to decode image tokens
        x = self.final_layer(image_out, time_cond)  # (B, C, H, W)

        return x

    def parameters_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
