"""Diffusion Transformer (DiT) architecture.

Implements the DiT model from "Scalable Diffusion Models with Transformers"
(Google Research, 2023): https://arxiv.org/abs/2212.09748

Also supports MMDiT (Multi-Modal DiT) from Stable Diffusion 3:
https://github.com/lucidrains/mmdit
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


try:
    from mmdit import MMDiT as MMDitModel

    MMDI_AVAILABLE = True
except ImportError:
    MMDI_AVAILABLE = False


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

        # Project timestep to per-block scale and shift parameters
        # Each layer gets 6 adaptive parameters (following official DiT)
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
            for each block
        """
        temb = self.ada_lin(temb)  # (B, num_layers * D * 6)
        temb = temb.view(temb.size(0), self.num_layers, -1)  # (B, num_layers, D*6)

        # Split into 6 parameters for each layer (following official DiT order)
        shifts_msa = []
        scales_msa = []
        gates_msa = []
        shifts_mlp = []
        scales_mlp = []
        gates_mlp = []

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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive modulation (shift and scale) to normalized input."""
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """DiT Transformer Block with Cross-Attention.

    Architecture:
        x -> LayerNorm -> Self-Attention -> Add
        x -> LayerNorm -> Cross-Attention (with text) -> Add
        x -> LayerNorm -> MLP -> Add

    AdaLN-Zero modulation is applied to self-attention and MLP branches.
    Cross-attention uses standard LayerNorm (text conditioning is separate).
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

        # Self-Attention on image tokens (AdaLN-Zero: no learnable affine params)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attn_dropout, batch_first=True
        )

        # Cross-Attention for text conditioning (standard LayerNorm)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attn_dropout, batch_first=True
        )

        # MLP (AdaLN-Zero: no learnable affine params)
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
        """Forward pass with AdaLN-Zero conditioning.

        Args:
            x: Input tokens (B, N, D)
            text_embeds: Text conditioning (B, L, D), optional
            shift_msa: AdaLN shift for self-attention (B, 1, D)
            scale_msa: AdaLN scale for self-attention (B, 1, D)
            gate_msa: AdaLN gate for self-attention (B, 1, D)
            shift_mlp: AdaLN shift for MLP (B, 1, D)
            scale_mlp: AdaLN scale for MLP (B, 1, D)
            gate_mlp: AdaLN gate for MLP (B, 1, D)

        Returns:
            Processed tokens (B, N, D)
        """
        # Self-Attention with AdaLN-Zero modulation
        x_norm = self.norm1(x)
        if shift_msa is not None and scale_msa is not None:
            x_norm = modulate(x_norm, shift_msa, scale_msa)

        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        if gate_msa is not None:
            x_attn = gate_msa * x_attn
        x = x + x_attn

        # Cross-Attention with text conditioning (no AdaLN modulation)
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

    Includes AdaLN modulation for timestep conditioning (following official DiT).
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

        # AdaLN modulation for final layer (2 parameters: shift, scale)
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
        # Apply AdaLN modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift.unsqueeze(1), scale.unsqueeze(1))
        x = self.linear(x)  # (B, N, patch_size^2 * C)

        # Reshape to image
        B, N, _ = x.shape
        h = w = int(N**0.5)

        # (B, N, p*p*C) -> (B, h, w, p, p, C)
        x = x.view(B, h, w, self.patch_size, self.patch_size, self.out_channels)

        # (B, h, w, p, p, C) -> (B, C, h, p, w, p)
        x = x.permute(0, 5, 1, 3, 2, 4)

        # (B, C, h*p, w*p)
        x = x.contiguous().view(B, self.out_channels, h * self.patch_size, w * self.patch_size)

        return x


class DiT(nn.Module):
    """Diffusion Transformer (DiT) or MMDiT Model.

    Architecture:
        Input Image -> Patch Embed -> Position Embed -> DiT Blocks -> Final Layer -> Output Image

    Conditioning:
        - Timestep: Via AdaLN-Zero in each block
        - Text: Via Cross-Attention in each block (DiT) or Joint Attention (MMDiT)

    Supports two model types:
        - "dit": Standard DiT with cross-attention for text conditioning
        - "mmdit": Multi-Modal DiT from Stable Diffusion 3 with joint text-image attention
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
        self.model_size = model_size
        self.model_type = model_type
        self.hidden_size = config["hidden"]
        self.num_layers = config["layers"]
        self.num_heads = config["heads"]
        self.clip_embed_dim = clip_embed_dim

        if model_type == "mmdit":
            if not MMDI_AVAILABLE:
                raise ImportError("mmdit is not installed. Install with: pip install mmdit")
            self._init_mmdit(
                in_channels,
                image_size,
                patch_size,
                clip_embed_dim,
                mlp_ratio,
                qk_rmsnorm,
                register_tokens,
            )
        else:
            self._init_standard_dit(
                in_channels,
                image_size,
                patch_size,
                clip_embed_dim,
                mlp_ratio,
                attn_dropout,
                mlp_dropout,
            )

    def _init_standard_dit(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        clip_embed_dim: int,
        mlp_ratio: float,
        attn_dropout: float,
        mlp_dropout: float,
    ) -> None:
        """Initialize standard DiT with cross-attention."""
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

    def _init_mmdit(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        clip_embed_dim: int,
        mlp_ratio: float,
        qk_rmsnorm: bool,
        register_tokens: int,
    ) -> None:
        """Initialize MMDiT from lucidrains/mmdit library."""
        # For MMDiT, we use dim_image as hidden_size and dim_cond for timestep conditioning
        self.mmdit = MMDitModel(
            depth=self.num_layers,
            dim_image=self.hidden_size,
            dim_text=clip_embed_dim,
            dim_cond=self.hidden_size,  # Enable timestep conditioning
            num_register_tokens=register_tokens,
            qk_rmsnorm=qk_rmsnorm,
        )

        # Patch embedding for images (output dimension = hidden_size)
        self.patch_embed = PatchEmbed(in_channels, self.hidden_size, patch_size, image_size)
        self.num_patches = self.patch_embed.num_patches

        # Position embedding for image tokens
        self.pos_embed = PositionEmbed(self.num_patches, self.hidden_size)

        # Register tokens count (stored for forward pass)
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
        self._init_weights_mmdit()

    def _init_weights(self) -> None:
        """Initialize weights following DiT paper.

        Key initialization strategies:
        - Xavier uniform for most linear layers
        - Zero initialization for AdaLN output layers (critical for stable training)
        - Normal distribution (std=0.02) for positional embeddings
        """
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

        # Zero-initialize AdaLN-Zero output layer (critical for DiT)
        nn.init.constant_(self.ada_ln_zero.ada_lin[-1].weight, 0)
        nn.init.constant_(self.ada_ln_zero.ada_lin[-1].bias, 0)

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
        # Zero-initialize FinalLayer's AdaLN output (following DiT paper)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def _init_weights_mmdit(self) -> None:
        """Initialize weights for MMDiT model."""
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

        # Initialize final layer
        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

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
        if self.model_type == "mmdit":
            return self._forward_mmdit(x, timestep, text_embeds)
        else:
            return self._forward_standard_dit(x, timestep, text_embeds)

    def _forward_standard_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for standard DiT with cross-attention."""
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

        # Get AdaLN-Zero parameters (6 per block)
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

        # Final layer (with timestep conditioning)
        x = self.final_layer(x, timestep)  # (B, C, H, W)

        return x

    def _forward_mmdit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for MMDiT with joint text-image attention.

        MMDiT uses joint attention where text and image tokens attend to each other.
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
        # Use the same dtype as the model parameters for consistency
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

    def parameters_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_size_info(self) -> dict:
        """Get model size information."""
        ESTIMATED_SIZES = {
            "S": "~30M",
            "B": "~130M",
            "L": "~300M",
            "XL": "~675M",
        }
        return {
            "model_size": self.model_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_parameters": self.parameters_count(),
            "estimated_size": ESTIMATED_SIZES.get(self.model_size, "~30M"),
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
    text = torch.randn(2, 1, 512)  # CLIP pooled embedding (B, 1, D)

    with torch.no_grad():
        out = model(x, t, text)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
