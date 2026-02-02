"""Motion Module for temporal modeling in video/GIF generation.

This module implements AnimateDiff-style temporal attention layers
that can be plugged into existing image diffusion models.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TemporalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for temporal sequence."""

    def __init__(self, hidden_size: int, max_frames: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_frames = max_frames

        # Pre-compute sinusoidal embeddings
        position = torch.arange(max_frames).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )

        pe = torch.zeros(max_frames, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, num_frames: int) -> torch.Tensor:
        """Get position embeddings for given number of frames.

        Args:
            num_frames: Number of frames in the sequence

        Returns:
            Position embeddings (1, F, D)
        """
        return self.pe[:num_frames].unsqueeze(0)


class TemporalTransformerBlock(nn.Module):
    """Transformer block for temporal attention across frames.

    This block performs self-attention across the temporal dimension,
    allowing the model to learn motion patterns between frames.

    Input shape: (B*N, F, D) where N = num_spatial_tokens, F = num_frames
    Output shape: (B*N, F, D)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_frames: int = 64,
    ) -> None:
        """Initialize TemporalTransformerBlock.

        Args:
            hidden_size: Hidden dimension (must match spatial model)
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            max_frames: Maximum number of frames supported
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Temporal position embedding
        self.pos_embed = TemporalPositionEmbedding(hidden_size, max_frames)

        # Self-attention for temporal modeling
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        # Zero-initialize output projection for stable training
        self._zero_init_output()

    def _zero_init_output(self) -> None:
        """Zero-initialize the output projection for residual connections.

        This ensures the motion module has no effect at initialization,
        allowing the pretrained spatial model to work normally.
        """
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal attention.

        Args:
            x: Input tensor (B*N, F, D)

        Returns:
            Output tensor with temporal modeling (B*N, F, D)
        """
        num_frames = x.shape[1]

        # Add temporal position embedding
        pos_embed = self.pos_embed(num_frames).to(x.device, dtype=x.dtype)
        x_with_pos = x + pos_embed

        # Temporal self-attention
        x_norm = self.norm1(x_with_pos)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feed-forward
        x = x + self.mlp(self.norm2(x))

        return x


class MotionModule(nn.Module):
    """Motion Module that adds temporal modeling to spatial features.

    This module is inserted after spatial transformer blocks to enable
    temporal coherence across video frames. It follows the AnimateDiff
    design where motion modules are trained while the base model is frozen.

    The module:
    1. Reshapes spatial features to temporal format
    2. Applies temporal transformer blocks
    3. Reshapes back to spatial format
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_frames: int = 64,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize MotionModule.

        Args:
            hidden_size: Hidden dimension (must match base model)
            num_layers: Number of temporal transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            max_frames: Maximum number of frames supported
            use_gradient_checkpointing: Enable gradient checkpointing for memory savings
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Stack of temporal transformer blocks
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_frames=max_frames,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Apply temporal modeling to spatial features.

        Args:
            x: Spatial features (B*F, N, D) where B*F is batch*frames
            num_frames: Number of frames in the video

        Returns:
            Temporally modeled features (B*F, N, D)
        """
        bf, n, d = x.shape
        batch_size = bf // num_frames

        # Reshape: (B*F, N, D) -> (B, F, N, D) -> (B*N, F, D)
        x = x.view(batch_size, num_frames, n, d)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, F, D)
        x = x.view(batch_size * n, num_frames, d)  # (B*N, F, D)

        # Apply temporal transformer blocks
        for block in self.temporal_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Reshape back: (B*N, F, D) -> (B, N, F, D) -> (B*F, N, D)
        x = x.view(batch_size, n, num_frames, d)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, F, N, D)
        x = x.view(batch_size * num_frames, n, d)  # (B*F, N, D)

        return x

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory optimization."""
        self.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.use_gradient_checkpointing = False

    def parameters_count(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_motion_module(
    hidden_size: int,
    num_layers: int = 2,
    num_heads: int = 8,
    max_frames: int = 32,
    use_gradient_checkpointing: bool = False,
) -> MotionModule:
    """Factory function to create a MotionModule.

    Args:
        hidden_size: Hidden dimension (must match base model)
        num_layers: Number of temporal transformer layers
        num_heads: Number of attention heads
        max_frames: Maximum number of frames
        use_gradient_checkpointing: Enable gradient checkpointing for memory savings

    Returns:
        Configured MotionModule instance
    """
    return MotionModule(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_frames=max_frames,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
