"""Animated MMDiT for video/GIF generation.

This module wraps a pretrained MMDiT model with Motion Modules
to enable temporal coherence across video frames.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from src.models.mmdit import MMDiT
from src.models.motion import MotionModule


class AnimatedMMDiT(nn.Module):
    """MMDiT wrapper with Motion Modules for video generation.

    This class wraps a pretrained (frozen) MMDiT model and adds
    trainable Motion Modules for temporal modeling. The architecture
    follows the AnimateDiff approach:

    1. Base MMDiT is frozen (pretrained weights)
    2. Motion Modules are trainable (randomly initialized)
    3. Temporal attention is applied to image tokens

    Input: (B, F, C, H, W) - batch of video frames
    Output: (B, F, C, H, W) - predicted velocity for each frame
    """

    def __init__(
        self,
        base_model: MMDiT,
        num_frames: int = 16,
        motion_num_layers: int = 2,
        motion_num_heads: int = 8,
        freeze_base: bool = True,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize AnimatedMMDiT.

        Args:
            base_model: Pretrained MMDiT model
            num_frames: Default number of frames
            motion_num_layers: Number of temporal transformer layers
            motion_num_heads: Number of attention heads in motion module
            freeze_base: Whether to freeze base model weights
            use_gradient_checkpointing: Enable gradient checkpointing for memory savings
        """
        super().__init__()

        self.base_model = base_model
        self.num_frames = num_frames
        self.hidden_size = base_model.hidden_size

        # Create motion module matching base model's hidden size
        self.motion_module = MotionModule(
            hidden_size=self.hidden_size,
            num_layers=motion_num_layers,
            num_heads=motion_num_heads,
            max_frames=64,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Freeze base model if requested
        if freeze_base:
            self._freeze_base_model()

    def _freeze_base_model(self) -> None:
        """Freeze all parameters in the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _unfreeze_base_model(self) -> None:
        """Unfreeze all parameters in the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for the motion module."""
        self.motion_module.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing for the motion module."""
        self.motion_module.disable_gradient_checkpointing()

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
        """Forward pass for video generation.

        Args:
            x: Noisy video frames (B, F, C, H, W) or (B*F, C, H, W)
            timestep: Diffusion timestep (B,) - same for all frames
            text_embeds: Text embeddings from CLIP (B, D_clip)

        Returns:
            Predicted velocity (B, F, C, H, W) or (B*F, C, H, W)
        """
        # Handle input shape
        if x.dim() == 5:
            # (B, F, C, H, W) format
            batch_size, num_frames, c, h, w = x.shape
            x = x.view(batch_size * num_frames, c, h, w)
            reshape_output = True
        else:
            # (B*F, C, H, W) format - infer from batch size
            bf, c, h, w = x.shape
            batch_size = timestep.shape[0]
            num_frames = bf // batch_size
            reshape_output = False

        # Expand timestep and text_embeds for all frames
        # timestep: (B,) -> (B*F,)
        timestep_expanded = timestep.repeat_interleave(num_frames)

        # text_embeds: (B, D) -> (B*F, D)
        text_embeds_expanded = text_embeds.repeat_interleave(num_frames, dim=0)

        # Get timestep embedding
        if timestep_expanded.dim() == 1:
            timestep_emb = self.base_model._get_timestep_embedding(timestep_expanded)
        else:
            timestep_emb = timestep_expanded

        time_cond = self.base_model.timestep_embed(timestep_emb)  # (B*F, D)

        # Patch embedding for images
        image_tokens = self.base_model.patch_embed(x)  # (B*F, N, D_hidden)
        image_tokens = self.base_model.pos_embed(image_tokens)  # (B*F, N, D_hidden)

        # Apply motion module BEFORE mmdit (pre-attention temporal modeling)
        image_tokens = self.motion_module(image_tokens, num_frames)

        # Prepare text tokens
        if text_embeds_expanded.dim() == 2:
            text_tokens = text_embeds_expanded.unsqueeze(1)  # (B*F, 1, D_text)
        else:
            text_tokens = text_embeds_expanded

        # MMDiT forward pass with joint text-image attention
        text_out, image_out = self.base_model.mmdit(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            text_mask=None,
            time_cond=time_cond,
        )

        # Apply motion module AFTER mmdit (post-attention temporal modeling)
        image_out = self.motion_module(image_out, num_frames)

        # Final layer to decode image tokens
        output = self.base_model.final_layer(image_out, time_cond)  # (B*F, C, H, W)

        # Reshape output if input was 5D
        if reshape_output:
            output = output.view(batch_size, num_frames, c, h, w)

        return output

    def parameters_count(self) -> dict[str, int]:
        """Count parameters in base model and motion module.

        Returns:
            Dictionary with parameter counts
        """
        base_params = sum(p.numel() for p in self.base_model.parameters())
        base_trainable = sum(
            p.numel() for p in self.base_model.parameters() if p.requires_grad
        )
        motion_params = self.motion_module.parameters_count()

        return {
            "base_total": base_params,
            "base_trainable": base_trainable,
            "motion_trainable": motion_params,
            "total_trainable": base_trainable + motion_params,
        }

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get only trainable parameters (for optimizer).

        Returns:
            List of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]


def create_animated_mmdit(
    base_model: MMDiT | None = None,
    # Base model config (if base_model is None)
    in_channels: int = 16,
    image_size: int = 8,
    patch_size: int = 2,
    model_size: Literal["S", "B", "L", "XL"] = "B",
    clip_embed_dim: int = 512,
    # Motion module config
    num_frames: int = 16,
    motion_num_layers: int = 2,
    motion_num_heads: int = 8,
    freeze_base: bool = True,
    use_gradient_checkpointing: bool = False,
) -> AnimatedMMDiT:
    """Factory function to create an AnimatedMMDiT.

    Args:
        base_model: Pretrained MMDiT (if None, creates new one)
        in_channels: Input channels (latent channels from VAE)
        image_size: Latent spatial size
        patch_size: Patch size for tokenization
        model_size: Model size (S/B/L/XL)
        clip_embed_dim: CLIP embedding dimension
        num_frames: Number of video frames
        motion_num_layers: Temporal transformer layers
        motion_num_heads: Temporal attention heads
        freeze_base: Whether to freeze base model
        use_gradient_checkpointing: Enable gradient checkpointing for memory savings

    Returns:
        Configured AnimatedMMDiT instance
    """
    if base_model is None:
        base_model = MMDiT(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            model_size=model_size,
            clip_embed_dim=clip_embed_dim,
        )

    return AnimatedMMDiT(
        base_model=base_model,
        num_frames=num_frames,
        motion_num_layers=motion_num_layers,
        motion_num_heads=motion_num_heads,
        freeze_base=freeze_base,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def load_animated_mmdit(
    base_checkpoint_path: str,
    motion_checkpoint_path: str | None = None,
    device: str = "cpu",
    **kwargs,
) -> AnimatedMMDiT:
    """Load AnimatedMMDiT from checkpoints.

    Args:
        base_checkpoint_path: Path to base MMDiT checkpoint
        motion_checkpoint_path: Path to motion module checkpoint (optional)
        device: Device to load model on
        **kwargs: Additional arguments for create_animated_mmdit

    Returns:
        Loaded AnimatedMMDiT instance
    """
    # Load base model checkpoint
    base_ckpt = torch.load(base_checkpoint_path, map_location=device, weights_only=False)

    # Extract model state dict
    if "model_state_dict" in base_ckpt:
        base_state_dict = base_ckpt["model_state_dict"]
    elif "ema_state_dict" in base_ckpt:
        base_state_dict = base_ckpt["ema_state_dict"]
    else:
        base_state_dict = base_ckpt

    # Get model config from checkpoint or use defaults
    model_config = base_ckpt.get("model_config", base_ckpt.get("config", {}))

    # Determine model size
    model_size = model_config.get("model_size")
    if model_size is None:
        # Try to infer from state dict
        patch_embed_weight = base_state_dict.get(
            "patch_embed.proj.weight",
            base_state_dict.get("base_model.patch_embed.proj.weight")
        )
        if patch_embed_weight is not None:
            hidden_size = patch_embed_weight.shape[0]
            size_map = {384: "S", 768: "B", 864: "B", 1024: "L", 1152: "XL"}
            model_size = size_map.get(hidden_size, "S")
        else:
            model_size = kwargs.get("model_size", "S")

    # Create base model
    base_model = MMDiT(
        in_channels=kwargs.get("in_channels", model_config.get("in_channels", 16)),
        image_size=kwargs.get("image_size", model_config.get("latent_size", 8)),
        patch_size=kwargs.get("patch_size", model_config.get("patch_size", 2)),
        model_size=model_size,
        clip_embed_dim=kwargs.get("clip_embed_dim", model_config.get("clip_embed_dim", 512)),
    )
    base_model.load_state_dict(base_state_dict, strict=False)

    # Create animated model
    animated_model = AnimatedMMDiT(
        base_model=base_model,
        num_frames=kwargs.get("num_frames", 16),
        motion_num_layers=kwargs.get("motion_num_layers", 2),
        motion_num_heads=kwargs.get("motion_num_heads", 8),
        freeze_base=kwargs.get("freeze_base", True),
        use_gradient_checkpointing=kwargs.get("use_gradient_checkpointing", False),
    )

    # Load motion module if provided
    if motion_checkpoint_path is not None:
        motion_ckpt = torch.load(motion_checkpoint_path, map_location=device, weights_only=False)
        if "motion_module_state_dict" in motion_ckpt:
            motion_state_dict = motion_ckpt["motion_module_state_dict"]
        else:
            motion_state_dict = motion_ckpt
        animated_model.motion_module.load_state_dict(motion_state_dict)

    return animated_model.to(device)
