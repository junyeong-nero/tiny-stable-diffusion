"""Animated Diffusion for video/GIF generation.

Extends the base Diffusion class to support temporal dimensions
for training motion modules and generating video sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import tqdm

from src.models.diffusion import Diffusion


class AnimatedDiffusion(Diffusion):
    """Rectified Flow diffusion extended for video generation.

    Supports temporal dimension in all operations:
    - Forward process: q_sample for video latents
    - Training: velocity prediction across frames
    - Sampling: generate coherent video sequences

    Latent shape: (B, F, C, H, W) where F = num_frames
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_frames: int = 16,
        guidance_scale: float = 7.5,
        cfg_probability: float = 0.1,
        uncond_embed: torch.Tensor | None = None,
        min_snr_gamma: float | None = 5.0,
        temporal_consistency_weight: float = 0.0,
    ) -> None:
        """Initialize AnimatedDiffusion.

        Args:
            num_timesteps: Number of diffusion timesteps
            num_frames: Default number of frames
            guidance_scale: CFG guidance scale for sampling
            cfg_probability: Probability of dropping text conditioning
            uncond_embed: Pre-computed unconditional embedding
            min_snr_gamma: Min-SNR gamma for loss weighting
            temporal_consistency_weight: Weight for temporal consistency loss
        """
        super().__init__(
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            cfg_probability=cfg_probability,
            uncond_embed=uncond_embed,
            min_snr_gamma=min_snr_gamma,
        )
        self.num_frames = num_frames
        self.temporal_consistency_weight = temporal_consistency_weight

    def q_sample_video(
        self,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward process for video latents.

        Applies the same timestep to all frames in a video,
        maintaining temporal coherence in the noising process.

        Args:
            x_0: Clean video latents (B, F, C, H, W)
            timesteps: Timestep indices (B,) - same for all frames
            noise: Noise tensor (B, F, C, H, W), optional

        Returns:
            Noisy video latents at timestep t (B, F, C, H, W)
        """
        b, f, c, h, w = x_0.shape

        if noise is None:
            noise = torch.randn_like(x_0)

        # Normalize timesteps to [0, 1]
        t = timesteps.float() / self.num_timesteps
        t = t.view(b, 1, 1, 1, 1)  # (B, 1, 1, 1, 1) for 5D broadcasting

        # Linear interpolation: x_t = (1 - t) * x_0 + t * noise
        x_t = (1.0 - t) * x_0 + t * noise

        return x_t

    def get_velocity_video(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target velocity for video training.

        Args:
            x_0: Clean video latents (B, F, C, H, W)
            noise: Noise tensor (B, F, C, H, W)

        Returns:
            Target velocity (B, F, C, H, W)
        """
        return noise - x_0

    def training_loss_video(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculate training loss for video diffusion.

        Args:
            model: AnimatedMMDiT that predicts velocity
            x_0: Clean video latents (B, F, C, H, W)
            timesteps: Timestep indices (B,)
            text_embeds: Text embeddings (B, D)

        Returns:
            Tuple of (total loss, loss dict with components)
        """
        b, f, c, h, w = x_0.shape

        noise = torch.randn_like(x_0)
        x_t = self.q_sample_video(x_0, timesteps, noise)

        # Target velocity: v = noise - x_0
        v_target = self.get_velocity_video(x_0, noise)

        # Apply CFG dropout using unconditional embedding
        if self.cfg_probability > 0 and self.uncond_embed is not None:
            drop_mask = torch.rand(b, device=x_0.device) < self.cfg_probability

            if drop_mask.any():
                text_embeds = text_embeds.clone()
                uncond_embed = self.uncond_embed.to(x_0.device)
                uncond_expanded = uncond_embed.expand(b, -1)
                text_embeds = torch.where(
                    drop_mask.unsqueeze(-1), uncond_expanded, text_embeds
                )

        # Predict velocity - model expects 5D input
        v_pred = model(x_t, timesteps, text_embeds)

        # Compute per-sample MSE loss
        per_sample_loss = nn.functional.mse_loss(
            v_pred, v_target, reduction="none"
        ).mean(dim=(1, 2, 3, 4))  # Mean over F, C, H, W

        # Apply Min-SNR weighting
        if self.min_snr_gamma is not None:
            snr_weights = self.get_min_snr_weight(timesteps)
            per_sample_loss = per_sample_loss * snr_weights

        # Mean over batch
        velocity_loss = per_sample_loss.mean()

        # Temporal consistency loss (optional)
        temporal_loss = torch.tensor(0.0, device=x_0.device)
        if self.temporal_consistency_weight > 0:
            # Encourage smooth velocity predictions across frames
            v_diff = v_pred[:, 1:] - v_pred[:, :-1]  # (B, F-1, C, H, W)
            temporal_loss = v_diff.pow(2).mean()

        total_loss = velocity_loss + self.temporal_consistency_weight * temporal_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "velocity_loss": velocity_loss.item(),
            "temporal_loss": temporal_loss.item(),
        }

        return total_loss, loss_dict

    def euler_step_video(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_curr: torch.Tensor,
        t_next: torch.Tensor,
        text_embeds: torch.Tensor,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        """Single Euler step for video ODE sampling.

        Args:
            model: AnimatedMMDiT that predicts velocity
            x_t: Current noisy video (B, F, C, H, W)
            t_curr: Current timestep (B,)
            t_next: Next timestep (B,)
            text_embeds: Text embeddings (B, D)
            use_cfg: Whether to apply classifier-free guidance

        Returns:
            Denoised video at t_next (B, F, C, H, W)
        """
        b, f, c, h, w = x_t.shape

        # Predict velocity (conditional)
        v_pred = model(x_t, t_curr, text_embeds)

        # Apply classifier-free guidance
        if use_cfg and self.guidance_scale != 1.0 and self.uncond_embed is not None:
            uncond_embed = self.uncond_embed.to(x_t.device)
            if uncond_embed.shape[0] == 1 and b > 1:
                uncond_embed = uncond_embed.expand(b, -1)

            with torch.no_grad():
                v_uncond = model(x_t, t_curr, uncond_embed)

            v_pred = v_uncond + self.guidance_scale * (v_pred - v_uncond)

        # Compute dt
        t_curr_norm = t_curr.float() / self.num_timesteps
        t_next_norm = t_next.float() / self.num_timesteps
        dt = (t_next_norm - t_curr_norm).view(b, 1, 1, 1, 1)

        # Euler step
        x_next = x_t + v_pred * dt

        return x_next

    def sample_video(
        self,
        model: nn.Module,
        batch_size: int,
        num_frames: int,
        latent_channels: int,
        latent_size: int,
        text_embeds: torch.Tensor,
        num_steps: int = 50,
        use_cfg: bool = True,
        seed: int | None = None,
        vae_decoder: "AutoencoderKL | None" = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Generate video samples using Euler ODE solver.

        Args:
            model: AnimatedMMDiT that predicts velocity
            batch_size: Number of videos to generate
            num_frames: Frames per video
            latent_channels: Latent channel dimension
            latent_size: Latent spatial size
            text_embeds: Text embeddings (B, D)
            num_steps: Number of sampling steps
            use_cfg: Use classifier-free guidance
            seed: Random seed
            vae_decoder: Optional VAE decoder for latent-to-image
            device: Device to generate on

        Returns:
            Generated videos (B, F, C, H, W) or (B, F, 3, img_h, img_w)
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = torch.device(device) if isinstance(device, str) else device

        # Start from pure noise (t=1)
        x_t = torch.randn(
            batch_size, num_frames, latent_channels, latent_size, latent_size,
            device=device,
        )

        # Create timestep schedule
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps + 1, device=device
        ).long()

        # Euler sampling loop
        for i in tqdm.tqdm(range(num_steps), desc="Sampling video"):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            t_curr_batch = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)
            t_next_batch = torch.full((batch_size,), t_next, device=device, dtype=torch.long)

            x_t = self.euler_step_video(
                model, x_t, t_curr_batch, t_next_batch, text_embeds, use_cfg=use_cfg
            )

        # Decode each frame if VAE decoder provided
        if vae_decoder is not None:
            b, f, c, h, w = x_t.shape
            x_flat = x_t.view(b * f, c, h, w)
            decoded = vae_decoder.decode_from_latent(x_flat)
            _, c_out, h_out, w_out = decoded.shape
            x_t = decoded.view(b, f, c_out, h_out, w_out)

        # Normalize to [0, 1]
        x_t = (x_t + 1.0) / 2.0
        x_t = torch.clamp(x_t, 0.0, 1.0)

        return x_t

    def __repr__(self) -> str:
        return (
            f"AnimatedDiffusion("
            f"num_timesteps={self.num_timesteps}, "
            f"num_frames={self.num_frames}, "
            f"guidance_scale={self.guidance_scale}, "
            f"cfg_probability={self.cfg_probability}, "
            f"min_snr_gamma={self.min_snr_gamma}, "
            f"temporal_consistency_weight={self.temporal_consistency_weight})"
        )


def create_animated_diffusion(
    num_timesteps: int = 1000,
    num_frames: int = 16,
    guidance_scale: float = 7.5,
    cfg_probability: float = 0.1,
    min_snr_gamma: float | None = 5.0,
    temporal_consistency_weight: float = 0.0,
) -> AnimatedDiffusion:
    """Factory function to create AnimatedDiffusion.

    Args:
        num_timesteps: Number of diffusion timesteps
        num_frames: Default number of frames
        guidance_scale: CFG guidance scale
        cfg_probability: CFG dropout probability
        min_snr_gamma: Min-SNR gamma
        temporal_consistency_weight: Temporal loss weight

    Returns:
        Configured AnimatedDiffusion instance
    """
    return AnimatedDiffusion(
        num_timesteps=num_timesteps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        cfg_probability=cfg_probability,
        min_snr_gamma=min_snr_gamma,
        temporal_consistency_weight=temporal_consistency_weight,
    )
