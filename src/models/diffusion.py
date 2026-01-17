"""Rectified Flow diffusion process implementation.

Implements Rectified Flow from Stable Diffusion 3:
- Linear interpolation forward process: x_t = (1-t)*x_0 + t*noise
- Velocity prediction: v = noise - x_0
- Euler ODE sampling
- CFG: Ho et al., "Classifier-Free Diffusion Guidance" (2021)

For Stable Diffusion 3 style latent-space diffusion:
- Diffusion operates on latent tensors (B, 16, 8, 8)
- VAE decoder converts latent to image after sampling
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import tqdm

if TYPE_CHECKING:
    from src.models.vae import AutoencoderKL


class Diffusion:
    """Rectified Flow diffusion model for image generation.

    Uses linear interpolation and velocity prediction following SD3.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        guidance_scale: float = 7.5,
        cfg_probability: float = 0.1,
        uncond_embed: torch.Tensor | None = None,
    ) -> None:
        """Initialize Rectified Flow diffusion.

        Args:
            num_timesteps: Number of diffusion timesteps (default 1000)
            guidance_scale: CFG guidance scale for sampling
            cfg_probability: Probability of dropping text conditioning during training
            uncond_embed: Pre-computed unconditional embedding for CFG
        """
        self.num_timesteps = num_timesteps
        self.guidance_scale = guidance_scale
        self.cfg_probability = cfg_probability
        self.uncond_embed = uncond_embed

    def q_sample(
        self,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward process: linear interpolation between x_0 and noise.

        Rectified Flow: x_t = (1 - t) * x_0 + t * noise
        where t is normalized to [0, 1].

        Args:
            x_0: Clean images/latents (B, C, H, W)
            timesteps: Timestep indices (B,) in range [0, num_timesteps-1]
            noise: Noise tensor (B, C, H, W), optional

        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Normalize timesteps to [0, 1]
        t = timesteps.float() / self.num_timesteps
        t = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting

        # Linear interpolation: x_t = (1 - t) * x_0 + t * noise
        x_t = (1.0 - t) * x_0 + t * noise

        return x_t

    def get_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target velocity for training.

        Velocity is defined as: v = noise - x_0
        This is the direction from x_0 to noise in the linear path.

        Args:
            x_0: Clean images/latents (B, C, H, W)
            noise: Noise tensor (B, C, H, W)

        Returns:
            Target velocity (B, C, H, W)
        """
        return noise - x_0

    def euler_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_curr: torch.Tensor,
        t_next: torch.Tensor,
        text_embeds: torch.Tensor,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        """Single Euler step for ODE sampling.

        Euler method: x_{t-1} = x_t + v(x_t, t) * dt
        where dt = (t_next - t_curr) / num_timesteps

        Args:
            model: Diffusion model that predicts velocity
            x_t: Current noisy sample (B, C, H, W)
            t_curr: Current timestep (B,)
            t_next: Next timestep (B,)
            text_embeds: Text embeddings (B, D)
            use_cfg: Whether to apply classifier-free guidance

        Returns:
            Denoised sample at t_next
        """
        # Predict velocity (conditional)
        v_pred = model(x_t, t_curr, text_embeds)

        # Apply classifier-free guidance
        if use_cfg and self.guidance_scale != 1.0 and self.uncond_embed is not None:
            uncond_embed = self.uncond_embed.to(x_t.device)
            # Expand uncond_embed to batch size if needed
            if uncond_embed.shape[0] == 1 and x_t.shape[0] > 1:
                uncond_embed = uncond_embed.expand(x_t.shape[0], -1)

            with torch.no_grad():
                v_uncond = model(x_t, t_curr, uncond_embed)

            # CFG: v = v_uncond + scale * (v_cond - v_uncond)
            v_pred = v_uncond + self.guidance_scale * (v_pred - v_uncond)

        # Compute dt (normalized timestep difference)
        t_curr_norm = t_curr.float() / self.num_timesteps
        t_next_norm = t_next.float() / self.num_timesteps

        # dt should be negative (going from t=1 to t=0)
        dt = (t_next_norm - t_curr_norm).view(-1, 1, 1, 1)

        # Euler step: x_next = x_t + v * dt
        x_next = x_t + v_pred * dt

        return x_next

    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        text_embeds: torch.Tensor,
        num_steps: int = 50,
        use_cfg: bool = True,
        seed: int | None = None,
        vae_decoder: "AutoencoderKL | None" = None,
    ) -> torch.Tensor:
        """Generate samples using Euler ODE solver.

        For latent-space diffusion (SD3 style):
            - shape should be (B, latent_channels, latent_h, latent_w)
            - Pass vae_decoder to convert latent to image

        Args:
            model: Diffusion model that predicts velocity
            shape: Output shape (B, C, H, W) - latent shape if using VAE
            text_embeds: Text embeddings (B, D)
            num_steps: Number of sampling steps
            use_cfg: Use classifier-free guidance
            seed: Random seed
            vae_decoder: Optional VAE decoder for latent-to-image conversion

        Returns:
            Generated images (B, C, H, W) normalized to [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, C, H, W = shape
        device = next(model.parameters()).device

        # Start from pure noise (t=1)
        x_t = torch.randn(shape, device=device)

        # Create timestep schedule: evenly spaced from T-1 to 0
        # Example: num_timesteps=1000, num_steps=50 -> [999, 979, 959, ..., 19, 0]
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps + 1, device=device
        ).long()

        # Euler sampling loop
        for i in tqdm.tqdm(range(num_steps), desc="Sampling"):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            t_curr_batch = torch.full((B,), t_curr, device=device, dtype=torch.long)
            t_next_batch = torch.full((B,), t_next, device=device, dtype=torch.long)

            x_t = self.euler_step(
                model,
                x_t,
                t_curr_batch,
                t_next_batch,
                text_embeds,
                use_cfg=use_cfg,
            )

        # Decode latent to image if VAE decoder provided
        if vae_decoder is not None:
            x_t = vae_decoder.decode_from_latent(x_t)

        # Normalize to [0, 1]
        x_t = (x_t + 1.0) / 2.0
        x_t = torch.clamp(x_t, 0.0, 1.0)

        return x_t

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate training loss with velocity prediction and CFG dropout.

        Uses unconditional embedding (empty string "" CLIP embedding) for samples
        where text conditioning is dropped, following Stable Diffusion approach.

        Args:
            model: Diffusion model that predicts velocity
            x_0: Clean images/latents (B, C, H, W)
            timesteps: Timestep indices (B,)
            text_embeds: Text embeddings (B, D)

        Returns:
            Mean squared error loss between predicted and target velocity
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, timesteps, noise)

        # Target velocity: v = noise - x_0
        v_target = self.get_velocity(x_0, noise)

        # Apply CFG dropout using unconditional embedding
        if self.cfg_probability > 0 and self.uncond_embed is not None:
            batch_size = x_0.shape[0]
            drop_mask = torch.rand(batch_size, device=x_0.device) < self.cfg_probability

            if drop_mask.any():
                text_embeds = text_embeds.clone()
                uncond_embed = self.uncond_embed.to(x_0.device)

                # Replace dropped samples with unconditional embedding
                for i in range(batch_size):
                    if drop_mask[i]:
                        text_embeds[i] = uncond_embed[0]

        # Predict velocity
        v_pred = model(x_t, timesteps, text_embeds)

        # MSE loss on velocity
        loss = nn.functional.mse_loss(v_pred, v_target)

        return loss

    def sample_timesteps_logit_normal(
        self,
        batch_size: int,
        device: torch.device,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> torch.Tensor:
        """Sample timesteps using logit-normal distribution (SD3 style).

        Logit-normal sampling concentrates samples around the middle timesteps,
        which improves training efficiency.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on
            mean: Mean of the normal distribution (default 0.0)
            std: Standard deviation of the normal distribution (default 1.0)

        Returns:
            Timestep indices (B,) in range [0, num_timesteps-1]
        """
        # Sample from normal distribution
        u = torch.randn(batch_size, device=device) * std + mean

        # Apply sigmoid to map to [0, 1]
        t = torch.sigmoid(u)

        # Scale to [0, num_timesteps-1] and convert to long
        timesteps = (t * (self.num_timesteps - 1)).long()

        # Clamp to valid range
        timesteps = torch.clamp(timesteps, 0, self.num_timesteps - 1)

        return timesteps

    def __repr__(self) -> str:
        return (
            f"Diffusion("
            f"num_timesteps={self.num_timesteps}, "
            f"guidance_scale={self.guidance_scale}, "
            f"cfg_probability={self.cfg_probability}, "
            f"type=RectifiedFlow)"
        )


if __name__ == "__main__":
    # Quick test
    diffusion = Diffusion(num_timesteps=1000)
    print(f"{diffusion}")

    # Test q_sample (linear interpolation)
    x_0 = torch.randn(2, 4, 8, 8)
    timesteps = torch.tensor([0, 500])
    noise = torch.randn_like(x_0)
    x_t = diffusion.q_sample(x_0, timesteps, noise)
    print(f"q_sample output shape: {x_t.shape}")

    # Test velocity computation
    v = diffusion.get_velocity(x_0, noise)
    print(f"Velocity shape: {v.shape}")

    # Test logit-normal sampling
    timesteps = diffusion.sample_timesteps_logit_normal(1000, torch.device("cpu"))
    print(f"Logit-normal timesteps - mean: {timesteps.float().mean():.1f}, std: {timesteps.float().std():.1f}")
