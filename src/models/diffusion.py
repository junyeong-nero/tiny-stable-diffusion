"""DDPM/DDIM diffusion process implementation.

Implements the diffusion process following:
- DDPM: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- DDIM: Song et al., "Denoising Diffusion Implicit Models" (2020)
- CFG: Ho et al., "Classifier-Free Diffusion Guidance" (2021)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


class Diffusion:
    """Diffusion model for image generation.

    Supports both DDPM (stochastic) and DDIM (deterministic) sampling.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: Literal["linear", "cosine", "quadratic"] = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        use_linear_variance: bool = False,
        guidance_scale: float = 7.5,
        cfg_probability: float = 0.1,
        epsilon: float = 1e-8,
        uncond_embed: torch.Tensor | None = None,
        uncond_mask: torch.Tensor | None = None,
        min_snr_gamma: float = 5.0,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.guidance_scale = guidance_scale
        self.cfg_probability = cfg_probability
        self.epsilon = epsilon
        self.min_snr_gamma = min_snr_gamma

        # Unconditional embedding for classifier-free guidance (empty string "" embedding)
        self.uncond_embed = uncond_embed
        self.uncond_mask = uncond_mask

        # Define beta schedule
        if beta_schedule == "cosine":
            betas = self._cosine_beta_schedule()
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "quadratic":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas

        # Precompute diffusion parameters
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # For DDIM sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_minus_1 = torch.sqrt(
            1.0 / self.alphas_cumprod - 1
        )

        # Posterior mean and variance for DDPM
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # SNR (Signal-to-Noise Ratio) for Min-SNR weighting
        # SNR(t) = alpha_cumprod(t) / (1 - alpha_cumprod(t))
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)

    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule from Improved DDPM paper.

        Args:
            s: Offset parameter

        Returns:
            Beta schedule tensor
        """
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0, 0.999)
        return betas

    def q_sample(
        self,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion process: add noise to images.

        Args:
            x_0: Clean images (B, C, H, W)
            timesteps: Timestep indices (B,)
            noise: Noise tensor (B, C, H, W), optional

        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, timesteps, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_0.shape
        )

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        """DDPM reverse process: denoise one step.

        Args:
            model: Diffusion model
            x_t: Noisy image at timestep t
            timesteps: Current timestep (B,)
            text_embeds: Text embeddings (B, L, D)
            text_mask: Attention mask for text (B, L), optional (True = attend)
            use_cfg: Whether to use classifier-free guidance

        Returns:
            Denoised image at timestep t-1
        """
        B, C, H, W = x_t.shape

        # Predict noise (conditional)
        predicted_noise = model(x_t, timesteps, text_embeds, text_mask)

        # Apply classifier-free guidance during sampling
        if use_cfg and self.guidance_scale != 1.0 and self.uncond_embed is not None:
            # Get unconditional prediction using pre-computed uncond_embed
            uncond_embed = self.uncond_embed.to(x_t.device)
            uncond_mask = (
                self.uncond_mask.to(x_t.device)
                if self.uncond_mask is not None
                else None
            )
            with torch.no_grad():
                unconditional_noise = model(x_t, timesteps, uncond_embed, uncond_mask)
            # CFG: noise = uncond + scale * (cond - uncond)
            predicted_noise = unconditional_noise + self.guidance_scale * (
                predicted_noise - unconditional_noise
            )

        # Predict x_0 from predicted noise
        # x_0 = (x_t - sqrt(1 - alpha_cumprod) * noise) / sqrt(alpha_cumprod)
        pred_x_0 = (
            self._extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - self._extract(
                self.sqrt_recip_alphas_cumprod_minus_1, timesteps, x_t.shape
            )
            * predicted_noise
        )
        # Clamp to valid range for stability
        pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)

        # Calculate posterior mean: coef1 * x_0 + coef2 * x_t
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, timesteps, x_t.shape) * pred_x_0
            + self._extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )

        # Add noise for stochasticity (except at t=0)
        nonzero_mask = (timesteps > 0).float().view(B, *([1] * (x_t.dim() - 1)))
        posterior_std = torch.sqrt(self.posterior_variance + self.epsilon)

        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        noise = (
            torch.randn_like(x_t)
            * nonzero_mask
            * self._extract(posterior_std, timesteps, x_t.shape)
        )

        return posterior_mean + noise

    def ddim_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        eta: float = 0.0,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        """DDIM reverse process: deterministic denoising (Corrected Formula)."""
        B, C, H, W = x_t.shape
        # timestep index retrieval
        t_index = timesteps[0].item()

        # 1. Predict noise (epsilon_theta)
        predicted_noise = model(x_t, timesteps, text_embeds, text_mask)

        if use_cfg and self.guidance_scale != 1.0 and self.uncond_embed is not None:
            # Get unconditional prediction using pre-computed uncond_embed
            uncond_embed = self.uncond_embed.to(x_t.device)
            uncond_mask = (
                self.uncond_mask.to(x_t.device)
                if self.uncond_mask is not None
                else None
            )
            with torch.no_grad():
                unconditional_noise = model(x_t, timesteps, uncond_embed, uncond_mask)
            predicted_noise = unconditional_noise + self.guidance_scale * (
                predicted_noise - unconditional_noise
            )

        # 2. Get alpha constants
        alpha_t = self.alphas_cumprod[t_index]
        alpha_t_prev = (
            self.alphas_cumprod[t_index - 1] if t_index > 0 else torch.tensor(1.0)
        )

        # Move to device
        alpha_t = alpha_t.to(x_t.device)
        alpha_t_prev = alpha_t_prev.to(x_t.device)

        # 3. Predict x_0 (Clamped)
        # x_0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        pred_x_0 = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / torch.sqrt(
            alpha_t
        )
        pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)

        # 4. Compute variance (sigma) for eta
        # sigma = eta * sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        sigma_t = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )

        # 5. Compute "Direction" pointing to x_t
        # dir = sqrt(1 - alpha_prev - sigma^2) * eps
        pred_dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * predicted_noise

        # 6. Compute x_{t-1}
        # x_{t-1} = sqrt(alpha_prev) * x_0 + dir + sigma * noise
        x_t_prev = torch.sqrt(alpha_t_prev) * pred_x_0 + pred_dir_xt

        if eta > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma_t * noise

        return x_t_prev

    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        num_steps: int = 50,
        eta: float = 0.0,
        use_ddim: bool = True,
        use_cfg: bool = True,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate samples from diffusion model.

        Args:
            model: Diffusion model
            shape: Output shape (B, C, H, W)
            text_embeds: Text embeddings (B, L, D)
            text_mask: Attention mask for text (B, L), optional (True = attend)
            num_steps: Number of sampling steps
            eta: DDIM stochasticity parameter
            use_ddim: Use DDIM (True) or DDPM (False)
            use_cfg: Use classifier-free guidance
            seed: Random seed

        Returns:
            Generated images (B, C, H, W)
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, C, H, W = shape
        device = next(model.parameters()).device

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Get sampling timesteps
        if use_ddim:
            # Use evenly spaced steps for DDIM (from T-1 to 0, reversed order)
            # Example: num_timesteps=1000, num_steps=50 -> [999, 979, 959, ..., 19, 0]
            step_indices = torch.linspace(0, self.num_timesteps - 1, num_steps + 1)
            step_indices = step_indices.long()
            # Reverse to go from high timesteps to low (T-1 -> 0)
            timesteps = torch.flip(step_indices, dims=[0])[:-1].to(device)
        else:
            # Use all timesteps for DDPM (from T-1 to 0)
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1, device=device)

        # Sampling loop (timesteps are already in descending order)
        for t in tqdm.tqdm(timesteps, desc="Sampling"):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            if use_ddim:
                x_t = self.ddim_sample(
                    model,
                    x_t,
                    t_batch,
                    text_embeds,
                    text_mask,
                    eta=eta,
                    use_cfg=use_cfg,
                )
            else:
                x_t = self.p_sample(
                    model, x_t, t_batch, text_embeds, text_mask, use_cfg=use_cfg
                )

        # Normalize to [0, 1]
        x_t = (x_t + 1.0) / 2.0
        x_t = torch.clamp(x_t, 0.0, 1.0)

        return x_t

    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Extract values from tensor at timestep indices.

        Args:
            a: Tensor to extract from
            t: Timestep indices (B,)
            x_shape: Shape to broadcast to

        Returns:
            Extracted values with correct shape for broadcasting
        """
        # Move tensor a to the same device as t
        a = a.to(device=t.device)
        B = t.shape[0]
        out = a.gather(dim=0, index=t)
        return out.reshape(B, *([1] * (len(x_shape) - 1)))

    def _get_min_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute Min-SNR loss weights for given timesteps.

        Min-SNR weighting from "Efficient Diffusion Training via Min-SNR Weighting Strategy"
        (Hang et al., 2023). Weights high-noise timesteps more to stabilize training.

        weight(t) = min(SNR(t), gamma) / SNR(t)

        Args:
            timesteps: Timestep indices (B,)

        Returns:
            Loss weights (B,)
        """
        snr = self._extract(
            self.snr, timesteps, (timesteps.shape[0], 1, 1, 1)
        ).squeeze()
        # min(SNR, gamma) / SNR = clamp(gamma / SNR, max=1.0)
        weights = torch.clamp(self.min_snr_gamma / snr, max=1.0)
        return weights

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate training loss with CFG Dropout and Min-SNR weighting.

        Uses unconditional embedding (empty string "" CLIP embedding) for samples
        where text conditioning is dropped, following Stable Diffusion approach.

        Min-SNR weighting helps stabilize training by reducing the weight of
        low-noise timesteps where the signal-to-noise ratio is high.
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, timesteps, noise)

        # Apply CFG dropout using unconditional embedding
        if self.cfg_probability > 0 and self.uncond_embed is not None:
            batch_size = x_0.shape[0]
            drop_mask = torch.rand(batch_size, device=x_0.device) < self.cfg_probability

            if drop_mask.any():
                text_embeds = text_embeds.clone()
                uncond_embed = self.uncond_embed.to(x_0.device)
                uncond_mask = (
                    self.uncond_mask.to(x_0.device)
                    if self.uncond_mask is not None
                    else None
                )

                # Replace dropped samples with unconditional embedding
                for i in range(batch_size):
                    if drop_mask[i]:
                        text_embeds[i] = uncond_embed[0]
                        if text_mask is not None and uncond_mask is not None:
                            text_mask[i] = uncond_mask[0]

        predicted_noise = model(x_t, timesteps, text_embeds, text_mask)

        # Compute per-sample MSE loss
        mse_loss = nn.functional.mse_loss(predicted_noise, noise, reduction="none")
        mse_loss = mse_loss.mean(dim=[1, 2, 3])  # Mean over C, H, W -> (B,)

        # Apply Min-SNR weighting
        snr_weights = self._get_min_snr_weights(timesteps)
        weighted_loss = (mse_loss * snr_weights).mean()

        return weighted_loss

    def __repr__(self) -> str:
        return (
            f"Diffusion("
            f"num_timesteps={self.num_timesteps}, "
            f"guidance_scale={self.guidance_scale}, "
            f"cfg_probability={self.cfg_probability})"
        )


import tqdm  # noqa: E402

if __name__ == "__main__":
    # Quick test
    diffusion = Diffusion(num_timesteps=1000, beta_schedule="cosine")
    print(f"{diffusion}")
    print(f"Betas shape: {diffusion.betas.shape}")
    print(f"Alphas cumprod shape: {diffusion.alphas_cumprod.shape}")
