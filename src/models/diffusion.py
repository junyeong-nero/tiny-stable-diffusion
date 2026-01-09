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
    ) -> None:
        self.num_timesteps = num_timesteps
        self.guidance_scale = guidance_scale
        self.cfg_probability = cfg_probability
        self.epsilon = epsilon

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
        if use_cfg and self.guidance_scale != 1.0:
            # Get unconditional prediction
            with torch.no_grad():
                unconditional_noise = model(
                    x_t, timesteps, torch.zeros_like(text_embeds), text_mask
                )
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

        if use_cfg and self.guidance_scale != 1.0:
            with torch.no_grad():
                unconditional_noise = model(
                    x_t, timesteps, torch.zeros_like(text_embeds), text_mask
                )
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
            # Use evenly spaced steps for DDIM
            step_indices = torch.linspace(0, self.num_timesteps - 1, num_steps + 1)
            step_indices = step_indices.long()
            timesteps = torch.zeros(num_steps, dtype=torch.long, device=device)

            for i, idx in enumerate(reversed(step_indices[:-1])):
                if idx > 0:
                    timesteps[i] = idx
                else:
                    timesteps[i] = self.num_timesteps - 1
        else:
            # Use all timesteps for DDPM
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1)

        # Sampling loop
        for i, t in enumerate(tqdm.tqdm(reversed(timesteps), desc="Sampling")):
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

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate training loss with CFG Dropout."""
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, timesteps, noise)

        # Apply CFG dropout
        if self.cfg_probability > 0:
            batch_size = x_0.shape[0]
            drop_mask = torch.rand(batch_size, device=x_0.device) < self.cfg_probability

            # (수정) 임베딩을 0으로 만듭니다. (이상적으로는 'Empty String' 임베딩이 좋습니다)
            drop_mask_expanded = drop_mask.view(batch_size, 1, 1)
            text_embeds = text_embeds.clone()  # 원본 보존을 위해 clone
            text_embeds = text_embeds * (~drop_mask_expanded).float()

            # (중요 수정) Mask를 0으로 만들면 Attention에서 NaN이 뜰 수 있습니다.
            # 1. 보통 Unconditional Embedding도 길이는 있으므로 마스크를 건드리지 않거나
            # 2. 혹은 마스크를 처리한다면 최소 1개 토큰은 True로 남겨야 합니다.
            # 여기서는 '임베딩이 0'이 되었으므로, 모델이 이를 'Null'로 인식하도록 학습시킵니다.
            # 따라서 text_mask는 건드리지 않는 것이 오히려 안전합니다. (Zero Vector를 '보게' 만듦)

            # 만약 꼭 마스크 처리를 해야 한다면, 아래 코드는 삭제하거나 주의해야 합니다.
            # if text_mask is not None:
            #     drop_mask_for_mask = drop_mask.view(batch_size, 1)
            #     text_mask = text_mask * (~drop_mask_for_mask).float()

        predicted_noise = model(x_t, timesteps, text_embeds, text_mask)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        return loss

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
