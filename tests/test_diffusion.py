"""Tests for Diffusion process (DDPM/DDIM)."""

import pytest
import torch
import torch.nn as nn

from src.models.diffusion import Diffusion


class SimpleDummyModel(nn.Module):
    """Simple dummy model for testing diffusion."""

    def __init__(self, channels: int = 3, image_size: int = 32):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        # Simple conv layer
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        # Ignore timesteps and text for this simple test
        return self.conv(x)


class TestDiffusion:
    """Test suite for Diffusion class."""

    @pytest.fixture
    def diffusion(self) -> Diffusion:
        """Create a Diffusion instance for testing."""
        return Diffusion(
            num_timesteps=1000,
            beta_schedule="cosine",
            guidance_scale=7.5,
            cfg_probability=0.1,
        )

    @pytest.fixture
    def dummy_model(self) -> SimpleDummyModel:
        """Create a dummy model for testing."""
        return SimpleDummyModel()

    def test_initialization(self, diffusion: Diffusion) -> None:
        """Test diffusion initialization."""
        assert diffusion.num_timesteps == 1000
        assert diffusion.guidance_scale == 7.5
        assert diffusion.betas.shape == (1000,)
        assert diffusion.alphas_cumprod.shape == (1000,)

    def test_beta_schedule_values(self, diffusion: Diffusion) -> None:
        """Test beta schedule has valid values."""
        assert (diffusion.betas >= 0).all()
        assert (diffusion.betas <= 1).all()
        assert (diffusion.alphas_cumprod >= 0).all()
        assert (diffusion.alphas_cumprod <= 1).all()

    def test_q_sample(self, diffusion: Diffusion) -> None:
        """Test forward diffusion process."""
        batch_size = 2
        x_0 = torch.randn(batch_size, 3, 32, 32)
        timesteps = torch.randint(0, 1000, (batch_size,))

        x_t = diffusion.q_sample(x_0, timesteps)

        assert x_t.shape == x_0.shape
        assert not torch.isnan(x_t).any()

    def test_q_sample_with_noise(self, diffusion: Diffusion) -> None:
        """Test forward diffusion with specific noise."""
        x_0 = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(x_0)
        timesteps = torch.tensor([100, 500])

        x_t = diffusion.q_sample(x_0, timesteps, noise)

        assert x_t.shape == x_0.shape

    def test_q_sample_t0_close_to_original(self, diffusion: Diffusion) -> None:
        """Test that t=0 gives result close to original."""
        x_0 = torch.randn(1, 3, 32, 32)
        timesteps = torch.tensor([0])

        x_t = diffusion.q_sample(x_0, timesteps)

        # At t=0, alpha_cumprod should be close to 1
        # so x_t should be close to x_0
        assert torch.allclose(x_t, x_0, atol=0.1)

    def test_training_loss(
        self, diffusion: Diffusion, dummy_model: SimpleDummyModel
    ) -> None:
        """Test training loss computation."""
        x_0 = torch.randn(2, 3, 32, 32)
        timesteps = torch.randint(0, 1000, (2,))
        text_embeds = torch.randn(2, 77, 512)

        loss = diffusion.training_loss(dummy_model, x_0, timesteps, text_embeds)

        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_training_loss_gradient(
        self, diffusion: Diffusion, dummy_model: SimpleDummyModel
    ) -> None:
        """Test that training loss is differentiable."""
        x_0 = torch.randn(2, 3, 32, 32)
        timesteps = torch.randint(0, 1000, (2,))
        text_embeds = torch.randn(2, 77, 512)

        loss = diffusion.training_loss(dummy_model, x_0, timesteps, text_embeds)
        loss.backward()

        # Check gradients exist
        for param in dummy_model.parameters():
            assert param.grad is not None

    def test_different_beta_schedules(self) -> None:
        """Test different beta schedule types."""
        for schedule in ["linear", "cosine", "quadratic"]:
            diffusion = Diffusion(
                num_timesteps=100,
                beta_schedule=schedule,
            )
            assert diffusion.betas.shape == (100,)
            assert (diffusion.betas >= 0).all()

    def test_repr(self, diffusion: Diffusion) -> None:
        """Test string representation."""
        repr_str = repr(diffusion)
        assert "Diffusion" in repr_str
        assert "num_timesteps=1000" in repr_str


class TestDiffusionNumericalStability:
    """Test numerical stability of diffusion process."""

    @pytest.fixture
    def diffusion(self) -> Diffusion:
        return Diffusion(num_timesteps=1000, beta_schedule="cosine")

    def test_no_nan_in_coefficients(self, diffusion: Diffusion) -> None:
        """Test that all precomputed coefficients are valid."""
        assert not torch.isnan(diffusion.betas).any()
        assert not torch.isnan(diffusion.alphas_cumprod).any()
        assert not torch.isnan(diffusion.sqrt_alphas_cumprod).any()
        assert not torch.isnan(diffusion.sqrt_one_minus_alphas_cumprod).any()
        assert not torch.isnan(diffusion.posterior_mean_coef1).any()
        assert not torch.isnan(diffusion.posterior_mean_coef2).any()

    def test_no_inf_in_coefficients(self, diffusion: Diffusion) -> None:
        """Test that no coefficients are infinite."""
        assert not torch.isinf(diffusion.betas).any()
        assert not torch.isinf(diffusion.alphas_cumprod).any()
        assert not torch.isinf(diffusion.sqrt_recip_alphas_cumprod).any()
        assert not torch.isinf(diffusion.sqrt_recip_alphas_cumprod_minus_1).any()
