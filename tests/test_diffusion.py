"""Tests for Rectified Flow diffusion process."""

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
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Ignore timesteps and text for this simple test
        return self.conv(x)


class TestDiffusion:
    """Test suite for Rectified Flow Diffusion class."""

    @pytest.fixture
    def diffusion(self) -> Diffusion:
        """Create a Diffusion instance for testing."""
        return Diffusion(
            num_timesteps=1000,
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
        assert diffusion.cfg_probability == 0.1

    def test_q_sample_linear_interpolation(self, diffusion: Diffusion) -> None:
        """Test forward diffusion process uses linear interpolation."""
        batch_size = 2
        x_0 = torch.randn(batch_size, 3, 32, 32)
        noise = torch.randn_like(x_0)

        # Test at t=0: x_t should equal x_0
        timesteps_0 = torch.tensor([0, 0])
        x_t_0 = diffusion.q_sample(x_0, timesteps_0, noise)
        assert torch.allclose(x_t_0, x_0, atol=1e-5)

        # Test at t=1000: x_t should equal noise
        timesteps_1000 = torch.tensor([1000, 1000])
        x_t_1000 = diffusion.q_sample(x_0, timesteps_1000, noise)
        assert torch.allclose(x_t_1000, noise, atol=1e-5)

        # Test at t=500: x_t should be midpoint
        timesteps_500 = torch.tensor([500, 500])
        x_t_500 = diffusion.q_sample(x_0, timesteps_500, noise)
        expected = 0.5 * x_0 + 0.5 * noise
        assert torch.allclose(x_t_500, expected, atol=1e-5)

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
        """Test that t=0 gives result equal to original."""
        x_0 = torch.randn(1, 3, 32, 32)
        timesteps = torch.tensor([0])

        x_t = diffusion.q_sample(x_0, timesteps)

        # At t=0, x_t should equal x_0 (linear interpolation)
        assert torch.allclose(x_t, x_0, atol=1e-5)

    def test_get_velocity(self, diffusion: Diffusion) -> None:
        """Test velocity computation."""
        x_0 = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(x_0)

        velocity = diffusion.get_velocity(x_0, noise)

        # Velocity should be noise - x_0
        expected = noise - x_0
        assert torch.allclose(velocity, expected, atol=1e-6)

    def test_training_loss(self, diffusion: Diffusion, dummy_model: SimpleDummyModel) -> None:
        """Test training loss computation."""
        x_0 = torch.randn(2, 3, 32, 32)
        timesteps = torch.randint(0, 1000, (2,))
        text_embeds = torch.randn(2, 512)

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
        text_embeds = torch.randn(2, 512)

        loss = diffusion.training_loss(dummy_model, x_0, timesteps, text_embeds)
        loss.backward()

        # Check gradients exist
        for param in dummy_model.parameters():
            assert param.grad is not None

    def test_logit_normal_sampling(self, diffusion: Diffusion) -> None:
        """Test logit-normal timestep sampling."""
        batch_size = 1000
        timesteps = diffusion.sample_timesteps_logit_normal(
            batch_size=batch_size,
            device=torch.device("cpu"),
        )

        assert timesteps.shape == (batch_size,)
        assert timesteps.dtype == torch.long
        assert (timesteps >= 0).all()
        assert (timesteps < 1000).all()

        # Logit-normal should concentrate around middle (mean ~500)
        mean = timesteps.float().mean().item()
        assert 400 < mean < 600  # Should be around 500

    def test_repr(self, diffusion: Diffusion) -> None:
        """Test string representation."""
        repr_str = repr(diffusion)
        assert "Diffusion" in repr_str
        assert "num_timesteps=1000" in repr_str
        assert "RectifiedFlow" in repr_str


class TestDiffusionNumericalStability:
    """Test numerical stability of Rectified Flow diffusion process."""

    @pytest.fixture
    def diffusion(self) -> Diffusion:
        return Diffusion(num_timesteps=1000)

    def test_q_sample_no_nan(self, diffusion: Diffusion) -> None:
        """Test that q_sample produces no NaN values."""
        x_0 = torch.randn(4, 3, 32, 32)

        for t in [0, 1, 100, 500, 999, 1000]:
            timesteps = torch.full((4,), t)
            x_t = diffusion.q_sample(x_0, timesteps)
            assert not torch.isnan(x_t).any(), f"NaN at timestep {t}"
            assert not torch.isinf(x_t).any(), f"Inf at timestep {t}"

    def test_velocity_no_nan(self, diffusion: Diffusion) -> None:
        """Test that velocity computation produces no NaN values."""
        x_0 = torch.randn(4, 3, 32, 32)
        noise = torch.randn_like(x_0)

        velocity = diffusion.get_velocity(x_0, noise)

        assert not torch.isnan(velocity).any()
        assert not torch.isinf(velocity).any()

    def test_extreme_timesteps(self, diffusion: Diffusion) -> None:
        """Test behavior at extreme timesteps."""
        x_0 = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(x_0)

        # t=0 should give x_0
        x_t_0 = diffusion.q_sample(x_0, torch.tensor([0, 0]), noise)
        assert torch.allclose(x_t_0, x_0, atol=1e-5)

        # t=num_timesteps should give noise
        x_t_max = diffusion.q_sample(
            x_0, torch.tensor([diffusion.num_timesteps, diffusion.num_timesteps]), noise
        )
        assert torch.allclose(x_t_max, noise, atol=1e-5)
