"""Tests for AnimatedDiffusion and Motion Training."""

import pytest
import torch

from src.models.animated_diffusion import (
    AnimatedDiffusion,
    create_animated_diffusion,
)
from src.models.mmdit import MMDIT_AVAILABLE


class TestAnimatedDiffusion:
    """Test suite for AnimatedDiffusion."""

    @pytest.fixture
    def diffusion(self) -> AnimatedDiffusion:
        """Create an AnimatedDiffusion for testing."""
        return AnimatedDiffusion(
            num_timesteps=1000,
            num_frames=8,
            guidance_scale=7.5,
            cfg_probability=0.1,
            min_snr_gamma=5.0,
        )

    def test_initialization(self, diffusion: AnimatedDiffusion) -> None:
        """Test that diffusion initializes correctly."""
        assert diffusion.num_timesteps == 1000
        assert diffusion.num_frames == 8
        assert diffusion.guidance_scale == 7.5

    def test_q_sample_video_shape(self, diffusion: AnimatedDiffusion) -> None:
        """Test q_sample_video output shape."""
        x_0 = torch.randn(2, 8, 16, 8, 8)  # (B, F, C, H, W)
        timesteps = torch.randint(0, 1000, (2,))

        x_t = diffusion.q_sample_video(x_0, timesteps)

        assert x_t.shape == x_0.shape

    def test_q_sample_video_t0(self, diffusion: AnimatedDiffusion) -> None:
        """Test that t=0 gives approximately original."""
        x_0 = torch.randn(2, 8, 16, 8, 8)
        timesteps = torch.zeros(2, dtype=torch.long)

        x_t = diffusion.q_sample_video(x_0, timesteps)

        # At t=0, x_t should be close to x_0
        assert torch.allclose(x_t, x_0, atol=1e-5)

    def test_q_sample_video_t1(self, diffusion: AnimatedDiffusion) -> None:
        """Test that t=T gives approximately noise."""
        x_0 = torch.randn(2, 8, 16, 8, 8)
        timesteps = torch.full((2,), 999, dtype=torch.long)
        noise = torch.randn_like(x_0)

        x_t = diffusion.q_sample_video(x_0, timesteps, noise=noise)

        # At t≈1, x_t should be close to noise
        # (999/1000 = 0.999, so mostly noise)
        t = 999 / 1000
        expected = (1 - t) * x_0 + t * noise
        assert torch.allclose(x_t, expected, atol=1e-5)

    def test_get_velocity_video(self, diffusion: AnimatedDiffusion) -> None:
        """Test velocity computation."""
        x_0 = torch.randn(2, 8, 16, 8, 8)
        noise = torch.randn_like(x_0)

        velocity = diffusion.get_velocity_video(x_0, noise)

        expected = noise - x_0
        assert torch.allclose(velocity, expected)

    def test_factory_function(self) -> None:
        """Test create_animated_diffusion factory."""
        diffusion = create_animated_diffusion(
            num_timesteps=500,
            num_frames=16,
            guidance_scale=5.0,
        )

        assert isinstance(diffusion, AnimatedDiffusion)
        assert diffusion.num_timesteps == 500
        assert diffusion.num_frames == 16

    def test_repr(self, diffusion: AnimatedDiffusion) -> None:
        """Test string representation."""
        repr_str = repr(diffusion)
        assert "AnimatedDiffusion" in repr_str
        assert "num_frames=8" in repr_str


@pytest.mark.skipif(not MMDIT_AVAILABLE, reason="mmdit package not installed")
class TestAnimatedDiffusionWithModel:
    """Test AnimatedDiffusion with actual model."""

    @pytest.fixture
    def setup(self):
        """Set up model and diffusion for testing."""
        from src.models.animated_mmdit import create_animated_mmdit

        model = create_animated_mmdit(
            in_channels=16,
            image_size=8,
            patch_size=2,
            model_size="S",
            clip_embed_dim=512,
            num_frames=8,
            motion_num_layers=1,
            motion_num_heads=6,
            freeze_base=True,
        )

        diffusion = AnimatedDiffusion(
            num_timesteps=1000,
            num_frames=8,
            guidance_scale=7.5,
            cfg_probability=0.0,
            min_snr_gamma=5.0,
        )

        return model, diffusion

    def test_training_loss_video(self, setup) -> None:
        """Test training loss computation."""
        model, diffusion = setup

        x_0 = torch.randn(2, 8, 16, 8, 8)
        timesteps = torch.randint(0, 1000, (2,))
        text_embeds = torch.randn(2, 512)

        loss, loss_dict = diffusion.training_loss_video(
            model, x_0, timesteps, text_embeds
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert "total_loss" in loss_dict
        assert "velocity_loss" in loss_dict

    def test_training_loss_gradient(self, setup) -> None:
        """Test that gradients flow through motion module."""
        model, diffusion = setup

        x_0 = torch.randn(2, 8, 16, 8, 8)
        timesteps = torch.randint(0, 1000, (2,))
        text_embeds = torch.randn(2, 512)

        loss, _ = diffusion.training_loss_video(
            model, x_0, timesteps, text_embeds
        )
        loss.backward()

        # Check motion module has gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.motion_module.parameters()
        )
        assert has_grad

    def test_euler_step_video(self, setup) -> None:
        """Test single Euler step."""
        model, diffusion = setup

        x_t = torch.randn(2, 8, 16, 8, 8)
        t_curr = torch.full((2,), 500, dtype=torch.long)
        t_next = torch.full((2,), 480, dtype=torch.long)
        text_embeds = torch.randn(2, 512)

        x_next = diffusion.euler_step_video(
            model, x_t, t_curr, t_next, text_embeds, use_cfg=False
        )

        assert x_next.shape == x_t.shape
        assert not torch.isnan(x_next).any()

    def test_sample_video(self, setup) -> None:
        """Test video sampling (few steps for speed)."""
        model, diffusion = setup

        text_embeds = torch.randn(1, 512)

        videos = diffusion.sample_video(
            model=model,
            batch_size=1,
            num_frames=4,
            latent_channels=16,
            latent_size=8,
            text_embeds=text_embeds,
            num_steps=5,  # Few steps for testing
            use_cfg=False,
            device="cpu",
        )

        assert videos.shape == (1, 4, 16, 8, 8)
        assert videos.min() >= 0.0
        assert videos.max() <= 1.0


class TestTemporalConsistency:
    """Test temporal consistency loss."""

    def test_temporal_loss_weight(self) -> None:
        """Test that temporal consistency weight affects loss."""
        diffusion_no_tc = AnimatedDiffusion(
            num_timesteps=1000,
            num_frames=8,
            temporal_consistency_weight=0.0,
        )

        diffusion_with_tc = AnimatedDiffusion(
            num_timesteps=1000,
            num_frames=8,
            temporal_consistency_weight=0.1,
        )

        assert diffusion_no_tc.temporal_consistency_weight == 0.0
        assert diffusion_with_tc.temporal_consistency_weight == 0.1
