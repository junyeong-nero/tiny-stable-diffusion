"""Tests for DiT (Diffusion Transformer) model."""

import pytest
import torch

from src.models.dit import DiT


class TestDiT:
    """Test suite for DiT model."""

    @pytest.fixture
    def model_small(self) -> DiT:
        """Create a small DiT model for testing."""
        return DiT(
            in_channels=3,
            image_size=32,
            patch_size=2,
            model_size="S",
            clip_embed_dim=512,
        )

    def test_model_initialization(self, model_small: DiT) -> None:
        """Test that model initializes correctly."""
        assert model_small is not None
        info = model_small.get_model_size_info()
        assert info["num_parameters"] > 0
        assert info["model_size"] == "S"

    def test_forward_pass(self, model_small: DiT) -> None:
        """Test forward pass with random inputs."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        timesteps = torch.randint(0, 1000, (batch_size,))
        text_embeds = torch.randn(batch_size, 77, 512)

        output = model_small(x, timesteps, text_embeds)

        assert output.shape == (batch_size, 3, 32, 32)

    def test_output_dtype(self, model_small: DiT) -> None:
        """Test that output dtype matches input dtype."""
        x = torch.randn(1, 3, 32, 32)
        timesteps = torch.tensor([500])
        text_embeds = torch.randn(1, 77, 512)

        output = model_small(x, timesteps, text_embeds)

        assert output.dtype == x.dtype

    def test_different_batch_sizes(self, model_small: DiT) -> None:
        """Test model with different batch sizes."""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 32, 32)
            timesteps = torch.randint(0, 1000, (batch_size,))
            text_embeds = torch.randn(batch_size, 77, 512)

            output = model_small(x, timesteps, text_embeds)

            assert output.shape == (batch_size, 3, 32, 32)

    def test_all_model_sizes(self) -> None:
        """Test all model size configurations."""
        for size in ["S", "B"]:  # Skip L and XL for speed
            model = DiT(
                in_channels=3,
                image_size=32,
                patch_size=2,
                model_size=size,
                clip_embed_dim=512,
            )

            x = torch.randn(1, 3, 32, 32)
            timesteps = torch.tensor([500])
            text_embeds = torch.randn(1, 77, 512)

            output = model(x, timesteps, text_embeds)
            assert output.shape == (1, 3, 32, 32)

    def test_timestep_range(self, model_small: DiT) -> None:
        """Test model with various timestep values."""
        x = torch.randn(1, 3, 32, 32)
        text_embeds = torch.randn(1, 77, 512)

        for t in [0, 100, 500, 999]:
            timesteps = torch.tensor([t])
            output = model_small(x, timesteps, text_embeds)
            assert output.shape == (1, 3, 32, 32)
            assert not torch.isnan(output).any()

    def test_gradient_flow(self, model_small: DiT) -> None:
        """Test that gradients flow through the model."""
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        timesteps = torch.randint(0, 1000, (2,))
        text_embeds = torch.randn(2, 77, 512)

        output = model_small(x, timesteps, text_embeds)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
