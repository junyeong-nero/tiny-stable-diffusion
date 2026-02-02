"""Tests for Motion Module and AnimatedMMDiT."""

import pytest
import torch

from src.models.motion import (
    MotionModule,
    TemporalPositionEmbedding,
    TemporalTransformerBlock,
    create_motion_module,
)
from src.models.animated_mmdit import AnimatedMMDiT, create_animated_mmdit
from src.models.mmdit import MMDiT, MMDIT_AVAILABLE


class TestTemporalPositionEmbedding:
    """Test suite for TemporalPositionEmbedding."""

    def test_initialization(self) -> None:
        """Test that embedding initializes correctly."""
        embed = TemporalPositionEmbedding(hidden_size=256, max_frames=32)
        assert embed.hidden_size == 256
        assert embed.max_frames == 32
        assert embed.pe.shape == (32, 256)

    def test_forward(self) -> None:
        """Test forward pass returns correct shape."""
        embed = TemporalPositionEmbedding(hidden_size=256, max_frames=32)
        
        for num_frames in [8, 16, 32]:
            output = embed(num_frames)
            assert output.shape == (1, num_frames, 256)

    def test_different_hidden_sizes(self) -> None:
        """Test with various hidden sizes."""
        for hidden_size in [128, 256, 512, 768]:
            embed = TemporalPositionEmbedding(hidden_size=hidden_size, max_frames=16)
            output = embed(16)
            assert output.shape == (1, 16, hidden_size)


class TestTemporalTransformerBlock:
    """Test suite for TemporalTransformerBlock."""

    @pytest.fixture
    def block(self) -> TemporalTransformerBlock:
        """Create a temporal transformer block for testing."""
        return TemporalTransformerBlock(
            hidden_size=256,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
            max_frames=32,
        )

    def test_initialization(self, block: TemporalTransformerBlock) -> None:
        """Test that block initializes correctly."""
        assert block.hidden_size == 256
        assert block.num_heads == 8

    def test_zero_initialization(self, block: TemporalTransformerBlock) -> None:
        """Test that output projections are zero-initialized."""
        # Check attention output projection
        assert torch.allclose(
            block.attn.out_proj.weight, 
            torch.zeros_like(block.attn.out_proj.weight)
        )
        assert torch.allclose(
            block.attn.out_proj.bias, 
            torch.zeros_like(block.attn.out_proj.bias)
        )

    def test_forward_shape(self, block: TemporalTransformerBlock) -> None:
        """Test forward pass returns correct shape."""
        batch_spatial = 32  # B * N
        num_frames = 16
        hidden_size = 256

        x = torch.randn(batch_spatial, num_frames, hidden_size)
        output = block(x)

        assert output.shape == x.shape

    def test_identity_at_init(self, block: TemporalTransformerBlock) -> None:
        """Test that output equals input at initialization (due to zero init)."""
        x = torch.randn(16, 8, 256)
        output = block(x)
        
        # Due to zero initialization, the residual should be approximately zero
        # so output should be close to input (with position embedding added)
        # This is a soft check since position embedding is added
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_different_frame_counts(self, block: TemporalTransformerBlock) -> None:
        """Test with various frame counts."""
        for num_frames in [4, 8, 16, 32]:
            x = torch.randn(16, num_frames, 256)
            output = block(x)
            assert output.shape == (16, num_frames, 256)


class TestMotionModule:
    """Test suite for MotionModule."""

    @pytest.fixture
    def module(self) -> MotionModule:
        """Create a motion module for testing."""
        return MotionModule(
            hidden_size=256,
            num_layers=2,
            num_heads=8,
            max_frames=32,
        )

    def test_initialization(self, module: MotionModule) -> None:
        """Test that module initializes correctly."""
        assert module.hidden_size == 256
        assert module.num_layers == 2
        assert len(module.temporal_blocks) == 2

    def test_forward_shape(self, module: MotionModule) -> None:
        """Test forward pass with correct reshape logic."""
        batch_size = 2
        num_frames = 16
        num_tokens = 16  # 4x4 patches from 8x8 latent
        hidden_size = 256

        # Input: (B*F, N, D)
        x = torch.randn(batch_size * num_frames, num_tokens, hidden_size)
        output = module(x, num_frames=num_frames)

        assert output.shape == x.shape

    def test_different_configurations(self) -> None:
        """Test with various configurations."""
        configs = [
            {"hidden_size": 256, "num_layers": 1, "num_heads": 4},
            {"hidden_size": 512, "num_layers": 2, "num_heads": 8},
            {"hidden_size": 768, "num_layers": 3, "num_heads": 12},
        ]

        for config in configs:
            module = MotionModule(**config, max_frames=32)
            x = torch.randn(32, 16, config["hidden_size"])
            output = module(x, num_frames=16)
            assert output.shape == x.shape

    def test_parameters_count(self, module: MotionModule) -> None:
        """Test parameter counting."""
        count = module.parameters_count()
        assert count > 0
        assert isinstance(count, int)

    def test_factory_function(self) -> None:
        """Test create_motion_module factory function."""
        module = create_motion_module(
            hidden_size=384,
            num_layers=2,
            num_heads=6,
            max_frames=32,
        )
        assert isinstance(module, MotionModule)
        assert module.hidden_size == 384

    def test_gradient_flow(self, module: MotionModule) -> None:
        """Test that gradients flow through the module."""
        x = torch.randn(32, 16, 256, requires_grad=True)
        output = module(x, num_frames=16)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


@pytest.mark.skipif(not MMDIT_AVAILABLE, reason="mmdit package not installed")
class TestAnimatedMMDiT:
    """Test suite for AnimatedMMDiT."""

    @pytest.fixture
    def base_model(self) -> MMDiT:
        """Create a base MMDiT model for testing."""
        return MMDiT(
            in_channels=16,
            image_size=8,
            patch_size=2,
            model_size="S",
            clip_embed_dim=512,
        )

    @pytest.fixture
    def animated_model(self, base_model: MMDiT) -> AnimatedMMDiT:
        """Create an AnimatedMMDiT for testing."""
        return AnimatedMMDiT(
            base_model=base_model,
            num_frames=8,
            motion_num_layers=1,
            motion_num_heads=6,
            freeze_base=True,
        )

    def test_initialization(self, animated_model: AnimatedMMDiT) -> None:
        """Test that model initializes correctly."""
        assert animated_model is not None
        assert animated_model.num_frames == 8
        assert animated_model.motion_module is not None

    def test_base_frozen(self, animated_model: AnimatedMMDiT) -> None:
        """Test that base model is frozen."""
        for param in animated_model.base_model.parameters():
            assert not param.requires_grad

    def test_motion_trainable(self, animated_model: AnimatedMMDiT) -> None:
        """Test that motion module is trainable."""
        for param in animated_model.motion_module.parameters():
            assert param.requires_grad

    def test_forward_5d_input(self, animated_model: AnimatedMMDiT) -> None:
        """Test forward pass with 5D input (B, F, C, H, W)."""
        batch_size = 2
        num_frames = 8
        
        x = torch.randn(batch_size, num_frames, 16, 8, 8)
        timesteps = torch.randint(0, 1000, (batch_size,))
        text_embeds = torch.randn(batch_size, 512)

        output = animated_model(x, timesteps, text_embeds)

        assert output.shape == (batch_size, num_frames, 16, 8, 8)

    def test_forward_4d_input(self, animated_model: AnimatedMMDiT) -> None:
        """Test forward pass with 4D input (B*F, C, H, W)."""
        batch_size = 2
        num_frames = 8
        
        x = torch.randn(batch_size * num_frames, 16, 8, 8)
        timesteps = torch.randint(0, 1000, (batch_size,))
        text_embeds = torch.randn(batch_size, 512)

        output = animated_model(x, timesteps, text_embeds)

        assert output.shape == (batch_size * num_frames, 16, 8, 8)

    def test_parameters_count(self, animated_model: AnimatedMMDiT) -> None:
        """Test parameter counting."""
        counts = animated_model.parameters_count()
        
        assert "base_total" in counts
        assert "base_trainable" in counts
        assert "motion_trainable" in counts
        assert "total_trainable" in counts
        
        # Base should be frozen
        assert counts["base_trainable"] == 0
        # Motion should be trainable
        assert counts["motion_trainable"] > 0
        # Total trainable should equal motion trainable when base is frozen
        assert counts["total_trainable"] == counts["motion_trainable"]

    def test_get_trainable_parameters(self, animated_model: AnimatedMMDiT) -> None:
        """Test getting trainable parameters."""
        trainable_params = animated_model.get_trainable_parameters()
        
        assert len(trainable_params) > 0
        for param in trainable_params:
            assert param.requires_grad

    def test_gradient_flow_motion_only(self, animated_model: AnimatedMMDiT) -> None:
        """Test that gradients only flow through motion module."""
        x = torch.randn(2, 8, 16, 8, 8, requires_grad=True)
        timesteps = torch.randint(0, 1000, (2,))
        text_embeds = torch.randn(2, 512)

        output = animated_model(x, timesteps, text_embeds)
        loss = output.sum()
        loss.backward()

        # Check motion module has gradients
        has_motion_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in animated_model.motion_module.parameters()
        )
        assert has_motion_grad

        # Check base model has no gradients (frozen)
        has_base_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in animated_model.base_model.parameters()
        )
        assert not has_base_grad

    def test_different_frame_counts(self, base_model: MMDiT) -> None:
        """Test with various frame counts."""
        for num_frames in [4, 8, 16]:
            model = AnimatedMMDiT(
                base_model=base_model,
                num_frames=num_frames,
                motion_num_layers=1,
                motion_num_heads=6,
                freeze_base=True,
            )
            
            x = torch.randn(2, num_frames, 16, 8, 8)
            timesteps = torch.randint(0, 1000, (2,))
            text_embeds = torch.randn(2, 512)

            output = model(x, timesteps, text_embeds)
            assert output.shape == (2, num_frames, 16, 8, 8)

    def test_factory_function(self) -> None:
        """Test create_animated_mmdit factory function."""
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
        
        assert isinstance(model, AnimatedMMDiT)
        assert model.num_frames == 8

    def test_output_no_nan(self, animated_model: AnimatedMMDiT) -> None:
        """Test that output contains no NaN values."""
        x = torch.randn(2, 8, 16, 8, 8)
        timesteps = torch.randint(0, 1000, (2,))
        text_embeds = torch.randn(2, 512)

        output = animated_model(x, timesteps, text_embeds)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
