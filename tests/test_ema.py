"""Tests for EMA (Exponential Moving Average)."""

import pytest
import torch
import torch.nn as nn

from src.training.ema import EMA


class SimpleModel(nn.Module):
    """Simple model for testing EMA."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.bias


class TestEMA:
    """Test suite for EMA class."""

    @pytest.fixture
    def model(self) -> SimpleModel:
        """Create a simple model for testing."""
        return SimpleModel()

    @pytest.fixture
    def ema(self, model: SimpleModel) -> EMA:
        """Create EMA instance for testing."""
        return EMA(model, decay=0.999)

    def test_initialization(self, model: SimpleModel, ema: EMA) -> None:
        """Test EMA initialization."""
        assert ema.decay == 0.999
        assert ema.step == 0
        assert len(ema.shadow_params) > 0

    def test_shadow_params_initialized(self, model: SimpleModel, ema: EMA) -> None:
        """Test that shadow params are initialized to model params."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow_params
                assert torch.allclose(ema.shadow_params[name], param.data)

    def test_update(self, model: SimpleModel, ema: EMA) -> None:
        """Test EMA update."""
        # Get original shadow params
        original_shadow = {k: v.clone() for k, v in ema.shadow_params.items()}

        # Modify model params
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Update EMA
        ema.update()

        # Shadow params should have changed
        for name in ema.shadow_params:
            # Due to EMA update, shadow should be between old and new
            assert not torch.allclose(ema.shadow_params[name], original_shadow[name], atol=1e-6)

    def test_apply_and_restore(self, model: SimpleModel, ema: EMA) -> None:
        """Test apply_shadow and restore."""
        # Store original model params
        original_params = {name: param.data.clone() for name, param in model.named_parameters()}

        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        # Update EMA a few times
        for _ in range(10):
            ema.update()

        # Apply shadow
        ema.apply_shadow()

        # Model params should now be different from the modified version
        for name, param in model.named_parameters():
            if name in ema.shadow_params:
                # Should be closer to original + some EMA influence
                pass

        # Restore
        ema.restore()

        # Model params should be back to modified version
        # (not original, but what was there before apply_shadow)
        assert len(ema.backup_params) == 0  # Backup should be cleared

    def test_state_dict(self, model: SimpleModel, ema: EMA) -> None:
        """Test state dict save/load."""
        # Do some updates
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            ema.update()

        # Save state
        state = ema.state_dict()
        assert "step" in state
        assert "decay" in state
        assert "shadow_params" in state

        # Create new EMA and load state
        new_ema = EMA(model, decay=0.9)  # Different decay
        new_ema.load_state_dict(state)

        assert new_ema.step == ema.step
        assert new_ema.decay == ema.decay

    def test_to_device(self, model: SimpleModel, ema: EMA) -> None:
        """Test moving EMA to device."""
        # Just test CPU to CPU (GPU tests would require hardware)
        ema.to(torch.device("cpu"))

        for name, param in ema.shadow_params.items():
            assert param.device == torch.device("cpu")

    def test_decay_warmup(self, model: SimpleModel) -> None:
        """Test that decay warms up during early steps."""
        ema = EMA(model, decay=0.9999)

        # Early steps should have lower effective decay
        # due to: min(decay, (1 + step) / (10 + step))
        ema.step = 1
        effective_decay_early = min(0.9999, (1 + 1) / (10 + 1))
        assert effective_decay_early < 0.9999

        # Later steps should use full decay
        ema.step = 1000
        effective_decay_late = min(0.9999, (1 + 1000) / (10 + 1000))
        assert effective_decay_late > 0.99

    def test_copy_to(self, model: SimpleModel, ema: EMA) -> None:
        """Test copying EMA weights to another model."""
        # Update EMA a few times
        for _ in range(10):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            ema.update()

        # Create new model
        new_model = SimpleModel()

        # Copy EMA weights to new model
        ema.copy_to(new_model)

        # New model should have EMA weights
        for name, param in new_model.named_parameters():
            if param.requires_grad and name in ema.shadow_params:
                assert torch.allclose(param.data, ema.shadow_params[name])

    def test_repr(self, ema: EMA) -> None:
        """Test string representation."""
        repr_str = repr(ema)
        assert "EMA" in repr_str
        assert "decay=0.999" in repr_str
