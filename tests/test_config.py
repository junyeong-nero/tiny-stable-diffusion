"""Tests for configuration module."""

import pytest

from src.config import (
    DataConfig,
    DiffusionConfig,
    InferenceConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    get_default_configs,
)


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelConfig()
        assert config.model_size == "S"
        assert config.patch_size == 2
        assert config.image_size == 32
        assert config.in_channels == 3

    def test_model_size_s(self) -> None:
        """Test DiT-S configuration."""
        config = ModelConfig(model_size="S")
        assert config.num_layers == 12
        assert config.num_heads == 6
        assert config.hidden_size == 384

    def test_model_size_b(self) -> None:
        """Test DiT-B configuration."""
        config = ModelConfig(model_size="B")
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.hidden_size == 768

    def test_model_size_l(self) -> None:
        """Test DiT-L configuration."""
        config = ModelConfig(model_size="L")
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.hidden_size == 1024


class TestDiffusionConfig:
    """Test suite for DiffusionConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DiffusionConfig()
        assert config.num_timesteps == 1000
        assert config.beta_schedule == "cosine"
        assert config.guidance_scale == 7.5
        assert config.cfg_probability == 0.1


class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.batch_size == 64
        assert config.epochs == 100
        assert config.learning_rate == 5e-4
        assert config.warmup_steps == 100
        assert config.weight_decay == 0.01

    def test_optimizer_settings(self) -> None:
        """Test optimizer configuration."""
        config = TrainingConfig()
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.gradient_clip_val == 1.0


class TestDataConfig:
    """Test suite for DataConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DataConfig()
        assert config.source == "huggingface"
        assert config.dataset_name == "junyeong-nero/emoji-32"
        assert config.image_size == 32
        assert config.use_augmentation is True


class TestInferenceConfig:
    """Test suite for InferenceConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = InferenceConfig()
        assert config.num_samples == 4
        assert config.num_steps == 50
        assert config.use_ddim is True
        assert config.guidance_scale == 7.5


class TestProjectConfig:
    """Test suite for ProjectConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ProjectConfig()
        assert config.name == "text-to-emoji-diffusion"
        assert config.seed == 42


class TestGetDefaultConfigs:
    """Test suite for get_default_configs function."""

    def test_default_config(self) -> None:
        """Test getting default configuration."""
        config = get_default_configs()

        assert "model" in config
        assert "diffusion" in config
        assert "training" in config
        assert "data" in config
        assert "inference" in config
        assert "project" in config

        assert isinstance(config["model"], ModelConfig)
        assert isinstance(config["diffusion"], DiffusionConfig)
        assert isinstance(config["training"], TrainingConfig)
