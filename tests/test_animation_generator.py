"""Tests for AnimationGenerator."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image

from src.models.mmdit import MMDIT_AVAILABLE


class TestAnimationGeneratorHelpers:
    """Test helper functions and utilities."""

    def test_imports(self) -> None:
        """Test that animation_generator can be imported."""
        from src.inference.animation_generator import (
            AnimationGenerator,
            generate_animation,
            animation_demo,
        )

        assert AnimationGenerator is not None
        assert generate_animation is not None
        assert animation_demo is not None

    def test_module_exports(self) -> None:
        """Test that animation_generator is exported from inference."""
        from src.inference import (
            AnimationGenerator,
            generate_animation,
            animation_demo,
        )

        assert AnimationGenerator is not None
        assert generate_animation is not None
        assert animation_demo is not None


@pytest.mark.skipif(not MMDIT_AVAILABLE, reason="mmdit package not installed")
class TestAnimationGeneratorMocked:
    """Test AnimationGenerator with mocked checkpoints."""

    @pytest.fixture
    def mock_vae_checkpoint(self, tmp_path: Path) -> Path:
        """Create a mock VAE checkpoint."""
        from src.models.vae import create_vae

        vae = create_vae(image_size=64, z_channels=16)
        checkpoint_path = tmp_path / "vae.pt"
        torch.save(
            {
                "model_state_dict": vae.state_dict(),
                "scaling_factor": 0.18215,
            },
            checkpoint_path,
        )
        return checkpoint_path

    @pytest.fixture
    def mock_diffusion_checkpoint(self, tmp_path: Path) -> Path:
        """Create a mock diffusion checkpoint."""
        from src.models.factory import DiT

        model = DiT(
            in_channels=16,
            image_size=8,
            patch_size=2,
            model_size="S",
            clip_embed_dim=512,
            model_type="mmdit",
        )

        checkpoint_path = tmp_path / "diffusion.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "in_channels": 16,
                    "image_size": 64,
                    "latent_size": 8,
                    "patch_size": 2,
                    "model_size": "S",
                    "model_type": "mmdit",
                    "qk_rmsnorm": True,
                    "register_tokens": 0,
                    "vae_checkpoint": str(tmp_path / "vae.pt"),
                },
            },
            checkpoint_path,
        )
        return checkpoint_path

    @pytest.fixture
    def mock_motion_checkpoint(self, tmp_path: Path) -> Path:
        """Create a mock motion checkpoint."""
        from src.models.motion import MotionModule

        motion = MotionModule(
            hidden_size=384,  # S size
            num_layers=2,
            num_heads=8,
        )

        checkpoint_path = tmp_path / "motion.pt"
        torch.save(
            {
                "motion_module_state_dict": motion.state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_path

    def test_animation_generator_init(
        self,
        mock_vae_checkpoint: Path,
        mock_diffusion_checkpoint: Path,
    ) -> None:
        """Test AnimationGenerator initialization without motion checkpoint."""
        from src.inference.animation_generator import AnimationGenerator

        generator = AnimationGenerator(
            vae_checkpoint=mock_vae_checkpoint,
            diffusion_checkpoint=mock_diffusion_checkpoint,
            motion_checkpoint=None,
            device="cpu",
            num_frames=8,
        )

        assert generator.num_frames == 8
        assert generator.device == torch.device("cpu")
        assert generator.latent_size == 8
        assert generator.latent_channels == 16

    def test_animation_generator_with_motion(
        self,
        mock_vae_checkpoint: Path,
        mock_diffusion_checkpoint: Path,
        mock_motion_checkpoint: Path,
    ) -> None:
        """Test AnimationGenerator initialization with motion checkpoint."""
        from src.inference.animation_generator import AnimationGenerator

        generator = AnimationGenerator(
            vae_checkpoint=mock_vae_checkpoint,
            diffusion_checkpoint=mock_diffusion_checkpoint,
            motion_checkpoint=mock_motion_checkpoint,
            device="cpu",
            num_frames=16,
        )

        assert generator.num_frames == 16

    def test_animation_generator_generate(
        self,
        mock_vae_checkpoint: Path,
        mock_diffusion_checkpoint: Path,
    ) -> None:
        """Test AnimationGenerator.generate() produces frames."""
        from src.inference.animation_generator import AnimationGenerator

        generator = AnimationGenerator(
            vae_checkpoint=mock_vae_checkpoint,
            diffusion_checkpoint=mock_diffusion_checkpoint,
            motion_checkpoint=None,
            device="cpu",
            num_frames=4,
        )

        # Generate with minimal steps for speed
        frames = generator.generate(
            prompt="a cat walking",
            num_frames=4,
            num_steps=2,  # Minimal for testing
            guidance_scale=1.0,  # No CFG for speed
            seed=42,
        )

        assert len(frames) == 4
        assert all(isinstance(f, Image.Image) for f in frames)
        assert all(f.size == (64, 64) for f in frames)

    def test_animation_generator_save_gif(
        self,
        mock_vae_checkpoint: Path,
        mock_diffusion_checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """Test AnimationGenerator.save_gif() creates valid GIF."""
        from src.inference.animation_generator import AnimationGenerator

        generator = AnimationGenerator(
            vae_checkpoint=mock_vae_checkpoint,
            diffusion_checkpoint=mock_diffusion_checkpoint,
            motion_checkpoint=None,
            device="cpu",
            num_frames=4,
        )

        # Create dummy frames
        frames = [Image.new("RGB", (64, 64), color=(i * 50, 0, 0)) for i in range(4)]

        output_path = tmp_path / "test.gif"
        result_path = generator.save_gif(frames, output_path, fps=8)

        assert result_path.exists()
        assert result_path.suffix == ".gif"

        # Verify GIF is valid
        with Image.open(result_path) as img:
            assert img.format == "GIF"
            assert img.n_frames == 4

    def test_animation_generator_generate_and_save(
        self,
        mock_vae_checkpoint: Path,
        mock_diffusion_checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """Test AnimationGenerator.generate_and_save() end-to-end."""
        from src.inference.animation_generator import AnimationGenerator

        generator = AnimationGenerator(
            vae_checkpoint=mock_vae_checkpoint,
            diffusion_checkpoint=mock_diffusion_checkpoint,
            motion_checkpoint=None,
            device="cpu",
            num_frames=4,
        )

        output_path = tmp_path / "output.gif"
        result_path = generator.generate_and_save(
            prompt="test animation",
            output_path=output_path,
            num_frames=4,
            num_steps=2,
            guidance_scale=1.0,
            seed=42,
            fps=8,
        )

        assert result_path.exists()
        with Image.open(result_path) as img:
            assert img.n_frames == 4


class TestAnimationGeneratorErrors:
    """Test error handling in AnimationGenerator."""

    def test_missing_vae_checkpoint(self, tmp_path: Path) -> None:
        """Test error when VAE checkpoint is missing."""
        from src.inference.animation_generator import AnimationGenerator

        with pytest.raises(FileNotFoundError, match="VAE checkpoint not found"):
            AnimationGenerator(
                vae_checkpoint=tmp_path / "nonexistent.pt",
                diffusion_checkpoint=tmp_path / "diff.pt",
                device="cpu",
            )

    def test_missing_diffusion_checkpoint(self, tmp_path: Path) -> None:
        """Test error when diffusion checkpoint is missing."""
        from src.inference.animation_generator import AnimationGenerator
        from src.models.vae import create_vae

        # Create proper VAE checkpoint
        vae = create_vae(image_size=64, z_channels=16)
        vae_path = tmp_path / "vae.pt"
        torch.save({"model_state_dict": vae.state_dict()}, vae_path)

        with pytest.raises(FileNotFoundError, match="Diffusion checkpoint not found"):
            AnimationGenerator(
                vae_checkpoint=vae_path,
                diffusion_checkpoint=tmp_path / "nonexistent.pt",
                device="cpu",
            )


@pytest.mark.skipif(not MMDIT_AVAILABLE, reason="mmdit package not installed")
class TestGenerateAnimationFunction:
    """Test the generate_animation convenience function."""

    @pytest.fixture
    def setup_checkpoints(self, tmp_path: Path) -> dict:
        """Create mock checkpoints for testing."""
        from src.models.vae import create_vae
        from src.models.factory import DiT

        # Create VAE checkpoint
        vae = create_vae(image_size=64, z_channels=16)
        vae_path = tmp_path / "checkpoints" / "vae.pt"
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": vae.state_dict(),
                "scaling_factor": 0.18215,
            },
            vae_path,
        )

        # Create diffusion checkpoint
        model = DiT(
            in_channels=16,
            image_size=8,
            patch_size=2,
            model_size="S",
            clip_embed_dim=512,
            model_type="mmdit",
        )
        diff_path = tmp_path / "checkpoints" / "diffusion.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "in_channels": 16,
                    "image_size": 64,
                    "latent_size": 8,
                    "patch_size": 2,
                    "model_size": "S",
                    "model_type": "mmdit",
                    "vae_checkpoint": str(vae_path),
                },
            },
            diff_path,
        )

        return {
            "vae_path": vae_path,
            "diff_path": diff_path,
        }

    def test_generate_animation_single_prompt(self, setup_checkpoints: dict) -> None:
        """Test generate_animation with single prompt."""
        from src.inference.animation_generator import generate_animation

        all_frames = generate_animation(
            prompts=["a cat"],
            vae_checkpoint=setup_checkpoints["vae_path"],
            diffusion_checkpoint=setup_checkpoints["diff_path"],
            motion_checkpoint=None,
            num_frames=4,
            num_steps=2,
            guidance_scale=1.0,
            seed=42,
            device="cpu",
        )

        assert len(all_frames) == 1
        assert len(all_frames[0]) == 4
        assert all(isinstance(f, Image.Image) for f in all_frames[0])

    def test_generate_animation_multiple_prompts(self, setup_checkpoints: dict) -> None:
        """Test generate_animation with multiple prompts."""
        from src.inference.animation_generator import generate_animation

        all_frames = generate_animation(
            prompts=["a cat", "a dog"],
            vae_checkpoint=setup_checkpoints["vae_path"],
            diffusion_checkpoint=setup_checkpoints["diff_path"],
            motion_checkpoint=None,
            num_frames=4,
            num_steps=2,
            guidance_scale=1.0,
            seed=42,
            device="cpu",
        )

        assert len(all_frames) == 2
        assert len(all_frames[0]) == 4
        assert len(all_frames[1]) == 4
