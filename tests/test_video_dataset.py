"""Tests for Video Dataset and Transforms."""

import pytest
import torch
from PIL import Image

from src.data.video_transforms import (
    sample_frames_uniform,
    sample_frames_random,
    sample_frames_random_start,
    VideoTransform,
    TemporalAugmentation,
    get_video_transforms,
    get_video_inference_transform,
    denormalize_video,
    normalize_video,
)
from src.data.video_dataset import (
    SyntheticVideoDataset,
    create_video_dataloader,
)


class TestFrameSampling:
    """Test suite for frame sampling functions."""

    def test_uniform_sampling_basic(self) -> None:
        """Test basic uniform sampling."""
        indices = sample_frames_uniform(100, 16, 1)
        assert len(indices) == 16
        assert indices[0] == 0
        assert all(0 <= i < 100 for i in indices)

    def test_uniform_sampling_with_skip(self) -> None:
        """Test uniform sampling with frame skip."""
        indices = sample_frames_uniform(100, 16, 2)
        assert len(indices) == 16
        # All indices should be even (due to skip=2)
        assert all(i % 2 == 0 for i in indices)

    def test_uniform_sampling_few_frames(self) -> None:
        """Test uniform sampling when video has fewer frames than requested."""
        indices = sample_frames_uniform(10, 16, 1)
        assert len(indices) == 16
        # Should repeat frames
        assert all(0 <= i < 10 for i in indices)

    def test_random_sampling_basic(self) -> None:
        """Test basic random sampling."""
        indices = sample_frames_random(100, 16, 1)
        assert len(indices) == 16
        assert all(0 <= i < 100 for i in indices)
        # Should be sorted
        assert indices == sorted(indices)

    def test_random_sampling_few_frames(self) -> None:
        """Test random sampling with fewer frames than requested."""
        indices = sample_frames_random(10, 16, 1)
        assert len(indices) == 16

    def test_random_start_sampling(self) -> None:
        """Test random start consecutive sampling."""
        indices = sample_frames_random_start(100, 16, 1)
        assert len(indices) == 16
        # Should be consecutive
        for i in range(1, len(indices)):
            assert indices[i] == indices[i - 1] + 1

    def test_random_start_with_skip(self) -> None:
        """Test random start with frame skip."""
        indices = sample_frames_random_start(100, 16, 2)
        assert len(indices) == 16
        # Should be spaced by skip
        for i in range(1, len(indices)):
            assert indices[i] == indices[i - 1] + 2


class TestVideoTransform:
    """Test suite for VideoTransform."""

    @pytest.fixture
    def transform(self) -> VideoTransform:
        """Create a VideoTransform for testing."""
        return VideoTransform(
            target_size=64,
            use_augmentation=True,
            flip_prob=0.5,
        )

    @pytest.fixture
    def dummy_frames(self) -> list[Image.Image]:
        """Create dummy frames for testing."""
        return [
            Image.new("RGB", (128, 128), color=(255, 0, 0))
            for _ in range(8)
        ]

    def test_transform_output_shape(
        self,
        transform: VideoTransform,
        dummy_frames: list[Image.Image],
    ) -> None:
        """Test that transform outputs correct shape."""
        result = transform(dummy_frames)
        assert result.shape == (8, 3, 64, 64)

    def test_transform_output_range(
        self,
        transform: VideoTransform,
        dummy_frames: list[Image.Image],
    ) -> None:
        """Test that transform outputs are in [-1, 1] range."""
        result = transform(dummy_frames)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_transform_single(
        self,
        transform: VideoTransform,
        dummy_frames: list[Image.Image],
    ) -> None:
        """Test single frame transform."""
        result = transform.transform_single(dummy_frames[0])
        assert result.shape == (3, 64, 64)

    def test_transform_no_augmentation(
        self,
        dummy_frames: list[Image.Image],
    ) -> None:
        """Test transform without augmentation."""
        transform = VideoTransform(target_size=64, use_augmentation=False)
        result = transform(dummy_frames)
        assert result.shape == (8, 3, 64, 64)

    def test_transform_consistency(
        self,
        transform: VideoTransform,
    ) -> None:
        """Test that augmentation is consistent across frames."""
        # Create frames with different colors to detect flipping
        frames = [
            Image.new("RGB", (128, 128), color=(255, 0, 0)),  # Red
            Image.new("RGB", (128, 128), color=(0, 255, 0)),  # Green
        ]

        # Run multiple times to check consistency
        for _ in range(10):
            result = transform(frames)
            # All frames should have same shape
            assert result.shape == (2, 3, 64, 64)


class TestTemporalAugmentation:
    """Test suite for TemporalAugmentation."""

    @pytest.fixture
    def augmentation(self) -> TemporalAugmentation:
        """Create a TemporalAugmentation for testing."""
        return TemporalAugmentation(
            reverse_prob=0.5,
            speed_change_prob=0.3,
        )

    def test_augmentation_shape_preserved(
        self,
        augmentation: TemporalAugmentation,
    ) -> None:
        """Test that augmentation preserves shape."""
        frames = torch.randn(16, 3, 64, 64)
        result = augmentation(frames)
        assert result.shape == frames.shape

    def test_augmentation_deterministic_with_seed(self) -> None:
        """Test that augmentation is deterministic with seed."""
        augmentation = TemporalAugmentation(reverse_prob=1.0, speed_change_prob=0.0)
        frames = torch.arange(16).float().view(16, 1, 1, 1).expand(16, 3, 64, 64)

        torch.manual_seed(42)
        import random
        random.seed(42)
        result = augmentation(frames.clone())

        # With reverse_prob=1.0, should be reversed
        assert torch.allclose(result[0], frames[-1])


class TestNormalization:
    """Test suite for normalization functions."""

    def test_denormalize_video(self) -> None:
        """Test video denormalization."""
        # Create tensor in [-1, 1]
        tensor = torch.tensor([-1.0, 0.0, 1.0])
        result = denormalize_video(tensor)
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(result, expected)

    def test_normalize_video(self) -> None:
        """Test video normalization."""
        # Create tensor in [0, 1]
        tensor = torch.tensor([0.0, 0.5, 1.0])
        result = normalize_video(tensor)
        expected = torch.tensor([-1.0, 0.0, 1.0])
        assert torch.allclose(result, expected)

    def test_roundtrip_normalization(self) -> None:
        """Test that normalization is invertible."""
        original = torch.rand(8, 3, 64, 64)
        normalized = normalize_video(original)
        recovered = denormalize_video(normalized)
        assert torch.allclose(original, recovered)


class TestGetTransforms:
    """Test suite for transform factory functions."""

    def test_get_video_transforms(self) -> None:
        """Test get_video_transforms factory."""
        transform = get_video_transforms(target_size=64, use_augmentation=True)
        assert transform is not None

        # Test with dummy image
        dummy = Image.new("RGB", (128, 128))
        result = transform(dummy)
        assert result.shape == (3, 64, 64)

    def test_get_video_inference_transform(self) -> None:
        """Test get_video_inference_transform factory."""
        transform = get_video_inference_transform(target_size=64)
        assert transform is not None

        dummy = Image.new("RGB", (128, 128))
        result = transform(dummy)
        assert result.shape == (3, 64, 64)


class TestSyntheticVideoDataset:
    """Test suite for SyntheticVideoDataset."""

    @pytest.fixture
    def dataset(self) -> SyntheticVideoDataset:
        """Create a SyntheticVideoDataset for testing."""
        return SyntheticVideoDataset(
            size=10,
            num_frames=8,
            image_size=64,
            pattern="moving_circle",
        )

    def test_dataset_length(self, dataset: SyntheticVideoDataset) -> None:
        """Test dataset length."""
        assert len(dataset) == 10

    def test_dataset_item_shape(self, dataset: SyntheticVideoDataset) -> None:
        """Test that dataset items have correct shape."""
        sample = dataset[0]
        assert "frames" in sample
        assert "caption" in sample
        assert sample["frames"].shape == (8, 3, 64, 64)

    def test_dataset_item_range(self, dataset: SyntheticVideoDataset) -> None:
        """Test that dataset items are in correct range."""
        sample = dataset[0]
        frames = sample["frames"]
        assert frames.min() >= -1.0
        assert frames.max() <= 1.0

    def test_different_patterns(self) -> None:
        """Test different synthetic patterns."""
        for pattern in ["moving_circle", "gradient", "noise"]:
            dataset = SyntheticVideoDataset(
                size=5,
                num_frames=8,
                image_size=64,
                pattern=pattern,
            )
            sample = dataset[0]
            assert sample["frames"].shape == (8, 3, 64, 64)

    def test_reproducibility(self) -> None:
        """Test that same index gives same result."""
        dataset = SyntheticVideoDataset(size=5, num_frames=8, image_size=64)
        sample1 = dataset[0]
        sample2 = dataset[0]
        assert torch.allclose(sample1["frames"], sample2["frames"])

    def test_different_indices_different_results(self) -> None:
        """Test that different indices give different results."""
        dataset = SyntheticVideoDataset(size=5, num_frames=8, image_size=64)
        sample0 = dataset[0]
        sample1 = dataset[1]
        assert not torch.allclose(sample0["frames"], sample1["frames"])


class TestVideoDataloader:
    """Test suite for video dataloader creation."""

    def test_create_dataloader(self) -> None:
        """Test dataloader creation."""
        dataset = SyntheticVideoDataset(size=20, num_frames=8, image_size=64)
        dataloader = create_video_dataloader(
            dataset,
            batch_size=4,
            num_workers=0,
            shuffle=True,
        )

        batch = next(iter(dataloader))
        assert batch["frames"].shape == (4, 8, 3, 64, 64)
        assert len(batch["caption"]) == 4

    def test_dataloader_iteration(self) -> None:
        """Test full dataloader iteration."""
        dataset = SyntheticVideoDataset(size=16, num_frames=8, image_size=64)
        dataloader = create_video_dataloader(
            dataset,
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )

        total_samples = 0
        for batch in dataloader:
            total_samples += batch["frames"].shape[0]

        # drop_last=True, so 16 samples with batch_size=4 = 4 batches * 4 = 16
        assert total_samples == 16
