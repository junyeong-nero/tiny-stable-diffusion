"""Video transformation utilities for data augmentation.

Provides transforms and utilities specifically designed for video/GIF data,
including frame sampling, temporal augmentation, and consistent spatial transforms.
"""

from __future__ import annotations

import random
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms


def sample_frames_uniform(
    total_frames: int,
    num_frames: int,
    frame_skip: int = 1,
) -> list[int]:
    """Sample frames uniformly from video.

    Args:
        total_frames: Total number of frames in video
        num_frames: Number of frames to sample
        frame_skip: Skip every N frames (temporal subsampling)

    Returns:
        List of frame indices
    """
    # Calculate effective frames after skipping
    effective_frames = (total_frames + frame_skip - 1) // frame_skip

    if effective_frames <= num_frames:
        # Not enough frames, sample with repetition
        indices = list(range(0, total_frames, frame_skip))
        while len(indices) < num_frames:
            indices = indices + indices
        return indices[:num_frames]

    # Uniform sampling
    step = effective_frames / num_frames
    indices = [int(i * step) * frame_skip for i in range(num_frames)]

    # Clamp to valid range
    indices = [min(idx, total_frames - 1) for idx in indices]

    return indices


def sample_frames_random(
    total_frames: int,
    num_frames: int,
    frame_skip: int = 1,
) -> list[int]:
    """Sample frames randomly from video with temporal ordering.

    Args:
        total_frames: Total number of frames in video
        num_frames: Number of frames to sample
        frame_skip: Minimum gap between consecutive frames

    Returns:
        List of frame indices (sorted)
    """
    # Calculate effective frames
    effective_frames = (total_frames + frame_skip - 1) // frame_skip

    if effective_frames <= num_frames:
        indices = list(range(0, total_frames, frame_skip))
        while len(indices) < num_frames:
            indices = indices + indices
        return sorted(indices[:num_frames])

    # Random sampling with minimum gap
    available = list(range(0, total_frames, frame_skip))
    selected = sorted(random.sample(available, num_frames))

    return selected


def sample_frames_random_start(
    total_frames: int,
    num_frames: int,
    frame_skip: int = 1,
) -> list[int]:
    """Sample consecutive frames starting from random position.

    Args:
        total_frames: Total number of frames in video
        num_frames: Number of frames to sample
        frame_skip: Skip every N frames

    Returns:
        List of consecutive frame indices
    """
    # Calculate required length
    required_length = num_frames * frame_skip

    if total_frames <= required_length:
        # Not enough frames, sample uniformly
        return sample_frames_uniform(total_frames, num_frames, frame_skip)

    # Random start position
    max_start = total_frames - required_length
    start = random.randint(0, max_start)

    indices = [start + i * frame_skip for i in range(num_frames)]

    return indices


class VideoTransform:
    """Consistent spatial transform applied to all frames in a video.

    Ensures that random augmentations (flip, crop, etc.) are applied
    consistently across all frames in a video clip.
    """

    def __init__(
        self,
        target_size: int = 64,
        use_augmentation: bool = True,
        flip_prob: float = 0.5,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
    ) -> None:
        """Initialize VideoTransform.

        Args:
            target_size: Target frame size
            use_augmentation: Whether to apply augmentation
            flip_prob: Probability of horizontal flip
            brightness: Brightness jitter range
            contrast: Contrast jitter range
            saturation: Saturation jitter range
        """
        self.target_size = target_size
        self.use_augmentation = use_augmentation
        self.flip_prob = flip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

        # Base transforms (applied to all frames)
        self.resize = transforms.Resize(
            (target_size, target_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # Augmentation transforms (with random state)
        if use_augmentation:
            self.color_jitter = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
            )
        else:
            self.color_jitter = None

    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        """Apply consistent transform to all frames.

        Args:
            frames: List of PIL Images

        Returns:
            Tensor of shape (F, C, H, W)
        """
        # Decide augmentation parameters once for all frames
        do_flip = self.use_augmentation and random.random() < self.flip_prob

        # Get color jitter parameters (applied consistently)
        if self.use_augmentation and self.color_jitter is not None:
            # Sample jitter factors manually for consistency
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            do_jitter = True
        else:
            do_jitter = False
            brightness_factor = contrast_factor = saturation_factor = 1.0

        transformed_frames = []
        for frame in frames:
            # Resize
            frame = self.resize(frame)

            # Apply horizontal flip
            if do_flip:
                frame = transforms.functional.hflip(frame)

            # Apply color jitter with pre-computed factors
            if do_jitter:
                frame = transforms.functional.adjust_brightness(frame, brightness_factor)
                frame = transforms.functional.adjust_contrast(frame, contrast_factor)
                frame = transforms.functional.adjust_saturation(frame, saturation_factor)

            # To tensor and normalize
            frame = self.to_tensor(frame)
            frame = self.normalize(frame)

            transformed_frames.append(frame)

        return torch.stack(transformed_frames, dim=0)

    def transform_single(self, frame: Image.Image) -> torch.Tensor:
        """Transform a single frame (for inference).

        Args:
            frame: PIL Image

        Returns:
            Tensor of shape (C, H, W)
        """
        frame = self.resize(frame)
        frame = self.to_tensor(frame)
        frame = self.normalize(frame)
        return frame


def get_video_transforms(
    target_size: int = 64,
    use_augmentation: bool = True,
    flip_prob: float = 0.5,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
) -> Callable:
    """Get single-frame transform for video data.

    This returns a transform that can be applied to individual frames.
    For consistent augmentation across frames, use VideoTransform class.

    Args:
        target_size: Target frame size
        use_augmentation: Whether to apply augmentation
        flip_prob: Probability of horizontal flip
        brightness: Brightness jitter range
        contrast: Contrast jitter range
        saturation: Saturation jitter range

    Returns:
        Transform function
    """
    transform_list = [
        transforms.Resize(
            (target_size, target_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
    ]

    if use_augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
            ),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    return transforms.Compose(transform_list)


def get_video_inference_transform(target_size: int = 64) -> Callable:
    """Get transform for video inference (no augmentation).

    Args:
        target_size: Target frame size

    Returns:
        Transform function
    """
    return transforms.Compose([
        transforms.Resize(
            (target_size, target_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class TemporalAugmentation:
    """Temporal augmentation for video data.

    Applies augmentations that affect the temporal dimension,
    such as speed changes, reversal, and frame dropping.
    """

    def __init__(
        self,
        reverse_prob: float = 0.5,
        speed_change_prob: float = 0.3,
        speed_range: tuple[float, float] = (0.8, 1.2),
    ) -> None:
        """Initialize temporal augmentation.

        Args:
            reverse_prob: Probability of reversing video
            speed_change_prob: Probability of speed change
            speed_range: Range of speed multipliers
        """
        self.reverse_prob = reverse_prob
        self.speed_change_prob = speed_change_prob
        self.speed_range = speed_range

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply temporal augmentation.

        Args:
            frames: Tensor of shape (F, C, H, W)

        Returns:
            Augmented tensor of shape (F, C, H, W)
        """
        # Random reverse
        if random.random() < self.reverse_prob:
            frames = torch.flip(frames, dims=[0])

        # Random speed change (via frame sampling)
        if random.random() < self.speed_change_prob:
            speed = random.uniform(*self.speed_range)
            num_frames = frames.shape[0]
            new_num = int(num_frames * speed)

            if new_num != num_frames and new_num > 1:
                # Interpolate frames
                indices = torch.linspace(0, num_frames - 1, new_num).long()
                frames = frames[indices]

                # Pad or trim to original length
                if frames.shape[0] < num_frames:
                    padding = num_frames - frames.shape[0]
                    frames = torch.cat([
                        frames,
                        frames[-1:].repeat(padding, 1, 1, 1),
                    ], dim=0)
                elif frames.shape[0] > num_frames:
                    frames = frames[:num_frames]

        return frames


def denormalize_video(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize video tensor from [-1, 1] to [0, 1].

    Args:
        tensor: Normalized video tensor (F, C, H, W) or (B, F, C, H, W)

    Returns:
        Denormalized tensor
    """
    return tensor * 0.5 + 0.5


def normalize_video(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize video tensor from [0, 1] to [-1, 1].

    Args:
        tensor: Input video tensor

    Returns:
        Normalized tensor
    """
    return tensor * 2 - 1


def video_to_gif(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 8,
    loop: int = 0,
) -> None:
    """Save video tensor as GIF.

    Args:
        frames: Video tensor (F, C, H, W) in [-1, 1] or [0, 1]
        output_path: Output GIF path
        fps: Frames per second
        loop: Number of loops (0 = infinite)
    """
    # Denormalize if needed
    if frames.min() < 0:
        frames = denormalize_video(frames)

    # Clamp and convert to uint8
    frames = (frames.clamp(0, 1) * 255).byte()

    # Convert to list of PIL Images
    pil_frames = []
    for frame in frames:
        # (C, H, W) -> (H, W, C)
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        pil_frames.append(Image.fromarray(frame_np))

    # Save as GIF
    duration = int(1000 / fps)  # milliseconds per frame
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
    )


def frames_to_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 24,
) -> None:
    """Save video tensor as MP4 video.

    Requires imageio-ffmpeg or opencv-python.

    Args:
        frames: Video tensor (F, C, H, W) in [-1, 1] or [0, 1]
        output_path: Output video path
        fps: Frames per second
    """
    try:
        import imageio

        # Denormalize if needed
        if frames.min() < 0:
            frames = denormalize_video(frames)

        # Clamp and convert to uint8
        frames = (frames.clamp(0, 1) * 255).byte()

        # Convert to numpy array (F, H, W, C)
        frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()

        # Save video
        imageio.mimwrite(output_path, frames_np, fps=fps)

    except ImportError:
        raise ImportError(
            "imageio not found. Install with: pip install imageio imageio-ffmpeg"
        )


if __name__ == "__main__":
    # Test frame sampling
    print("Testing frame sampling...")

    total = 100
    num = 16
    skip = 2

    uniform = sample_frames_uniform(total, num, skip)
    print(f"Uniform sampling: {uniform}")

    rand = sample_frames_random(total, num, skip)
    print(f"Random sampling: {rand}")

    rand_start = sample_frames_random_start(total, num, skip)
    print(f"Random start sampling: {rand_start}")

    # Test video transform
    print("\nTesting VideoTransform...")
    transform = VideoTransform(target_size=64, use_augmentation=True)

    # Create dummy frames
    dummy_frames = [Image.new("RGB", (128, 128), color=(255, 0, 0)) for _ in range(8)]
    result = transform(dummy_frames)
    print(f"Transform result shape: {result.shape}")

    print("\n✓ All tests passed!")
