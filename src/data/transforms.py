"""Image transformation utilities for data augmentation."""

from __future__ import annotations

from typing import Any

from torchvision import transforms


def get_transforms(
    image_size: int = 32,
    use_augmentation: bool = True,
    flip_prob: float = 0.5,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
) -> transforms.Compose:
    """Get image transforms for training or inference.

    Args:
        image_size: Target image size (square)
        use_augmentation: Whether to apply augmentation
        flip_prob: Probability of horizontal flip
        brightness: Brightness jitter range
        contrast: Contrast jitter range
        saturation: Saturation jitter range

    Returns:
        Transforms composition
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]

    if use_augmentation:
        augmentations = [
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
            ),
        ]
        # Insert augmentations before ToTensor
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ] + augmentations + [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]

    return transforms.Compose(transform_list)


def get_inference_transform(image_size: int = 32) -> transforms.Compose:
    """Get transforms for inference (no augmentation).

    Args:
        image_size: Target image size

    Returns:
        Transforms for inference
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize image tensor from [-1, 1] to [0, 1].

    Args:
        tensor: Normalized image tensor

    Returns:
        Denormalized tensor
    """
    return tensor * 0.5 + 0.5


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor from [0, 1] to [-1, 1].

    Args:
        tensor: Input image tensor

    Returns:
        Normalized tensor
    """
    return tensor * 2 - 1


if __name__ == "__main__":
    # Test transforms
    train_transform = get_transforms(image_size=32, use_augmentation=True)
    print(f"Training transforms: {train_transform}")

    infer_transform = get_inference_transform(image_size=32)
    print(f"Inference transforms: {infer_transform}")
