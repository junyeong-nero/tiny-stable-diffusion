"""Dataset loading utilities."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset

from src.data.dataset import (
    CIFAR100Dataset,
    CaptionDataset,
    EmojiDataset,
    StreamingCaptionDataset,
)


def get_dataset(config: dict[str, Any]) -> Dataset:
    """Create dataset based on config.

    Args:
        config: Configuration dictionary with data_source and related settings

    Returns:
        Dataset instance

    Raises:
        ValueError: If data_source is unknown
    """
    data_source = config["data_source"]

    if data_source == "local":
        local_path = config.get("local_dataset_path")
        print(f"Loading local dataset from: {local_path}")
        return EmojiDataset(dataset_name=local_path, split="train")

    elif data_source == "huggingface":
        dataset_name = config.get("dataset_name", "junyeong-nero/emoji-32")
        print(f"Loading Hugging Face dataset: {dataset_name}")
        return EmojiDataset(dataset_name=dataset_name, split="train")

    elif data_source == "caption":
        # General image-caption dataset (Flickr8k, CC3M, Pokemon BLIP, etc.)
        dataset_name = config.get("dataset_name", "ariG23498/flickr8k")
        image_field = config.get("image_field", "image")
        caption_field = config.get("caption_field", "caption")
        target_size = config.get("image_size", 32)
        streaming = config.get("streaming", False)
        split = config.get("split", "train")
        url_timeout = config.get("url_timeout", 10)
        max_retries = config.get("max_retries", 3)

        print(f"Loading caption dataset: {dataset_name}")
        return CaptionDataset(
            dataset_name=dataset_name,
            split=split,
            image_field=image_field,
            caption_field=caption_field,
            target_size=target_size,
            streaming=streaming,
            url_timeout=url_timeout,
            max_retries=max_retries,
        )

    elif data_source == "streaming_caption":
        # Streaming dataset for very large datasets (LAION, Open Images, etc.)
        dataset_name = config.get("dataset_name")
        image_field = config.get("image_field", "image")
        caption_field = config.get("caption_field", "caption")
        target_size = config.get("image_size", 64)
        split = config.get("split", "train")
        url_timeout = config.get("url_timeout", 10)
        max_retries = config.get("max_retries", 3)
        skip_failures = config.get("skip_failures", True)
        buffer_size = config.get("buffer_size", 1000)

        print(f"Loading streaming caption dataset: {dataset_name}")
        return StreamingCaptionDataset(
            dataset_name=dataset_name,
            split=split,
            image_field=image_field,
            caption_field=caption_field,
            target_size=target_size,
            url_timeout=url_timeout,
            max_retries=max_retries,
            skip_failures=skip_failures,
            buffer_size=buffer_size,
        )

    elif data_source == "cifar100":
        use_coarse = config.get("use_coarse_labels", False)
        print("Loading CIFAR-100 dataset")
        return CIFAR100Dataset(train=True, use_coarse_labels=use_coarse)

    else:
        raise ValueError(f"Unknown data_source: {data_source}")


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the given dataset.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
