"""Dataset loading utilities."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.data.dataset import CaptionDataset, StreamingCaptionDataset


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

    if data_source in ("huggingface", "caption"):
        # HuggingFace image-caption dataset
        return CaptionDataset(
            dataset_name=config.get("dataset_name", "reach-vb/pokemon-blip-captions"),
            split=config.get("split", "train"),
            image_field=config.get("image_field", "image"),
            caption_field=config.get("caption_field", "caption"),
            target_size=config.get("image_size", 64),
            streaming=config.get("streaming", False),
            url_timeout=config.get("url_timeout", 10),
            max_retries=config.get("max_retries", 3),
        )

    elif data_source == "streaming_caption":
        # Streaming dataset for very large datasets (LAION, etc.)
        return StreamingCaptionDataset(
            dataset_name=config.get("dataset_name"),
            split=config.get("split", "train"),
            image_field=config.get("image_field", "image"),
            caption_field=config.get("caption_field", "caption"),
            target_size=config.get("image_size", 64),
            url_timeout=config.get("url_timeout", 10),
            max_retries=config.get("max_retries", 3),
            skip_failures=config.get("skip_failures", True),
            buffer_size=config.get("buffer_size", 1000),
        )

    else:
        raise ValueError(f"Unknown data_source: {data_source}. Use 'huggingface' or 'streaming_caption'.")


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
    if isinstance(dataset, IterableDataset):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
