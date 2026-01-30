"""Dataset loading utilities."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from src.data.dataset import CaptionDataset, StreamingCaptionDataset, WebDatasetCaptionDataset


def get_dataset(config: dict[str, Any]) -> Dataset:
    """Create dataset based on config.

    All datasets are loaded from HuggingFace. Use 'stream' option to control
    whether to use streaming mode for large datasets.

    Args:
        config: Configuration dictionary with dataset settings
            - dataset_name: HuggingFace dataset name
            - stream: If True, use StreamingCaptionDataset (IterableDataset)
                      If False, use CaptionDataset (map-style Dataset)
            - use_webdataset: If True, use WebDatasetCaptionDataset for tar files
            - Other settings: split, image_field, caption_field, etc.

    Returns:
        Dataset instance
    """
    stream = config.get("stream", False)
    use_webdataset = config.get("use_webdataset", False)

    if use_webdataset:
        # WebDataset format (tar files, e.g., pixparse/cc3m-wds)
        return WebDatasetCaptionDataset(
            dataset_name=config.get("dataset_name"),
            image_field=config.get("image_field", "jpg"),
            caption_field=config.get("caption_field", "txt"),
            target_size=config.get("image_size", 64),
            buffer_size=config.get("buffer_size", 1000),
            skip_failures=config.get("skip_failures", True),
        )
    elif stream:
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
        # Standard HuggingFace dataset (fully loaded into memory)
        return CaptionDataset(
            dataset_name=config.get("dataset_name", "reach-vb/pokemon-blip-captions"),
            split=config.get("split", "train"),
            image_field=config.get("image_field", "image"),
            caption_field=config.get("caption_field", "caption"),
            target_size=config.get("image_size", 64),
            streaming=False,
            url_timeout=config.get("url_timeout", 10),
            max_retries=config.get("max_retries", 3),
        )


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


def get_train_val_datasets(
    config: dict[str, Any],
    val_split: float = 0.1,
) -> tuple[Dataset, Dataset | None]:
    """Create train and validation datasets based on config.

    For map-style datasets (non-streaming), splits the dataset by indices.
    For streaming datasets, returns None for validation (streaming doesn't support split).

    Args:
        config: Configuration dictionary with dataset settings
        val_split: Fraction of data to use for validation (0.0-1.0)

    Returns:
        Tuple of (train_dataset, val_dataset). val_dataset is None for streaming.
    """
    import random

    stream = config.get("stream", False)
    use_webdataset = config.get("use_webdataset", False)

    if use_webdataset:
        # WebDataset format (tar files) - streaming, no validation split
        train_dataset = WebDatasetCaptionDataset(
            dataset_name=config.get("dataset_name"),
            image_field=config.get("image_field", "jpg"),
            caption_field=config.get("caption_field", "txt"),
            target_size=config.get("image_size", 64),
            buffer_size=config.get("buffer_size", 1000),
            skip_failures=config.get("skip_failures", True),
        )
        return train_dataset, None

    if stream:
        # Streaming datasets don't support splitting
        # Return train dataset only, validation will use a separate approach
        train_dataset = StreamingCaptionDataset(
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
        return train_dataset, None

    # Non-streaming dataset: load and split
    full_dataset = CaptionDataset(
        dataset_name=config.get("dataset_name", "reach-vb/pokemon-blip-captions"),
        split=config.get("split", "train"),
        image_field=config.get("image_field", "image"),
        caption_field=config.get("caption_field", "caption"),
        target_size=config.get("image_size", 64),
        streaming=False,
        url_timeout=config.get("url_timeout", 10),
        max_retries=config.get("max_retries", 3),
    )

    if val_split <= 0 or val_split >= 1:
        return full_dataset, None

    # Create train/val split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    # Use seed for reproducibility
    seed = config.get("seed", 42)
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(dataset_size * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Dataset split: train={len(train_indices)}, val={len(val_indices)}")

    return train_dataset, val_dataset


def create_train_val_dataloaders(
    config: dict[str, Any],
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and validation dataloaders.

    Args:
        config: Configuration dictionary
        val_split: Fraction of data for validation

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataset, val_dataset = get_train_val_datasets(config, val_split)

    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 0)

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_dataloader, val_dataloader
