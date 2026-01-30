"""Data loading and preprocessing for tiny-stable-diffusion.

Supports image-caption datasets from HuggingFace for training:
- CaptionDataset: General HuggingFace datasets (Oxford Pets, MSCOCO, etc.)
- StreamingCaptionDataset: Large streaming datasets (LAION, etc.)
- WebDatasetCaptionDataset: WebDataset format (pixparse/cc3m-wds, etc.)
"""

from __future__ import annotations

import io
import random
from typing import Callable

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms


class CaptionDataset(Dataset):
    """Dataset for image-caption pairs from HuggingFace.

    Supports various datasets like Oxford Pets, MSCOCO, Pokemon BLIP, etc.
    Automatically handles different field names and resizes images to target size.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split ("train", "validation", "test")
        cache_dir: Cache directory for downloaded datasets
        transform: Optional custom transform
        image_field: Name of the image field in the dataset
        caption_field: Name of the caption field in the dataset
        target_size: Target image size (default: 64)
        streaming: Use streaming mode for large datasets
        url_timeout: Timeout for URL requests in seconds
        max_retries: Maximum retries for failed URL requests

    Example datasets:
        - visual-layer/oxford-iiit-pet-vl-enriched: image="image", caption="caption_enriched"
        - clip-benchmark/wds_mscoco_captions: image="jpg", caption="txt"
        - reach-vb/pokemon-blip-captions: image="image", caption="text"
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: str = "~/.cache/tiny-stable-diffusion",
        transform: Callable | None = None,
        image_field: str = "image",
        caption_field: str = "caption",
        target_size: int = 64,
        streaming: bool = False,
        url_timeout: int = 10,
        max_retries: int = 3,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.image_field = image_field
        self.caption_field = caption_field
        self.target_size = target_size
        self.streaming = streaming
        self.url_timeout = url_timeout
        self.max_retries = max_retries
        self._buffer = []

        from pathlib import Path

        self.cache_dir = Path(cache_dir).expanduser()

        try:
            from datasets import load_dataset

            print(f"Loading dataset: {dataset_name} (split={split}, streaming={streaming})")

            if streaming:
                self.dataset_split = load_dataset(dataset_name, split=split, streaming=True)
                self.size = getattr(self.dataset_split, "num_rows", None) or 100000
            else:
                self.dataset = load_dataset(dataset_name, split=split, cache_dir=str(self.cache_dir))
                self.dataset_split = self.dataset
                self.size = len(self.dataset_split)

            print(f"✓ Loaded {self.dataset_name}: {self.size} samples")

        except ImportError:
            raise ImportError("datasets library not found. Install with: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform

    def _load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.url_timeout, headers={"User-Agent": "tiny-stable-diffusion/1.0"})
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.url_timeout}s"
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e.response.status_code}"
                if e.response.status_code == 404:
                    break
            except Exception as e:
                last_error = f"Failed: {e}"
        raise RuntimeError(f"Failed to load image from {url}: {last_error}")

    def _load_image(self, image_data) -> Image.Image:
        """Load image from various sources (URL, PIL Image, numpy array, bytes)."""
        if isinstance(image_data, str):
            if image_data.startswith(("http://", "https://")):
                image = self._load_image_from_url(image_data)
            else:
                image = Image.open(image_data)
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            import numpy as np
            if isinstance(image_data, np.ndarray):
                image = Image.fromarray(image_data)
            else:
                raise ValueError(f"Unsupported image type: {type(image_data)}")

        # Convert to RGB
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_caption(self, sample: dict, idx: int) -> str:
        """Extract caption from sample with fallback logic."""
        caption = None

        # Check for multi-column captions (e.g., caption_0, caption_1, ...)
        caption_columns = [key for key in sample.keys() if key.startswith("caption_")]
        if caption_columns:
            caption = sample.get(random.choice(caption_columns))
        else:
            # Try common caption fields
            for field in [self.caption_field, "text", "txt", "caption", "captions", "json"]:
                value = sample.get(field)
                if value is not None:
                    if isinstance(value, dict):
                        # Support various nested caption formats (COCO uses "raw")
                        caption = value.get("raw") or value.get("caption") or value.get("text")
                    else:
                        caption = value
                    if caption is not None:
                        break

        if caption is None:
            caption = f"image_{idx}"

        if isinstance(caption, list):
            caption = random.choice(caption)

        return str(caption)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if self.streaming:
            while len(self._buffer) <= idx:
                try:
                    sample = next(iter(self.dataset_split))
                    self._buffer.append(sample)
                except StopIteration:
                    break
            if idx >= len(self._buffer):
                raise IndexError(f"Index {idx} out of range")
            sample = self._buffer[idx]
        else:
            sample = self.dataset_split[idx]

        image_data = sample.get(self.image_field)
        if image_data is None:
            raise ValueError(f"Image field '{self.image_field}' not found in sample")

        image = self._load_image(image_data)
        caption = self._get_caption(sample, idx)

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption}

    def __repr__(self) -> str:
        return f"CaptionDataset({self.dataset_name}, size={self.size})"


class StreamingCaptionDataset(IterableDataset):
    """Streaming dataset for very large image-caption datasets.

    Uses true streaming with IterableDataset for memory-efficient training.
    Supports loading images from URLs with retry logic.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        transform: Optional custom transform
        image_field: Name of the image field
        caption_field: Name of the caption field
        target_size: Target image size
        url_timeout: Timeout for URL requests
        max_retries: Maximum retries for failed requests
        skip_failures: Skip failed samples instead of raising errors
        buffer_size: Shuffle buffer size for randomization
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Callable | None = None,
        image_field: str = "image",
        caption_field: str = "caption",
        target_size: int = 64,
        url_timeout: int = 10,
        max_retries: int = 3,
        skip_failures: bool = True,
        buffer_size: int = 1000,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.image_field = image_field
        self.caption_field = caption_field
        self.target_size = target_size
        self.url_timeout = url_timeout
        self.max_retries = max_retries
        self.skip_failures = skip_failures
        self.buffer_size = buffer_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform

        print(f"StreamingCaptionDataset: {dataset_name} (split={split})")
        print("Loading dataset (this may take a moment)...")
        from datasets import load_dataset
        self._dataset = load_dataset(dataset_name, split=split, streaming=True)
        print("Dataset loaded successfully.")

    def _load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.url_timeout, headers={"User-Agent": "tiny-stable-diffusion/1.0"})
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.url_timeout}s"
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e.response.status_code}"
                if e.response.status_code == 404:
                    break
            except Exception as e:
                last_error = f"Failed: {e}"
        raise RuntimeError(f"Failed to load image from {url}: {last_error}")

    def _load_image(self, image_data) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image_data, str):
            if image_data.startswith(("http://", "https://")):
                image = self._load_image_from_url(image_data)
            else:
                image = Image.open(image_data)
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            import numpy as np
            if isinstance(image_data, np.ndarray):
                image = Image.fromarray(image_data)
            else:
                raise ValueError(f"Unsupported image type: {type(image_data)}")

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_caption(self, sample: dict) -> str:
        """Extract caption from sample."""
        caption = None

        caption_columns = [key for key in sample.keys() if key.startswith("caption_")]
        if caption_columns:
            caption = sample.get(random.choice(caption_columns))
        else:
            for field in [self.caption_field, "text", "txt", "caption", "captions", "json"]:
                value = sample.get(field)
                if value is not None:
                    if isinstance(value, dict):
                        # Support various nested caption formats (COCO uses "raw")
                        caption = value.get("raw") or value.get("caption") or value.get("text")
                    else:
                        caption = value
                    if caption is not None:
                        break

        if caption is None:
            caption = "an image"

        if isinstance(caption, list):
            caption = random.choice(caption)

        return str(caption)

    def _process_sample(self, sample: dict) -> dict[str, torch.Tensor | str] | None:
        """Process a single sample."""
        try:
            image_data = sample.get(self.image_field)
            if image_data is None:
                return None

            image = self._load_image(image_data)
            caption = self._get_caption(sample)

            if self.transform:
                image = self.transform(image)

            return {"image": image, "caption": caption}
        except Exception:
            if self.skip_failures:
                return None
            raise

    def __iter__(self):
        buffer = []
        min_buffer_size = min(10, self.buffer_size)

        for sample in self._dataset:
            processed = self._process_sample(sample)
            if processed is not None:
                buffer.append(processed)

                if len(buffer) >= min_buffer_size:
                    if len(buffer) >= self.buffer_size:
                        idx = random.randint(0, len(buffer) - 1)
                        yield buffer.pop(idx)
                    else:
                        idx = random.randint(0, len(buffer) - 1)
                        yield buffer.pop(idx)

        random.shuffle(buffer)
        for sample in buffer:
            yield sample

    def __repr__(self) -> str:
        return f"StreamingCaptionDataset({self.dataset_name}, buffer_size={self.buffer_size})"


class WebDatasetCaptionDataset(IterableDataset):
    """WebDataset format streaming dataset for large image-caption datasets.

    Uses webdataset library for efficient streaming from tar files.
    Supports HuggingFace datasets in WebDataset format (e.g., pixparse/cc3m-wds).

    Args:
        dataset_name: HuggingFace dataset name or URL pattern
        transform: Optional custom transform
        image_field: Name of the image field (default: "jpg")
        caption_field: Name of the caption field (default: "txt")
        target_size: Target image size
        buffer_size: Shuffle buffer size for randomization
        skip_failures: Skip failed samples instead of raising errors
    """

    def __init__(
        self,
        dataset_name: str,
        transform: Callable | None = None,
        image_field: str = "jpg",
        caption_field: str = "txt",
        target_size: int = 64,
        buffer_size: int = 1000,
        skip_failures: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.image_field = image_field
        self.caption_field = caption_field
        self.target_size = target_size
        self.buffer_size = buffer_size
        self.skip_failures = skip_failures

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform

        # Build URL pattern for HuggingFace WebDataset
        self._urls = self._build_urls(dataset_name)
        print(f"WebDatasetCaptionDataset: {dataset_name}")
        print(f"  URLs: {len(self._urls)} shards")

    def _build_urls(self, dataset_name: str) -> list[str]:
        """Build list of tar URLs from HuggingFace dataset."""
        if dataset_name.startswith(("http://", "https://")):
            return [dataset_name]

        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        files = fs.ls(f"datasets/{dataset_name}", detail=False)
        tar_files = sorted([f for f in files if f.endswith(".tar")])

        urls = [
            f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{f.split('/')[-1]}"
            for f in tar_files
        ]
        return urls

    def _process_sample(self, sample: dict) -> dict[str, torch.Tensor | str] | None:
        """Process a single WebDataset sample."""
        try:
            image = sample.get(self.image_field)
            if image is None:
                return None

            # WebDataset with decode('pil') returns PIL Image directly
            if not isinstance(image, Image.Image):
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image))
                else:
                    return None

            # Convert to RGB
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            # Get caption
            caption = sample.get(self.caption_field, "an image")
            if isinstance(caption, bytes):
                caption = caption.decode("utf-8")

            if self.transform:
                image = self.transform(image)

            return {"image": image, "caption": str(caption)}
        except Exception:
            if self.skip_failures:
                return None
            raise

    def __iter__(self):
        import webdataset as wds

        # Create WebDataset pipeline with shuffling
        dataset = (
            wds.WebDataset(self._urls, shardshuffle=True)
            .shuffle(self.buffer_size)
            .decode("pil")
        )

        for sample in dataset:
            processed = self._process_sample(sample)
            if processed is not None:
                yield processed

    def __repr__(self) -> str:
        return f"WebDatasetCaptionDataset({self.dataset_name}, shards={len(self._urls)})"


if __name__ == "__main__":
    print("Testing CaptionDataset...")
    try:
        dataset = CaptionDataset(
            dataset_name="reach-vb/pokemon-blip-captions",
            split="train",
            image_field="image",
            caption_field="text",
            target_size=64,
        )
        print(f"Dataset: {dataset}")
        sample = dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample caption: {sample['caption']}")
        print("✓ Test passed!")
    except Exception as e:
        print(f"Error: {e}")
