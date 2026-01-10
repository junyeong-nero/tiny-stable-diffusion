"""Data loading and preprocessing for emoji dataset.

Supports multiple datasets:
- junyeong-nero/emoji-32: Custom emoji dataset (32x32)
- CIFAR-100: General image classification dataset (32x32)
- Flickr8k, CC3M, Pokemon BLIP: General image-caption datasets
- Any HuggingFace dataset with image and text/caption fields

Images are automatically resized to 32x32 for consistency.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from torchvision.datasets import CIFAR100


class EmojiDataset(Dataset):
    """Dataset for loading and processing emoji images.

    Loads images from the junyeong-nero/emoji-32 Hugging Face dataset.
    Images are already 32x32 RGB, no resizing needed.

    Dataset source: https://huggingface.co/datasets/junyeong-nero/emoji-32

    Dataset structure:
        - image_apple: PIL Image (32x32, RGBA)
        - short_name: str (e.g., "rocket")
    """

    def __init__(
        self,
        dataset_name: str = "junyeong-nero/emoji-32",
        split: str = "train",
        cache_dir: str = "~/.cache/tiny-stable-diffusion",
        transform: Callable | None = None,
        streaming: bool = False,
        image_field: str = "image_apple",
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = Path(cache_dir).expanduser()
        self.streaming = streaming
        self.image_field = image_field
        self.caption_field = "short_name"
        self._buffer = []

        try:
            from datasets import load_dataset

            if streaming:
                self.dataset_split = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=True,
                )
                self.size = getattr(self.dataset_split, "num_rows", None) or 10000
            else:
                self.dataset = load_dataset(
                    dataset_name,
                    split=split,
                    cache_dir=str(self.cache_dir),
                )
                self.dataset_split = self.dataset
                self.size = len(self.dataset_split)

        except ImportError:
            raise ImportError("datasets library not found. Install with: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

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

        image = sample[self.image_field]
        caption = sample.get(self.caption_field, f"emoji_{idx}")

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
        }

    def __getitems__(self, indices: list[int]) -> list[dict]:
        if self.streaming:
            return [self[i] for i in indices]
        else:
            samples = self.dataset_split.select(indices)
            result = []
            for sample in samples:
                image = sample[self.image_field]
                if image is None:
                    continue
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                if image.mode == "RGBA":
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
                result.append(
                    {
                        "image": image,
                        "caption": sample.get(self.caption_field, ""),
                    }
                )
            return result

    def __repr__(self) -> str:
        return (
            f"EmojiDataset("
            f"dataset={self.dataset_name}, "
            f"split={self.split}, "
            f"num_samples={self.size})"
        )


class CIFAR100Dataset(Dataset):
    """CIFAR-100 dataset wrapper for text-to-image generation.

    Uses torchvision.datasets.CIFAR100 with its built-in class names.

    Dataset source: torchvision.datasets.CIFAR100

    Features:
        - 60,000 images (32x32 RGB)
        - 100 classes (20 superclasses)
        - Can use either coarse or fine labels as text conditioning
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        use_coarse_labels: bool = False,
        transform: Callable | None = None,
        download: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.train = train
        self.use_coarse_labels = use_coarse_labels

        # Load CIFAR-100 from torchvision (classes are available after download)
        self.dataset = CIFAR100(
            root=str(self.root),
            train=train,
            transform=transform,
            download=download,
        )

        # Use torchvision's built-in class names
        self.fine_labels: list[str] = self.dataset.classes

        # CIFAR-100 coarse labels (20 superclasses)
        self.coarse_labels = [
            "aquatic_mammals",
            "fish",
            "flowers",
            "food_containers",
            "fruit_and_vegetables",
            "household_electrical_devices",
            "household_furniture",
            "insects",
            "large_carnivores",
            "large_man-made_outdoor_things",
            "large_natural_outdoor_scenes",
            "large_omnivores_and_herbivores",
            "medium_sized_mammals",
            "non-insect_invertebrates",
            "people",
            "reptiles",
            "small_mammals",
            "trees",
            "vehicles_1",
            "vehicles_2",
        ]

        # Coarse label mapping (fine -> coarse)
        self.fine_to_coarse = [
            4,
            1,
            14,
            8,
            0,
            6,
            7,
            3,
            2,
            15,
            11,
            17,
            5,
            10,
            4,
            13,
            12,
            16,
            9,
            5,
            10,
            11,
            10,
            10,
            8,
            4,
            3,
            2,
            9,
            15,
            14,
            6,
            7,
            1,
            13,
            5,
            6,
            3,
            18,
            17,
            12,
            18,
            4,
            11,
            16,
            0,
            15,
            2,
            17,
            8,
            14,
            13,
            12,
            18,
            1,
            9,
            11,
            5,
            3,
            6,
            15,
            0,
            2,
            7,
            4,
            19,
            6,
            16,
            5,
            8,
            9,
            10,
            10,
            11,
            3,
            15,
            7,
            18,
            19,
            14,
            12,
            19,
            2,
            13,
            16,
            17,
            18,
            2,
            4,
            6,
            11,
            8,
            2,
            12,
            1,
            8,
            4,
            7,
            0,
            16,
            12,
            17,
            3,
            19,
            15,
            4,
            9,
            14,
            1,
            2,
        ]

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        # Validate label index
        if label < 0 or label >= len(self.fine_labels):
            raise IndexError(
                f"Label index {label} is out of range for fine_labels (size={len(self.fine_labels)}). "
                "This may indicate a corrupted or incompatible dataset."
            )

        # Get text label
        if self.use_coarse_labels:
            coarse_idx = self.fine_to_coarse[label]
            if coarse_idx < 0 or coarse_idx >= len(self.coarse_labels):
                raise IndexError(
                    f"Coarse label index {coarse_idx} is out of range (size={len(self.coarse_labels)})."
                )
            coarse_label = self.coarse_labels[coarse_idx]
            caption = coarse_label.replace("_", " ")
        else:
            caption = self.fine_labels[label]

        return {
            "image": image,
            "caption": caption,
        }

    def __repr__(self) -> str:
        label_type = "coarse" if self.use_coarse_labels else "fine"
        return f"CIFAR100Dataset(train={self.train}, labels={label_type}, num_samples={len(self)})"


class CaptionDataset(Dataset):
    """General-purpose dataset for image-caption pairs from HuggingFace.

    Supports various datasets like Flickr8k, CC3M, Pokemon BLIP, etc.
    Automatically handles different field names and resizes images to target size.
    Supports loading images from URLs.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "ariG23498/flickr8k")
        split: Dataset split ("train", "validation", "test")
        cache_dir: Cache directory for downloaded datasets
        transform: Optional custom transform (default: ToTensor + Normalize)
        image_field: Name of the image field in the dataset
        caption_field: Name of the caption field in the dataset
        target_size: Target image size (default: 32)
        streaming: Use streaming mode for large datasets
        url_timeout: Timeout for URL requests in seconds (default: 10)
        max_retries: Maximum retries for failed URL requests (default: 3)

    Example datasets:
        - ariG23498/flickr8k: image="image", caption="caption"
        - reach-vb/pokemon-blip-captions: image="image", caption="text"
        - dalle-mini/open-images: image="url", caption="caption"
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: str = "~/.cache/tiny-stable-diffusion",
        transform: Callable | None = None,
        image_field: str = "image",
        caption_field: str = "caption",
        target_size: int = 32,
        streaming: bool = False,
        url_timeout: int = 10,
        max_retries: int = 3,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = Path(cache_dir).expanduser()
        self.image_field = image_field
        self.caption_field = caption_field
        self.target_size = target_size
        self.streaming = streaming
        self.url_timeout = url_timeout
        self.max_retries = max_retries
        self._buffer = []

        try:
            from datasets import load_dataset

            print(f"Loading dataset: {dataset_name} (split={split}, streaming={streaming})")

            if streaming:
                self.dataset_split = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=True,
                )
                self.size = getattr(self.dataset_split, "num_rows", None) or 100000
            else:
                self.dataset = load_dataset(
                    dataset_name,
                    split=split,
                    cache_dir=str(self.cache_dir),
                )
                self.dataset_split = self.dataset
                self.size = len(self.dataset_split)

            print(f"✓ Loaded {self.dataset_name}: {self.size} samples")

        except ImportError:
            raise ImportError("datasets library not found. Install with: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (target_size, target_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

    def _load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL with retry logic.

        Args:
            url: Image URL

        Returns:
            PIL Image

        Raises:
            RuntimeError: If image cannot be loaded after max retries
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    timeout=self.url_timeout,
                    headers={"User-Agent": "tiny-stable-diffusion/1.0"},
                )
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                return image
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.url_timeout}s"
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e.response.status_code}"
                if e.response.status_code == 404:
                    break  # Don't retry on 404
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"
            except Exception as e:
                last_error = f"Failed to open image: {e}"

        raise RuntimeError(f"Failed to load image from {url}: {last_error}")

    def _load_image(self, image_data) -> Image.Image:
        """Load image from various sources (URL, PIL Image, numpy array, bytes).

        Args:
            image_data: Image data (URL string, PIL Image, numpy array, or bytes)

        Returns:
            PIL Image in RGB format
        """
        # Handle URL strings
        if isinstance(image_data, str):
            if image_data.startswith(("http://", "https://")):
                image = self._load_image_from_url(image_data)
            else:
                # Assume it's a file path
                image = Image.open(image_data)
        # Handle bytes
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        # Handle PIL Image
        elif isinstance(image_data, Image.Image):
            image = image_data
        # Handle numpy array
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
        """Extract caption from sample with fallback logic.

        Args:
            sample: Dataset sample dictionary
            idx: Sample index (for fallback caption)

        Returns:
            Caption string
        """
        caption = None

        # Check for multi-column captions (e.g., caption_0, caption_1, ...)
        caption_columns = [key for key in sample.keys() if key.startswith("caption_")]
        if caption_columns:
            selected_col = random.choice(caption_columns)
            caption = sample.get(selected_col)
        else:
            # Try single caption field
            for field in [self.caption_field, "text", "caption", "captions", "json"]:
                value = sample.get(field)
                if value is not None:
                    # Handle nested dict (e.g., json: {"caption": "..."})
                    if isinstance(value, dict):
                        caption = value.get("caption") or value.get("text")
                    else:
                        caption = value
                    if caption is not None:
                        break

        if caption is None:
            caption = f"image_{idx}"

        # Handle multiple captions if it's a list
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

        # Get image data
        image_data = sample.get(self.image_field)
        if image_data is None:
            raise ValueError(f"Image field '{self.image_field}' not found in sample")

        # Load and process image
        image = self._load_image(image_data)

        # Get caption
        caption = self._get_caption(sample, idx)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
        }

    def __repr__(self) -> str:
        return (
            f"CaptionDataset("
            f"dataset={self.dataset_name}, "
            f"split={self.split}, "
            f"size={self.target_size}x{self.target_size}, "
            f"streaming={self.streaming}, "
            f"num_samples={self.size})"
        )


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
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (target_size, target_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

        # Load dataset once and cache it
        print(f"StreamingCaptionDataset: {dataset_name} (split={split})")
        print("Loading dataset (this may take a moment)...")
        from datasets import load_dataset

        self._dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
        )
        print("Dataset loaded successfully.")

    def _load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    timeout=self.url_timeout,
                    headers={"User-Agent": "tiny-stable-diffusion/1.0"},
                )
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                return image
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.url_timeout}s"
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e.response.status_code}"
                if e.response.status_code == 404:
                    break
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"
            except Exception as e:
                last_error = f"Failed to open image: {e}"

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

        # Convert to RGB
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
            selected_col = random.choice(caption_columns)
            caption = sample.get(selected_col)
        else:
            for field in [self.caption_field, "text", "caption", "captions", "json"]:
                value = sample.get(field)
                if value is not None:
                    # Handle nested dict (e.g., json: {"caption": "..."})
                    if isinstance(value, dict):
                        caption = value.get("caption") or value.get("text")
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

            return {
                "image": image,
                "caption": caption,
            }
        except Exception as e:
            if self.skip_failures:
                return None
            raise e

    def __iter__(self):
        # Shuffle buffer for randomization
        buffer = []

        for sample in self._dataset:
            processed = self._process_sample(sample)
            if processed is not None:
                buffer.append(processed)

                if len(buffer) >= self.buffer_size:
                    # Yield a random sample from buffer
                    idx = random.randint(0, len(buffer) - 1)
                    yield buffer.pop(idx)

        # Yield remaining samples in buffer
        random.shuffle(buffer)
        for sample in buffer:
            yield sample

    def __repr__(self) -> str:
        return (
            f"StreamingCaptionDataset("
            f"dataset={self.dataset_name}, "
            f"split={self.split}, "
            f"size={self.target_size}x{self.target_size}, "
            f"buffer_size={self.buffer_size})"
        )


class LocalEmojiDataset(Dataset):
    """Local emoji dataset for offline training."""

    def __init__(
        self,
        data_dir: str = "data",
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)

        self.extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
        self.image_paths = []
        for ext in self.extensions:
            self.image_paths.extend(self.data_dir.rglob(f"*{ext}"))
        self.image_paths = sorted(self.image_paths)

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = self._extract_caption(image_path.name)

        return {
            "image": image,
            "caption": caption,
            "path": str(image_path),
        }

    def _extract_caption(self, filename: str) -> str:
        name = Path(filename).stem
        import re

        if re.match(r"^[0-9a-fA-F]+$", name):
            return name
        parts = name.split("_")
        caption = " ".join(parts)
        caption = caption.replace("-", " ")
        caption = re.sub(r"\s+", " ", caption).strip()
        return caption.lower()

    def __repr__(self) -> str:
        return f"LocalEmojiDataset(data_dir={self.data_dir}, num_samples={len(self)})"


def get_dataset(
    dataset_source: str = "huggingface",
    dataset_name: str = "junyeong-nero/emoji-32",
    **kwargs,
) -> Dataset:
    """Get dataset by source type.

    Args:
        dataset_source: Source type - "huggingface", "cifar100", or "local"
        dataset_name: Dataset name (for HuggingFace) or path (for local)
        **kwargs: Additional arguments for specific datasets

    Returns:
        Dataset instance
    """
    if dataset_source == "huggingface":
        return EmojiDataset(dataset_name=dataset_name, **kwargs)
    elif dataset_source == "cifar100":
        use_coarse = kwargs.get("use_coarse_labels", False)
        return CIFAR100Dataset(use_coarse_labels=use_coarse, **kwargs)
    else:
        return LocalEmojiDataset(**kwargs)


if __name__ == "__main__":
    print("Testing Hugging Face dataset (emoji-32)...")
    try:
        dataset = EmojiDataset(
            dataset_name="junyeong-nero/emoji-32",
            split="train",
            streaming=True,
        )
        print(f"Dataset: {dataset}")

        sample = dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample caption: {sample['caption']}")

        for i in [1, 5, 10, 50, 100]:
            sample = dataset[i]
            print(f"[{i}] {sample['caption']}")

        print("\n✓ Dataset loading test passed!")
    except ImportError:
        print("datasets library not installed. Install with: pip install datasets")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
