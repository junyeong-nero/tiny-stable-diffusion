"""Data loading and preprocessing for emoji dataset.

Uses the junyeong-nero/emoji-32 dataset from Hugging Face.
Images are already 32x32 RGB, no resizing needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        cache_dir: str = "~/.cache/pixmoji",
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
            raise ImportError(
                "datasets library not found. Install with: pip install datasets"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
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
                result.append({
                    "image": image,
                    "caption": sample.get(self.caption_field, ""),
                })
            return result

    def __repr__(self) -> str:
        return (
            f"EmojiDataset("
            f"dataset={self.dataset_name}, "
            f"split={self.split}, "
            f"num_samples={self.size})"
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
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
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
        return (
            f"LocalEmojiDataset("
            f"data_dir={self.data_dir}, "
            f"num_samples={len(self)})"
        )


def get_dataset(
    dataset_source: str = "huggingface",
    **kwargs,
) -> Dataset:
    if dataset_source == "huggingface":
        return EmojiDataset(**kwargs)
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
