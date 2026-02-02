"""Video dataset for GIF/animation generation training.

Supports video-caption datasets from HuggingFace for motion module training:
- VideoDataset: General video datasets (UCF101, WebVid, etc.)
- GIFDataset: GIF-specific datasets with shorter clips
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms

from src.data.video_transforms import (
    get_video_transforms,
    sample_frames_uniform,
    sample_frames_random,
)


class VideoDataset(Dataset):
    """Dataset for video-caption pairs from HuggingFace.

    Loads video files and extracts frames for training motion modules.
    Supports various video datasets with different formats.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split ("train", "validation", "test")
        num_frames: Number of frames to extract per video
        frame_skip: Skip every N frames (temporal subsampling)
        target_size: Target frame size (default: 64)
        transform: Optional custom transform for frames
        video_field: Name of the video field in the dataset
        caption_field: Name of the caption field in the dataset
        sampling_mode: Frame sampling mode ("uniform" or "random")
        cache_dir: Cache directory for downloaded datasets

    Example datasets:
        - sayakpaul/ucf101-subset: video="video", caption="label"
        - HuggingFaceM4/webvid: video="mp4", caption="txt"
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        num_frames: int = 16,
        frame_skip: int = 1,
        target_size: int = 64,
        transform: Callable | None = None,
        video_field: str = "video",
        caption_field: str = "label",
        sampling_mode: str = "uniform",
        cache_dir: str = "~/.cache/tiny-stable-diffusion",
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.target_size = target_size
        self.video_field = video_field
        self.caption_field = caption_field
        self.sampling_mode = sampling_mode
        self.cache_dir = Path(cache_dir).expanduser()

        # Load dataset
        try:
            from datasets import load_dataset

            print(f"Loading video dataset: {dataset_name} (split={split})")
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(self.cache_dir),
            )
            self.size = len(self.dataset)
            print(f"✓ Loaded {self.dataset_name}: {self.size} videos")

        except ImportError:
            raise ImportError(
                "datasets library not found. Install with: pip install datasets"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")

        # Set up transforms
        if transform is None:
            self.transform = get_video_transforms(
                target_size=target_size,
                use_augmentation=True,
            )
        else:
            self.transform = transform

    def _extract_frames_from_video(
        self,
        video_data,
    ) -> list[Image.Image]:
        """Extract frames from video data.

        Args:
            video_data: Video data (path, bytes, or decoded frames)

        Returns:
            List of PIL Images
        """
        frames = []

        # Handle different video formats
        if isinstance(video_data, (str, Path)):
            # Video file path
            frames = self._decode_video_file(str(video_data))
        elif isinstance(video_data, bytes):
            # Video bytes
            frames = self._decode_video_bytes(video_data)
        elif isinstance(video_data, dict):
            # HuggingFace video format with 'path' key
            if "path" in video_data:
                frames = self._decode_video_file(video_data["path"])
            elif "bytes" in video_data:
                frames = self._decode_video_bytes(video_data["bytes"])
        elif isinstance(video_data, list):
            # Already decoded frames (list of arrays/images)
            for frame in video_data:
                if isinstance(frame, Image.Image):
                    frames.append(frame)
                else:
                    import numpy as np

                    if isinstance(frame, np.ndarray):
                        frames.append(Image.fromarray(frame))

        return frames

    def _decode_video_file(self, video_path: str) -> list[Image.Image]:
        """Decode video file to frames using decord or cv2."""
        frames = []

        try:
            # Try decord first (faster)
            import decord

            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)

            # Sample frame indices
            indices = self._get_frame_indices(total_frames)

            # Extract frames
            for idx in indices:
                frame = vr[idx].asnumpy()
                frames.append(Image.fromarray(frame))

        except ImportError:
            # Fallback to OpenCV
            try:
                import cv2

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = self._get_frame_indices(total_frames)

                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame))

                cap.release()

            except ImportError:
                raise ImportError(
                    "Neither decord nor opencv-python found. "
                    "Install with: pip install decord or pip install opencv-python"
                )

        return frames

    def _decode_video_bytes(self, video_bytes: bytes) -> list[Image.Image]:
        """Decode video bytes to frames."""
        import tempfile

        # Write to temporary file and decode
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            f.write(video_bytes)
            f.flush()
            return self._decode_video_file(f.name)

    def _get_frame_indices(self, total_frames: int) -> list[int]:
        """Get frame indices based on sampling mode.

        Args:
            total_frames: Total number of frames in video

        Returns:
            List of frame indices to extract
        """
        if self.sampling_mode == "uniform":
            return sample_frames_uniform(
                total_frames,
                self.num_frames,
                self.frame_skip,
            )
        elif self.sampling_mode == "random":
            return sample_frames_random(
                total_frames,
                self.num_frames,
                self.frame_skip,
            )
        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")

    def _get_caption(self, sample: dict) -> str:
        """Extract caption from sample."""
        caption = sample.get(self.caption_field)

        if caption is None:
            # Try common caption fields
            for field in ["label", "text", "txt", "caption", "description"]:
                caption = sample.get(field)
                if caption is not None:
                    break

        if caption is None:
            caption = "a video"

        if isinstance(caption, list):
            caption = random.choice(caption)

        return str(caption)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Get a video sample.

        Returns:
            Dictionary with:
                - "frames": Tensor of shape (F, C, H, W)
                - "caption": String caption
        """
        sample = self.dataset[idx]

        # Extract video data
        video_data = sample.get(self.video_field)
        if video_data is None:
            raise ValueError(f"Video field '{self.video_field}' not found in sample")

        # Extract frames
        frames = self._extract_frames_from_video(video_data)

        if len(frames) < self.num_frames:
            # Repeat frames if not enough
            while len(frames) < self.num_frames:
                frames = frames + frames
            frames = frames[: self.num_frames]

        # Sample if we have more frames than needed
        if len(frames) > self.num_frames:
            indices = self._get_frame_indices(len(frames))
            frames = [frames[i] for i in indices]

        # Apply transforms to each frame
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames: list of (C, H, W) -> (F, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)

        # Get caption
        caption = self._get_caption(sample)

        return {"frames": frames_tensor, "caption": caption}

    def __repr__(self) -> str:
        return (
            f"VideoDataset({self.dataset_name}, "
            f"size={self.size}, "
            f"num_frames={self.num_frames})"
        )


class GIFDataset(Dataset):
    """Dataset for GIF files with caption pairs.

    Optimized for shorter clips typical in GIFs (2-5 seconds).
    Can load from local directory or HuggingFace datasets.

    Args:
        data_path: Path to directory containing GIFs or HuggingFace dataset name
        num_frames: Number of frames to extract per GIF
        target_size: Target frame size
        transform: Optional custom transform
        loop: Whether to loop short GIFs to reach num_frames
    """

    def __init__(
        self,
        data_path: str,
        num_frames: int = 16,
        target_size: int = 64,
        transform: Callable | None = None,
        loop: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self.target_size = target_size
        self.loop = loop

        # Check if local directory or HuggingFace dataset
        if self.data_path.exists() and self.data_path.is_dir():
            # Local directory with GIF files
            self.gif_files = list(self.data_path.glob("**/*.gif"))
            self.captions = self._load_captions()
            self.size = len(self.gif_files)
            self.mode = "local"
            print(f"✓ Loaded {self.size} GIFs from {self.data_path}")
        else:
            # Try as HuggingFace dataset
            self.mode = "huggingface"
            self._load_huggingface_dataset(str(data_path))

        # Set up transforms
        if transform is None:
            self.transform = get_video_transforms(
                target_size=target_size,
                use_augmentation=True,
            )
        else:
            self.transform = transform

    def _load_huggingface_dataset(self, dataset_name: str) -> None:
        """Load GIF dataset from HuggingFace."""
        from datasets import load_dataset

        print(f"Loading GIF dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name, split="train")
        self.size = len(self.dataset)
        print(f"✓ Loaded {self.size} samples")

    def _load_captions(self) -> dict[str, str]:
        """Load captions from accompanying text files or use filenames."""
        captions = {}
        for gif_path in self.gif_files:
            # Try to find caption file
            txt_path = gif_path.with_suffix(".txt")
            if txt_path.exists():
                captions[str(gif_path)] = txt_path.read_text().strip()
            else:
                # Use filename as caption
                captions[str(gif_path)] = gif_path.stem.replace("_", " ")
        return captions

    def _extract_gif_frames(self, gif_path: Path) -> list[Image.Image]:
        """Extract frames from GIF file."""
        frames = []
        with Image.open(gif_path) as gif:
            try:
                while True:
                    frame = gif.copy().convert("RGB")
                    frames.append(frame)
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
        return frames

    def _process_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        """Process frames to match target num_frames."""
        if len(frames) == 0:
            raise ValueError("No frames in GIF")

        if len(frames) < self.num_frames:
            if self.loop:
                # Loop frames to reach target
                while len(frames) < self.num_frames:
                    frames = frames + frames
            else:
                # Repeat last frame
                while len(frames) < self.num_frames:
                    frames.append(frames[-1])

        if len(frames) > self.num_frames:
            # Uniform sampling
            indices = sample_frames_uniform(len(frames), self.num_frames, 1)
            frames = [frames[i] for i in indices]

        return frames[: self.num_frames]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Get a GIF sample."""
        if self.mode == "local":
            gif_path = self.gif_files[idx]
            frames = self._extract_gif_frames(gif_path)
            caption = self.captions.get(str(gif_path), "a gif")
        else:
            sample = self.dataset[idx]
            # Handle HuggingFace GIF format
            gif_data = sample.get("gif") or sample.get("image") or sample.get("video")
            if isinstance(gif_data, dict) and "bytes" in gif_data:
                gif_bytes = gif_data["bytes"]
                frames = self._extract_gif_bytes(gif_bytes)
            elif isinstance(gif_data, Image.Image):
                frames = self._extract_gif_from_image(gif_data)
            else:
                raise ValueError(f"Unsupported GIF format: {type(gif_data)}")
            caption = sample.get("caption", sample.get("text", "a gif"))

        # Process frames
        frames = self._process_frames(frames)

        # Apply transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames_tensor = torch.stack(frames, dim=0)

        return {"frames": frames_tensor, "caption": str(caption)}

    def _extract_gif_bytes(self, gif_bytes: bytes) -> list[Image.Image]:
        """Extract frames from GIF bytes."""
        frames = []
        with Image.open(io.BytesIO(gif_bytes)) as gif:
            try:
                while True:
                    frame = gif.copy().convert("RGB")
                    frames.append(frame)
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
        return frames

    def _extract_gif_from_image(self, gif_image: Image.Image) -> list[Image.Image]:
        """Extract frames from PIL Image (animated GIF)."""
        frames = []
        try:
            while True:
                frame = gif_image.copy().convert("RGB")
                frames.append(frame)
                gif_image.seek(gif_image.tell() + 1)
        except EOFError:
            pass
        return frames if frames else [gif_image.convert("RGB")]

    def __repr__(self) -> str:
        return f"GIFDataset({self.data_path}, size={self.size}, num_frames={self.num_frames})"


class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset for testing motion modules.

    Generates simple animated patterns (moving shapes, gradients, etc.)
    useful for debugging and quick iteration without real video data.

    Args:
        size: Number of synthetic videos to generate
        num_frames: Frames per video
        image_size: Frame resolution
        pattern: Animation pattern ("moving_circle", "gradient", "noise")
    """

    def __init__(
        self,
        size: int = 1000,
        num_frames: int = 16,
        image_size: int = 64,
        pattern: str = "moving_circle",
    ) -> None:
        super().__init__()
        self.size = size
        self.num_frames = num_frames
        self.image_size = image_size
        self.pattern = pattern

        # Pre-generate seeds for reproducibility
        self.seeds = [random.randint(0, 2**32 - 1) for _ in range(size)]

        print(f"✓ Created SyntheticVideoDataset: {size} videos, pattern={pattern}")

    def _generate_moving_circle(self, seed: int) -> torch.Tensor:
        """Generate video of a moving circle."""
        random.seed(seed)
        torch.manual_seed(seed)

        frames = []
        size = self.image_size

        # Random circle properties
        radius = random.randint(size // 8, size // 4)
        color = torch.rand(3)

        # Random movement
        start_x = random.randint(radius, size - radius)
        start_y = random.randint(radius, size - radius)
        dx = random.uniform(-2, 2)
        dy = random.uniform(-2, 2)

        for f in range(self.num_frames):
            # Create frame
            frame = torch.zeros(3, size, size)

            # Calculate circle position
            cx = int(start_x + dx * f) % size
            cy = int(start_y + dy * f) % size

            # Draw circle
            y, x = torch.meshgrid(
                torch.arange(size), torch.arange(size), indexing="ij"
            )
            mask = ((x - cx) ** 2 + (y - cy) ** 2) < radius**2

            for c in range(3):
                frame[c][mask] = color[c]

            frames.append(frame)

        return torch.stack(frames, dim=0)  # (F, C, H, W)

    def _generate_gradient(self, seed: int) -> torch.Tensor:
        """Generate video of shifting color gradient."""
        random.seed(seed)
        torch.manual_seed(seed)

        frames = []
        size = self.image_size

        # Random gradient direction
        angle = random.uniform(0, 2 * 3.14159)
        speed = random.uniform(0.05, 0.2)

        for f in range(self.num_frames):
            frame = torch.zeros(3, size, size)

            # Create gradient
            y, x = torch.meshgrid(
                torch.linspace(0, 1, size),
                torch.linspace(0, 1, size),
                indexing="ij",
            )

            # Shift gradient over time
            offset = f * speed
            grad = (
                torch.sin(x * 3.14159 * 2 + offset) * 0.5
                + torch.cos(y * 3.14159 * 2 + offset) * 0.5
            )
            grad = (grad + 1) / 2  # Normalize to [0, 1]

            # Apply different phases to each channel
            frame[0] = grad
            frame[1] = torch.roll(grad, shifts=size // 3, dims=0)
            frame[2] = torch.roll(grad, shifts=size // 3, dims=1)

            frames.append(frame)

        return torch.stack(frames, dim=0)

    def _generate_noise(self, seed: int) -> torch.Tensor:
        """Generate video of evolving noise."""
        torch.manual_seed(seed)

        frames = []
        size = self.image_size

        # Start with random noise
        noise = torch.rand(3, size, size)

        for f in range(self.num_frames):
            # Slowly evolve noise
            noise = noise * 0.95 + torch.rand(3, size, size) * 0.05
            frames.append(noise.clone())

        return torch.stack(frames, dim=0)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Generate a synthetic video sample."""
        seed = self.seeds[idx]

        if self.pattern == "moving_circle":
            frames = self._generate_moving_circle(seed)
        elif self.pattern == "gradient":
            frames = self._generate_gradient(seed)
        elif self.pattern == "noise":
            frames = self._generate_noise(seed)
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

        # Normalize to [-1, 1]
        frames = frames * 2 - 1

        return {
            "frames": frames,
            "caption": f"synthetic {self.pattern} animation",
        }

    def __repr__(self) -> str:
        return f"SyntheticVideoDataset(size={self.size}, pattern={self.pattern})"


def create_video_dataloader(
    dataset: Dataset | IterableDataset,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for video datasets.

    Args:
        dataset: Video dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (ignored for IterableDataset)
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Configured DataLoader
    """
    from torch.utils.data import DataLoader

    is_iterable = isinstance(dataset, IterableDataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not is_iterable else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
