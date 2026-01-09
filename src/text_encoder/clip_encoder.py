"""CLIP Text Encoder for text conditioning.

Uses OpenAI's CLIP model to encode text prompts into embeddings
that condition the diffusion model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import clip
except ImportError:
    raise ImportError("Please install clip: pip install git+https://github.com/openai/CLIP.git")


class CLIPTextEncoder(nn.Module):
    """Wrapper for OpenAI CLIP text encoder.

    Encodes text prompts into embeddings for diffusion conditioning.
    Returns pooled text embeddings (B, D).
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # Load CLIP model
        self.model, _ = clip.load(model_name, device=device)

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.eval()

    @property
    def embedding_dim(self) -> int:
        """Get CLIP embedding dimension."""
        # ViT-B/32 and ViT-B/16 have 512 dim, ViT-L/14 has 768 dim
        return self.model.text_projection.shape[1]

    @property
    def device(self) -> torch.device:
        """Get device."""
        return torch.device(self._device)

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype."""
        return self.model.dtype

    def to(self, device: str | torch.device) -> "CLIPTextEncoder":
        """Move model to device."""
        if isinstance(device, torch.device):
            device = str(device)
        self._device = device
        self.model = self.model.to(device)
        return self

    @torch.no_grad()
    def forward(self, text: str | list[str]) -> torch.Tensor:
        """Encode text to embeddings.

        Args:
            text: Single string or list of strings

        Returns:
            Text embeddings (B, D) - pooled/mean-pooled representation
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize and encode
        text_tokens = clip.tokenize(text, truncate=True).to(self._device)
        text_encoding = self.model.encode_text(text_tokens)

        # Normalize (CLIP embeddings are typically normalized)
        text_encoding = text_encoding / text_encoding.norm(dim=-1, keepdim=True)

        # Convert to float32 for compatibility with DiT
        text_encoding = text_encoding.float()

        return text_encoding

    @torch.no_grad()
    def encode(
        self,
        text: str | list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode text in batches.

        Args:
            text: Text prompts to encode
            batch_size: Batch size for encoding

        Returns:
            Text embeddings (B, D)
        """
        if isinstance(text, str):
            text = [text]

        all_embeddings = []
        for i in range(0, len(text), batch_size):
            batch = text[i : i + batch_size]
            embeddings = self.forward(batch)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def train(self, mode: bool = False) -> "CLIPTextEncoder":
        """Set training mode (always False, model is frozen)."""
        super().train(mode)
        self.model.eval()
        return self

    def eval(self) -> "CLIPTextEncoder":
        """Set to evaluation mode."""
        super().eval()
        self.model.eval()
        return self

    def __repr__(self) -> str:
        return f"CLIPTextEncoder(model_name={self.model_name}, embedding_dim={self.embedding_dim})"


if __name__ == "__main__":
    # Quick test
    encoder = CLIPTextEncoder()

    # Test single prompt
    embedding = encoder("a cute robot")
    print(f"Single prompt: embedding={embedding.shape}")

    # Test batch
    prompts = ["a robot", "a cat", "an apple", "a ghost"]
    embeddings = encoder.encode(prompts)
    print(f"Batch prompts: embeddings={embeddings.shape}")

    print(f"\n{encoder}")
