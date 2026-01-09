"""CLIP Text Encoder for text conditioning.

Uses OpenAI's CLIP model to encode text prompts into embeddings
that condition the diffusion model.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    """Wrapper for CLIP text encoder.

    Encodes text prompts into embeddings for diffusion conditioning.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        # Load tokenizer and model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

        # Move to device and set dtype
        if device is not None:
            self.model = self.model.to(device)
        if dtype is not None:
            self.model = self.model.to(dtype)

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.eval()

    @property
    def embedding_dim(self) -> int:
        """Get CLIP embedding dimension."""
        return self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        """Get device."""
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype."""
        return next(self.model.parameters()).dtype

    def forward(self, text: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text to embeddings.

        Args:
            text: Single string or list of strings

        Returns:
            Tuple of (text_embeddings, attention_mask):
                - text_embeddings: Text embeddings (B, L, D) where B=batch, L=77, D=768
                - attention_mask: Attention mask (B, L) where 1 = real token, 0 = padding
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # (B, L, D)

        return embeddings, attention_mask

    def encode(
        self,
        text: str | list[str],
        batch_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text in batches.

        Args:
            text: Text prompts to encode
            batch_size: Batch size for encoding

        Returns:
            Tuple of (all_embeddings, all_masks):
                - all_embeddings: Text embeddings (B, L, D)
                - all_masks: Attention masks (B, L) where 1 = real token, 0 = padding
        """
        if isinstance(text, str):
            text = [text]

        all_embeddings = []
        all_masks = []
        for i in range(0, len(text), batch_size):
            batch = text[i : i + batch_size]
            embeddings, mask = self.forward(batch)
            all_embeddings.append(embeddings)
            all_masks.append(mask)

        return torch.cat(all_embeddings, dim=0), torch.cat(all_masks, dim=0)

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

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "CLIPTextEncoder":
        """Load pretrained CLIP encoder from local path.

        Args:
            path: Local path to saved model
            device: Device to load on
            dtype: Dtype to use

        Returns:
            CLIPTextEncoder instance
        """
        tokenizer = CLIPTokenizer.from_pretrained(path)
        model = CLIPTextModel.from_pretrained(path)

        encoder = cls.__new__(cls)
        encoder.model_name = path
        encoder.max_length = 77
        encoder.tokenizer = tokenizer
        encoder.model = model

        if device is not None:
            encoder.model = encoder.model.to(device)
        if dtype is not None:
            encoder.model = encoder.model.to(dtype)

        for param in encoder.model.parameters():
            param.requires_grad = False

        encoder.model.eval()
        return encoder

    def save_pretrained(self, path: str) -> None:
        """Save encoder to local path.

        Args:
            path: Path to save model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def __repr__(self) -> str:
        return (
            f"CLIPTextEncoder("
            f"model_name={self.model_name}, "
            f"embedding_dim={self.embedding_dim}, "
            f"max_length={self.max_length})"
        )


if __name__ == "__main__":
    # Quick test
    encoder = CLIPTextEncoder()

    # Test single prompt
    embedding, mask = encoder("a cute robot")
    print(f"Single prompt: embedding={embedding.shape}, mask={mask.shape}")

    # Test batch
    prompts = ["a robot", "a cat", "an apple", "a ghost"]
    embeddings, masks = encoder.encode(prompts)
    print(f"Batch prompts: embeddings={embeddings.shape}, masks={masks.shape}")
    print(f"Mask example (first sample): {masks[0]}")

    print(f"\n{encoder}")
