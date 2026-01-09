#!/usr/bin/env python3
"""Convert pretrained checkpoint for fine-tuning.

This script helps with two-stage training:
1. Stage 1: Pretraining on CIFAR-100
2. Stage 2: Fine-tuning on emoji dataset

The script can:
- Load a pretrained checkpoint
- Optionally reset cross-attention weights for fine-tuning
- Optionally reset final layer weights
- Save the converted checkpoint

Usage:
    # Convert pretrained checkpoint for fine-tuning (keep cross-attention weights)
    python scripts/convert_pretrain_to_finetune.py \
        --input checkpoints/pretrain_cifar100.pt \
        --output checkpoints/finetune.pt

    # Convert and reset cross-attention weights
    python scripts/convert_pretrain_to_finetune.py \
        --input checkpoints/pretrain_cifar100.pt \
        --output checkpoints/finetune.pt \
        --reset-cross-attn

    # Convert and reset both cross-attention and final layer
    python scripts/convert_pretrain_to_finetune.py \
        --input checkpoints/pretrain_cifar100.pt \
        --output checkpoints/finetune.pt \
        --reset-cross-attn \
        --reset-final-layer
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def convert_checkpoint(
    input_path: str,
    output_path: str,
    reset_cross_attn: bool = False,
    reset_final_layer: bool = False,
    reset_ada_ln: bool = False,
) -> None:
    """Convert pretrained checkpoint for fine-tuning.

    Args:
        input_path: Path to pretrained checkpoint
        output_path: Path to save converted checkpoint
        reset_cross_attn: Reset cross-attention weights
        reset_final_layer: Reset final layer weights
        reset_ada_ln: Reset AdaLN-Zero parameters
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")

    # Show checkpoint info
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")

    # Get model state dict
    model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
    print(f"  Model parameters: {sum(p.numel() for p in model_state.values()):,}")

    # Reset layers
    reset_count = 0
    layers_reset = []

    for name, param in model_state.items():
        should_reset = False

        # Check if this is a cross-attention parameter
        if reset_cross_attn and "cross_attn" in name:
            should_reset = True
            layers_reset.append("cross_attention")

        # Check if this is the final layer
        elif reset_final_layer and "final_layer" in name:
            should_reset = True
            layers_reset.append("final_layer")

        # Check if this is an AdaLN-Zero parameter
        elif reset_ada_ln and "ada_ln_zero" in name:
            should_reset = True
            layers_reset.append("ada_ln_zero")

        if should_reset:
            # Reset to small random values or zeros
            if "norm" in name or "bias" in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.xavier_uniform_(param)
            reset_count += 1

    print(f"\nReset {reset_count} parameters:")
    for layer_type in set(layers_reset):
        count = layers_reset.count(layer_type)
        print(f"  - {layer_type}: {count} parameters")

    # Add metadata
    checkpoint["is_finetuned"] = True
    checkpoint["finetune_info"] = {
        "reset_cross_attn": reset_cross_attn,
        "reset_final_layer": reset_final_layer,
        "reset_ada_ln": reset_ada_ln,
    }

    # Save converted checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"\n✓ Saved converted checkpoint: {output_path}")

    # Show summary
    print("\n" + "=" * 60)
    print("Checkpoint Conversion Summary")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Reset cross-attention: {reset_cross_attn}")
    print(f"Reset final layer: {reset_final_layer}")
    print(f"Reset AdaLN-Zero: {reset_ada_ln}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pretrained checkpoint for fine-tuning")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to pretrained checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save converted checkpoint",
    )
    parser.add_argument(
        "--reset-cross-attn",
        action="store_true",
        default=False,
        help="Reset cross-attention weights",
    )
    parser.add_argument(
        "--no-reset-cross-attn",
        action="store_false",
        dest="reset_cross_attn",
        help="Don't reset cross-attention weights",
    )
    parser.add_argument(
        "--reset-final-layer",
        action="store_true",
        default=False,
        help="Reset final layer weights",
    )
    parser.add_argument(
        "--reset-ada-ln",
        action="store_true",
        default=False,
        help="Reset AdaLN-Zero parameters",
    )
    args = parser.parse_args()

    convert_checkpoint(
        input_path=args.input,
        output_path=args.output,
        reset_cross_attn=args.reset_cross_attn,
        reset_final_layer=args.reset_final_layer,
        reset_ada_ln=args.reset_ada_ln,
    )


if __name__ == "__main__":
    main()
