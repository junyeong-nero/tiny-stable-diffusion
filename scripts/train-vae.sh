#!/bin/bash
# VAE Training Script
#
# Usage:
#   ./scripts/train-vae.sh [extra args]
# Examples:
#   ./scripts/train-vae.sh
#   ./scripts/train-vae.sh --epochs 50 --batch-size 128

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-./checkpoints/vae.pt}"

mkdir -p checkpoints

uv run main.py --train-vae \
    --wandb \
    --checkpoint "$CHECKPOINT" \
    --resume \
    "$@"
