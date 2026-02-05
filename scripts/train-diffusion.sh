#!/bin/bash
# Diffusion Training Script
#
# Usage:
#   ./scripts/train-diffusion.sh [extra args]
# Examples:
#   ./scripts/train-diffusion.sh
#   ./scripts/train-diffusion.sh --epochs 200 --batch-size 64

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-./checkpoints/diffusion.pt}"

mkdir -p checkpoints

uv run main.py --train-diffusion \
    --wandb \
    --checkpoint "$CHECKPOINT" \
    --resume \
    "$@"
