#!/bin/bash
# Motion Module Training Script
# Trains the temporal attention layers for GIF generation while keeping the base MMDiT frozen.
#
# Usage:
#   ./scripts/train-motion.sh
#   ./scripts/train-motion.sh --epochs 200 --batch-size 4
#
# Default configuration (from config.yaml or main.py defaults):
#   - Epochs: 100
#   - Batch Size: 8
#   - Gradient Accumulation: 4

set -euo pipefail

echo "Starting Motion Module Training..."

# Ensure checkpoints directory exists
mkdir -p checkpoints

CHECKPOINT="${CHECKPOINT:-checkpoints/motion.pt}"

uv run main.py --train-motion \
    --wandb \
    --checkpoint "$CHECKPOINT" \
    "$@"
