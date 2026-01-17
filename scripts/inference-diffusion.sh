#!/bin/bash
# Diffusion Model Inference Script
# Generates images from text prompts using trained diffusion model
#
# Usage:
#   ./scripts/inference-diffusion.sh [prompt] [output]
#
# Examples:
#   ./scripts/inference-diffusion.sh "a Siamese cat with blue eyes"
#   ./scripts/inference-diffusion.sh "a robot with blue eyes" "robot.png"

set -e

DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-checkpoints/diffusion.pt}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-checkpoints/vae_e40.pt}"
PROMPT="${1:-a photo of a cat}"
OUTPUT="${2:-output.png}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-7.5}"
SEED="${SEED:-}"

echo "Diffusion Model Inference"
echo "========================="
echo "Diffusion checkpoint: $DIFFUSION_CHECKPOINT"
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"
echo "Steps: $STEPS"
echo "Guidance scale: $GUIDANCE"
if [ -n "$SEED" ]; then
    echo "Seed: $SEED"
fi
echo ""

# Build command
CMD="uv run main.py --generate"
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --checkpoint \"$DIFFUSION_CHECKPOINT\""
CMD="$CMD --vae-checkpoint \"$VAE_CHECKPOINT\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --steps $STEPS"
CMD="$CMD --guidance $GUIDANCE"

if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

echo "Running: $CMD"
echo ""

eval $CMD
