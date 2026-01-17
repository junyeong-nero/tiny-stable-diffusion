#!/bin/bash
# Diffusion Model Inference Script
# Generates images from text prompts using trained diffusion model
#
# Usage:
#   ./scripts/inference-diffusion.sh [prompt] [output]
#
# Examples:
#   ./scripts/inference-diffusion.sh "a Siamese cat with blue eyes"
#   ./scripts/inference-diffusion.sh "a golden retriever" "dog.png"
#
# Environment variables:
#   NUM_SAMPLES=4      Generate multiple images (default: 1)
#   STEPS=50           Number of diffusion steps
#   GUIDANCE=7.5       CFG guidance scale
#   SEED=42            Random seed for reproducibility
#
# Multiple images example:
#   NUM_SAMPLES=4 ./scripts/inference-diffusion.sh "a cute tabby kitten"

set -e

DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-checkpoints/diffusion.pt}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-checkpoints/vae.pt}"
PROMPT="${1:-a photo of a cat}"
OUTPUT="${2:-output.png}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-7.5}"
SEED="${SEED:-}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"

echo "Diffusion Model Inference"
echo "========================="
echo "Diffusion checkpoint: $DIFFUSION_CHECKPOINT"
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"
echo "Steps: $STEPS"
echo "Guidance scale: $GUIDANCE"
echo "Number of samples: $NUM_SAMPLES"
if [ -n "$SEED" ]; then
    echo "Seed: $SEED"
fi
echo ""

# Build command
CMD="uv run main.py --generate"
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --checkpoint \"$DIFFUSION_CHECKPOINT\""
CMD="$CMD --vae-checkpoint \"$VAE_CHECKPOINT\""
CMD="$CMD --steps $STEPS"
CMD="$CMD --guidance $GUIDANCE"
CMD="$CMD --num-samples $NUM_SAMPLES"

# Set output based on number of samples
if [ "$NUM_SAMPLES" -gt 1 ]; then
    # For multiple samples, output to directory
    OUTPUT_DIR="${OUTPUT%.png}"
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
    echo "Saving $NUM_SAMPLES samples to: $OUTPUT_DIR/"
else
    CMD="$CMD --output \"$OUTPUT\""
fi

if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD
