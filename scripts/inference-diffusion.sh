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
#   SCALING_FACTOR=0.4869  VAE scaling factor
#
# Multiple images example:
#   NUM_SAMPLES=4 ./scripts/inference-diffusion.sh "a cute tabby kitten"

set -euo pipefail

DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-checkpoints/diffusion.pt}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-checkpoints/vae.pt}"
PROMPT="${1:-a photo of a cat}"
OUTPUT="${2:-output.png}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-7.5}"
SEED="${SEED:-}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SCALING_FACTOR="${SCALING_FACTOR:-0.4869}"

echo "Diffusion Model Inference"
echo "========================="
echo "Diffusion checkpoint: $DIFFUSION_CHECKPOINT"
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"
echo "Steps: $STEPS"
echo "Guidance scale: $GUIDANCE"
echo "Number of samples: $NUM_SAMPLES"
echo "Scaling factor: $SCALING_FACTOR"
if [ -n "$SEED" ]; then
    echo "Seed: $SEED"
fi
echo ""

CMD=(
    uv run main.py --generate
    --prompt "$PROMPT"
    --checkpoint "$DIFFUSION_CHECKPOINT"
    --vae-checkpoint "$VAE_CHECKPOINT"
    --steps "$STEPS"
    --guidance "$GUIDANCE"
    --scaling-factor "$SCALING_FACTOR"
    --num-samples "$NUM_SAMPLES"
)

if [ "$NUM_SAMPLES" -gt 1 ]; then
    OUTPUT_DIR="${OUTPUT%.png}"
    echo "Saving $NUM_SAMPLES samples to: $OUTPUT_DIR/"
else
    CMD+=(--output "$OUTPUT")
fi

if [ -n "$SEED" ]; then
    CMD+=(--seed "$SEED")
fi

echo ""
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"

if [ "$NUM_SAMPLES" -gt 1 ]; then
    mkdir -p "$OUTPUT_DIR"
    for ((i = 0; i < NUM_SAMPLES; i++)); do
        src="output_${i}.png"
        if [ -f "$src" ]; then
            mv -f "$src" "$OUTPUT_DIR/$src"
            echo "Saved: $OUTPUT_DIR/$src"
        fi
    done
fi
