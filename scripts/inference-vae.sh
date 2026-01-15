#!/bin/bash
# VAE Inference Script
# Reconstructs images through VAE encoder/decoder to verify VAE quality
#
# Usage:
#   ./scripts/inference-vae.sh [input_image] [output_image]
#   ./scripts/inference-vae.sh --all                          # Process all sample images
#
# Examples:
#   ./scripts/inference-vae.sh samples/original/sample_000_cattle.png samples/reconstructed.png
#   ./scripts/inference-vae.sh --all
#
# Environment variables:
#   VAE_CHECKPOINT - Path to VAE checkpoint (default: checkpoints/vae_e30.pt)

set -e

VAE_CHECKPOINT="${VAE_CHECKPOINT:-checkpoints/vae_e30.pt}"

# Check for --all flag
if [ "$1" = "--all" ]; then
    echo "VAE Batch Inference"
    echo "==================="
    echo "Checkpoint: $VAE_CHECKPOINT"
    echo ""

    uv run main.py --reconstruct-vae \
        --input-dir "samples/original" \
        --output-dir "samples/vae_reconstructed" \
        --vae-checkpoint "$VAE_CHECKPOINT"
else
    INPUT_IMAGE="${1:-samples/original/sample_000_cattle.png}"
    OUTPUT_IMAGE="${2:-samples/vae_reconstructed.png}"

    echo "VAE Inference"
    echo "============="
    echo "Checkpoint: $VAE_CHECKPOINT"
    echo "Input: $INPUT_IMAGE"
    echo "Output: $OUTPUT_IMAGE"
    echo ""

    uv run main.py --reconstruct-vae \
        --input "$INPUT_IMAGE" \
        --output "$OUTPUT_IMAGE" \
        --vae-checkpoint "$VAE_CHECKPOINT"
fi