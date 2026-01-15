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

    OUTPUT_DIR="samples/vae_reconstructed"
    mkdir -p "$OUTPUT_DIR"

    count=0
    for img in samples/original/sample_*.png; do
        if [ -f "$img" ]; then
            filename=$(basename "$img")
            output="$OUTPUT_DIR/$filename"
            echo "[$((count + 1))] Processing: $filename"
            uv run main.py --reconstruct-vae \
                --input "$img" \
                --output "$output" \
                --vae-checkpoint "$VAE_CHECKPOINT"
            count=$((count + 1))
        fi
    done

    echo ""
    echo "Done! Processed $count images."
    echo "Reconstructions saved to: $OUTPUT_DIR/"
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