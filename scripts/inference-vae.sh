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
#   RESULTS_DIR - Base output directory (default: results/vae)
#   INPUT_DIR - Input directory in --all mode (default: samples/original)
#   EVAL_MAX_SAMPLES - Max samples for reconstruction evaluation (default: 100)
#   NO_LPIPS=1 - Disable LPIPS metric during evaluation

set -euo pipefail

VAE_CHECKPOINT="${VAE_CHECKPOINT:-checkpoints/vae.pt}"
RESULTS_DIR="${RESULTS_DIR:-results/vae}"
INPUT_DIR="${INPUT_DIR:-samples/original}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-100}"
NO_LPIPS="${NO_LPIPS:-0}"

# Check for --all flag
if [ "${1:-}" = "--all" ]; then
    OUTPUT_DIR="$RESULTS_DIR/reconstructions"
    METRICS_JSON="$RESULTS_DIR/reconstruction_metrics.json"
    mkdir -p "$RESULTS_DIR"

    echo "VAE Batch Inference"
    echo "==================="
    echo "Checkpoint: $VAE_CHECKPOINT"
    echo "Input dir: $INPUT_DIR"
    echo "Output dir: $OUTPUT_DIR"
    echo ""

    uv run main.py --reconstruct-vae \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --vae-checkpoint "$VAE_CHECKPOINT"

    echo ""
    echo "VAE Reconstruction Evaluation"
    echo "============================="
    EVAL_CMD=(
        uv run python -m src.evaluation.vae_evaluator
        --input-dir "$INPUT_DIR"
        --checkpoint "$VAE_CHECKPOINT"
        --max-samples "$EVAL_MAX_SAMPLES"
        --save "$METRICS_JSON"
    )
    if [ "$NO_LPIPS" = "1" ]; then
        EVAL_CMD+=(--no-lpips)
    fi

    echo "Running: ${EVAL_CMD[*]}"
    "${EVAL_CMD[@]}"
else
    INPUT_IMAGE="${1:-samples/original/sample_000_cattle.png}"
    OUTPUT_IMAGE="${2:-reconstructed.png}"
    if [[ "$OUTPUT_IMAGE" == results/* ]]; then
        OUTPUT_PATH="$OUTPUT_IMAGE"
    else
        OUTPUT_PATH="$RESULTS_DIR/$OUTPUT_IMAGE"
    fi
    METRICS_JSON="$RESULTS_DIR/reconstruction_metrics_single.json"
    mkdir -p "$(dirname "$OUTPUT_PATH")"

    echo "VAE Inference"
    echo "============="
    echo "Checkpoint: $VAE_CHECKPOINT"
    echo "Input: $INPUT_IMAGE"
    echo "Output: $OUTPUT_PATH"
    echo ""

    uv run main.py --reconstruct-vae \
        --input "$INPUT_IMAGE" \
        --output "$OUTPUT_PATH" \
        --vae-checkpoint "$VAE_CHECKPOINT"

    # Evaluate reconstruction quality on the single input image.
    TMP_DIR="$(mktemp -d)"
    cp "$INPUT_IMAGE" "$TMP_DIR/$(basename "$INPUT_IMAGE")"

    EVAL_CMD=(
        uv run python -m src.evaluation.vae_evaluator
        --input-dir "$TMP_DIR"
        --checkpoint "$VAE_CHECKPOINT"
        --max-samples 1
        --save "$METRICS_JSON"
    )
    if [ "$NO_LPIPS" = "1" ]; then
        EVAL_CMD+=(--no-lpips)
    fi

    echo ""
    echo "VAE Reconstruction Evaluation (single image)"
    echo "==========================================="
    echo "Running: ${EVAL_CMD[*]}"
    "${EVAL_CMD[@]}"
    rm -rf "$TMP_DIR"
fi
