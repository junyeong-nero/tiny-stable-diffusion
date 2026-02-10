#!/bin/bash
# VAE Reconstruction Quality Evaluation Script
# Evaluates how well the VAE can reconstruct images using PSNR, SSIM, MSE, and LPIPS metrics.
#
# Usage:
#   ./scripts/evaluate-vae.sh [options]
#
# Options:
#   --input-dir DIR     Evaluate images from a local directory
#   --dataset NAME      Evaluate on a HuggingFace dataset
#   --checkpoint PATH   VAE checkpoint path (default: auto-detect latest)
#   --max-samples N     Maximum samples to evaluate (default: 100)
#   --save PATH         Save results to JSON file
#   --no-lpips          Disable LPIPS metric (faster, no extra dependency)
#
# Examples:
#   # Evaluate on local images
#   ./scripts/evaluate-vae.sh --input-dir results/original
#
#   # Evaluate on HuggingFace dataset
#   ./scripts/evaluate-vae.sh --dataset reach-vb/pokemon-blip-captions --max-samples 200
#
#   # Use specific checkpoint and save results
#   ./scripts/evaluate-vae.sh --input-dir results/original \
#       --checkpoint checkpoints/vae_e50.pt \
#       --save results/vae_eval.json
#
# Metrics:
#   - PSNR (Peak Signal-to-Noise Ratio): Higher is better, typical 20-40 dB
#   - SSIM (Structural Similarity): 0-1, higher is better
#   - MSE (Mean Squared Error): Lower is better
#   - LPIPS (Learned Perceptual): Lower is better (requires lpips package)
#
# Environment variables:
#   VAE_CHECKPOINT - Default checkpoint path (default: auto-detect)

set -euo pipefail

# Default values
VAE_CHECKPOINT="${VAE_CHECKPOINT:-}"
INPUT_DIR=""
DATASET=""
MAX_SAMPLES="100"
SAVE_PATH=""
NO_LPIPS=""
SPLIT="train"
IMAGE_FIELD="image"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --checkpoint)
            VAE_CHECKPOINT="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --save)
            SAVE_PATH="$2"
            shift 2
            ;;
        --no-lpips)
            NO_LPIPS="--no-lpips"
            shift
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --image-field)
            IMAGE_FIELD="$2"
            shift 2
            ;;
        -h|--help)
            head -40 "$0" | tail -38
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ -z "$INPUT_DIR" ] && [ -z "$DATASET" ]; then
    echo "Error: Either --input-dir or --dataset is required"
    echo ""
    echo "Usage:"
    echo "  ./scripts/evaluate-vae.sh --input-dir results/original"
    echo "  ./scripts/evaluate-vae.sh --dataset reach-vb/pokemon-blip-captions"
    exit 1
fi

echo "========================================"
echo "VAE Reconstruction Quality Evaluation"
echo "========================================"

# Build command
CMD=(uv run python -m src.evaluation.vae_evaluator)

if [ -n "$INPUT_DIR" ]; then
    echo "Input directory: $INPUT_DIR"
    CMD+=(--input-dir "$INPUT_DIR")
fi

if [ -n "$DATASET" ]; then
    echo "Dataset: $DATASET"
    CMD+=(--dataset "$DATASET" --split "$SPLIT" --image-field "$IMAGE_FIELD")
fi

if [ -n "$VAE_CHECKPOINT" ]; then
    echo "Checkpoint: $VAE_CHECKPOINT"
    CMD+=(--checkpoint "$VAE_CHECKPOINT")
else
    echo "Checkpoint: auto-detect"
fi

echo "Max samples: $MAX_SAMPLES"
CMD+=(--max-samples "$MAX_SAMPLES")

if [ -n "$SAVE_PATH" ]; then
    echo "Save results to: $SAVE_PATH"
    CMD+=(--save "$SAVE_PATH")
fi

if [ -n "$NO_LPIPS" ]; then
    echo "LPIPS: disabled"
    CMD+=("$NO_LPIPS")
fi

echo "========================================"
echo ""

echo "Running: ${CMD[*]}"
"${CMD[@]}"
