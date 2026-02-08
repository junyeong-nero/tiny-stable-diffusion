#!/bin/bash
# Diffusion Model Sampling Benchmark Script
# Measures generation speed, throughput, and memory usage.
#
# Usage:
#   ./scripts/benchmark-diffusion.sh [options]
#
# Options:
#   --checkpoint PATH       Diffusion checkpoint (default: auto-detect)
#   --vae-checkpoint PATH   VAE checkpoint (default: checkpoints/vae.pt)
#   --prompt TEXT            Prompt for benchmarking
#   --steps LIST            Comma-separated step counts (default: 10,25,50,100)
#   --batch-sizes LIST      Comma-separated batch sizes (default: 1,4,8)
#   --single                Single-run mode (use --steps and --batch-size values)
#   --batch-size N          Batch size for single-run mode (default: 1)
#
# Examples:
#   # Full sweep benchmark
#   ./scripts/benchmark-diffusion.sh
#
#   # Custom sweep
#   ./scripts/benchmark-diffusion.sh --steps 25,50 --batch-sizes 1,2,4
#
#   # Single configuration benchmark
#   ./scripts/benchmark-diffusion.sh --single --steps 50 --batch-size 4
#
#   # Custom prompt
#   ./scripts/benchmark-diffusion.sh --prompt "a detailed landscape painting"
#
# Environment variables:
#   DIFFUSION_CHECKPOINT - Default diffusion checkpoint path
#   VAE_CHECKPOINT       - Default VAE checkpoint path

set -euo pipefail

# Default values
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-checkpoints/vae.pt}"
PROMPT="a cat sitting on a couch"
STEPS_LIST="10,25,50,100"
BATCH_SIZES="1,4,8"
SINGLE_MODE=""
SINGLE_STEPS="50"
SINGLE_BATCH="1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            DIFFUSION_CHECKPOINT="$2"
            shift 2
            ;;
        --vae-checkpoint)
            VAE_CHECKPOINT="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --steps)
            STEPS_LIST="$2"
            SINGLE_STEPS="$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --batch-size)
            SINGLE_BATCH="$2"
            shift 2
            ;;
        --single)
            SINGLE_MODE="true"
            shift
            ;;
        -h|--help)
            head -38 "$0" | tail -36
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Diffusion Model Benchmark"
echo "========================================"

# Build command
CMD=(
    uv run main.py --benchmark
    --vae-checkpoint "$VAE_CHECKPOINT"
    --prompt "$PROMPT"
)

if [ -n "$DIFFUSION_CHECKPOINT" ]; then
    echo "Diffusion checkpoint: $DIFFUSION_CHECKPOINT"
    CMD+=(--checkpoint "$DIFFUSION_CHECKPOINT")
else
    echo "Diffusion checkpoint: auto-detect"
fi

echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "Prompt: $PROMPT"

if [ -n "$SINGLE_MODE" ]; then
    echo "Mode: single run"
    echo "Steps: $SINGLE_STEPS"
    echo "Batch size: $SINGLE_BATCH"
    CMD+=(--steps "$SINGLE_STEPS" --batch-size "$SINGLE_BATCH")
else
    echo "Mode: sweep"
    echo "Steps: $STEPS_LIST"
    echo "Batch sizes: $BATCH_SIZES"
    CMD+=(--benchmark-steps "$STEPS_LIST" --benchmark-batch-sizes "$BATCH_SIZES")
fi

echo "========================================"
echo ""

echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"
