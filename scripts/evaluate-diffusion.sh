#!/bin/bash
# Diffusion Model Quantitative Evaluation Script
# Computes FID, CLIP-FID, Inception Score, and CLIPScore metrics.
#
# Usage:
#   ./scripts/evaluate-diffusion.sh [options]
#
# Options:
#   --checkpoint PATH       Diffusion checkpoint (default: auto-detect)
#   --vae-checkpoint PATH   VAE checkpoint (default: checkpoints/vae.pt)
#   --real-images-dir DIR   Directory with real images for FID
#   --dataset NAME          HuggingFace dataset for real images
#   --prompts-file PATH     JSON file with evaluation prompts
#   --prompt TEXT            Prompt(s) separated by '||'
#   --metrics LIST          Comma-separated metrics (default: fid,clip_fid,is,clip_score)
#   --num-samples N         Number of generated samples (default: 1000)
#   --steps N               Diffusion sampling steps (default: 50)
#   --guidance SCALE        CFG guidance scale (default: 7.5)
#   --seed N                Random seed (default: 42)
#
# Examples:
#   # Full evaluation with real images directory
#   ./scripts/evaluate-diffusion.sh --real-images-dir data/pokemon_64x64
#
#   # CLIPScore only (no real images needed)
#   ./scripts/evaluate-diffusion.sh --metrics clip_score --num-samples 100
#
#   # FID against HuggingFace dataset
#   ./scripts/evaluate-diffusion.sh --dataset reach-vb/pokemon-blip-captions \
#       --num-samples 500
#
#   # Custom prompts
#   ./scripts/evaluate-diffusion.sh --metrics clip_score \
#       --prompt "a cat sitting || a dog running || a bird flying"
#
# Environment variables:
#   DIFFUSION_CHECKPOINT - Default diffusion checkpoint path
#   VAE_CHECKPOINT       - Default VAE checkpoint path

set -euo pipefail

# Default values
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-checkpoints/vae.pt}"
REAL_IMAGES_DIR=""
DATASET=""
PROMPTS_FILE=""
PROMPT=""
METRICS="fid,clip_fid,is,clip_score"
NUM_SAMPLES="1000"
STEPS="50"
GUIDANCE="7.5"
SEED="42"

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
        --real-images-dir)
            REAL_IMAGES_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --prompts-file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            head -44 "$0" | tail -42
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Diffusion Model Evaluation"
echo "========================================"

# Build command
CMD=(
    uv run main.py --evaluate
    --eval-metrics "$METRICS"
    --num-eval-samples "$NUM_SAMPLES"
    --steps "$STEPS"
    --guidance "$GUIDANCE"
    --seed "$SEED"
    --vae-checkpoint "$VAE_CHECKPOINT"
)

if [ -n "$DIFFUSION_CHECKPOINT" ]; then
    echo "Diffusion checkpoint: $DIFFUSION_CHECKPOINT"
    CMD+=(--checkpoint "$DIFFUSION_CHECKPOINT")
else
    echo "Diffusion checkpoint: auto-detect"
fi

echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "Metrics: $METRICS"
echo "Num samples: $NUM_SAMPLES"
echo "Steps: $STEPS"
echo "Guidance: $GUIDANCE"
echo "Seed: $SEED"

if [ -n "$REAL_IMAGES_DIR" ]; then
    echo "Real images: $REAL_IMAGES_DIR"
    CMD+=(--real-images-dir "$REAL_IMAGES_DIR")
fi

if [ -n "$DATASET" ]; then
    echo "Dataset: $DATASET"
    CMD+=(--dataset "$DATASET")
fi

if [ -n "$PROMPTS_FILE" ]; then
    echo "Prompts file: $PROMPTS_FILE"
    CMD+=(--eval-prompts-file "$PROMPTS_FILE")
fi

if [ -n "$PROMPT" ]; then
    echo "Prompt: $PROMPT"
    CMD+=(--prompt "$PROMPT")
fi

echo "========================================"
echo ""

echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"
