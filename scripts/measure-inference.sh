#!/bin/bash
# Diffusion inference profiling script
# Measures VRAM usage and inference latency over repeated runs.
#
# Usage:
#   ./scripts/measure-inference.sh [--checkpoint PATH] [--vae-checkpoint PATH]
#                                  [--prompt TEXT] [--steps N] [--batch-size N]
#                                  [--repeats N] [--warmup-runs N] [--device DEV]
#                                  [--save PATH]

set -euo pipefail

DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-checkpoints/vae.pt}"
PROMPT="${PROMPT:-a cat sitting on a couch}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
REPEATS="${REPEATS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"
DEVICE="${DEVICE:-auto}"
SAVE_PATH="${SAVE_PATH:-results/benchmarks/inference_profile.json}"

while [[ $# -gt 0 ]]; do
    case "$1" in
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
            STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --warmup-runs)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --save)
            SAVE_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./scripts/measure-inference.sh [options]"
            echo "  --checkpoint PATH       Diffusion checkpoint path"
            echo "  --vae-checkpoint PATH   VAE checkpoint path"
            echo "  --prompt TEXT           Prompt for inference"
            echo "  --steps N               Diffusion steps (default: 50)"
            echo "  --batch-size N          Batch size (default: 1)"
            echo "  --repeats N             Measured run count (default: 5)"
            echo "  --warmup-runs N         Warmup runs for first run (default: 3)"
            echo "  --device DEV            auto/cuda/mps/cpu"
            echo "  --save PATH             JSON output path"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD=(
    uv run python -m src.evaluation.inference_profile
    --vae-checkpoint "$VAE_CHECKPOINT"
    --prompt "$PROMPT"
    --steps "$STEPS"
    --batch-size "$BATCH_SIZE"
    --repeats "$REPEATS"
    --warmup-runs "$WARMUP_RUNS"
    --device "$DEVICE"
    --save "$SAVE_PATH"
)

if [[ -n "$DIFFUSION_CHECKPOINT" ]]; then
    CMD+=(--checkpoint "$DIFFUSION_CHECKPOINT")
fi

echo "Inference Profiling"
echo "===================="
echo "Diffusion checkpoint: ${DIFFUSION_CHECKPOINT:-auto-detect}"
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "Prompt: $PROMPT"
echo "Steps: $STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Repeats: $REPEATS"
echo "Warmup runs: $WARMUP_RUNS"
echo "Device: $DEVICE"
echo "Save path: $SAVE_PATH"
echo ""
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"
