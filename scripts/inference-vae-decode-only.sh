#!/bin/bash
# VAE Decode-Only Inference Script
# Generates images from random latent vectors using only the VAE decoder
# Samples from the learned latent distribution (mean/std computed from reference images)
#
# Usage:
#   ./scripts/inference-vae-decode-only.sh                    # Generate 16 images as grid
#   ./scripts/inference-vae-decode-only.sh --num 25           # Generate 25 images
#   ./scripts/inference-vae-decode-only.sh --seed 42          # Use specific seed
#   ./scripts/inference-vae-decode-only.sh --scale 0.5        # Use smaller latent scale
#   ./scripts/inference-vae-decode-only.sh --output-dir dir/  # Save individual images
#   ./scripts/inference-vae-decode-only.sh --ref-dir path/    # Use custom reference images
#
# Environment variables:
#   VAE_CHECKPOINT - Path to VAE checkpoint (default: checkpoints/vae_e30.pt)
#   REFERENCE_DIR  - Path to reference images for latent statistics (default: samples/original)

set -e

VAE_CHECKPOINT="${VAE_CHECKPOINT:-checkpoints/vae.pt}"
REFERENCE_DIR="${REFERENCE_DIR:-samples/original}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
SEED=""
LATENT_SCALE="1.0"
OUTPUT="samples/random_latent_grid.png"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num|-n)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --seed|-s)
            SEED="$2"
            shift 2
            ;;
        --scale)
            LATENT_SCALE="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT="$2/"
            shift 2
            ;;
        --ref-dir|--reference-dir)
            REFERENCE_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "VAE Decode-Only Inference"
            echo ""
            echo "Generates images from random latent vectors using only the VAE decoder."
            echo "Samples from the learned latent distribution (mean/std computed from reference images)."
            echo ""
            echo "Usage:"
            echo "  ./scripts/inference-vae-decode-only.sh [options]"
            echo ""
            echo "Options:"
            echo "  --num, -n NUM       Number of images to generate (default: 16)"
            echo "  --seed, -s SEED     Random seed for reproducibility"
            echo "  --scale SCALE       Latent scale factor (default: 1.0)"
            echo "  --output, -o PATH   Output path for grid image (default: samples/random_latent_grid.png)"
            echo "  --output-dir DIR    Save individual images to directory instead of grid"
            echo "  --ref-dir DIR       Reference images directory for latent statistics (default: samples/original)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  VAE_CHECKPOINT      Path to VAE checkpoint (default: checkpoints/vae_e30.pt)"
            echo "  REFERENCE_DIR       Reference images directory (default: samples/original)"
            echo ""
            echo "Examples:"
            echo "  ./scripts/inference-vae-decode-only.sh"
            echo "  ./scripts/inference-vae-decode-only.sh --num 25 --seed 42"
            echo "  ./scripts/inference-vae-decode-only.sh --scale 0.5 --output test.png"
            echo "  ./scripts/inference-vae-decode-only.sh --output-dir samples/random_latents/"
            echo "  ./scripts/inference-vae-decode-only.sh --ref-dir my_images/"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "VAE Decode-Only Inference"
echo "========================="
echo "Checkpoint: $VAE_CHECKPOINT"
echo "Reference dir: $REFERENCE_DIR"
echo "Num samples: $NUM_SAMPLES"
echo "Latent scale: $LATENT_SCALE"
echo "Output: $OUTPUT"
if [ -n "$SEED" ]; then
    echo "Seed: $SEED"
fi
echo ""

# Build command
CMD="uv run main.py --decode-random \
    --vae-checkpoint \"$VAE_CHECKPOINT\" \
    --reference-dir \"$REFERENCE_DIR\" \
    --num-samples $NUM_SAMPLES \
    --latent-scale $LATENT_SCALE \
    --output \"$OUTPUT\""

if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

# Execute
eval $CMD
