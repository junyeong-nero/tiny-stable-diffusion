#!/bin/bash
# Generate emoji images using trained PixMoji-Diffusion model

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎨 PixMoji-Diffusion Inference${NC}"
echo "========================================"

# Default values
PROMPT="a cute robot"
NUM_SAMPLES=4
CHECKPOINT="checkpoints/model_final.pt"
GUIDANCE_SCALE=7.5
STEPS=50
OUTPUT_DIR="generated"
SEED=42
UPS_SCALE=256

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --upscale)
            UPSCALE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prompt TEXT         Text prompt for generation (default: 'a cute robot')"
            echo "  --num-samples N       Number of images to generate (default: 4)"
            echo "  --checkpoint PATH     Path to model checkpoint (default: checkpoints/model_final.pt)"
            echo "  --guidance-scale N    CFG scale (default: 7.5, range: 4-15)"
            echo "  --steps N             Sampling steps (default: 50, lower = faster)"
            echo "  --output-dir DIR      Output directory (default: generated)"
            echo "  --seed N              Random seed (default: 42)"
            echo "  --upscale N           Upscale size (default: 256)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --prompt 'a robot'"
            echo "  $0 --prompt 'a smiling face' --num-samples 8"
            echo "  $0 --checkpoint checkpoints/checkpoint_epoch_50.pt --steps 30"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Configuration:${NC}"
echo "  Prompt:          '${PROMPT}'"
echo "  Num Samples:     ${NUM_SAMPLES}"
echo "  Checkpoint:      ${CHECKPOINT}"
echo "  Guidance Scale:  ${GUIDANCE_SCALE}"
echo "  Steps:           ${STEPS}"
echo "  Output Dir:      ${OUTPUT_DIR}"
echo "  Seed:            ${SEED}"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: ${CHECKPOINT}${NC}"
    echo ""
    echo "Please either:"
    echo "  1. Train the model first: ./scripts/train.sh"
    echo "  2. Provide a valid checkpoint path with --checkpoint"
    exit 1
fi

# Build command
CMD="uv run python src/inference/generate.py \
    --prompt \"${PROMPT}\" \
    --num-samples ${NUM_SAMPLES} \
    --checkpoint ${CHECKPOINT} \
    --guidance-scale ${GUIDANCE_SCALE} \
    --steps ${STEPS} \
    --output-dir ${OUTPUT_DIR} \
    --seed ${SEED}"

echo -e "${GREEN}Generating images...${NC}"
echo "Command: $CMD"
echo ""

# Run inference
eval $CMD

echo ""
echo -e "${GREEN}✅ Generation complete!${NC}"
echo "Images saved in: ${OUTPUT_DIR}"
