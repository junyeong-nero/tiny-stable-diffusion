#!/bin/bash
# Train text-to-emoji model

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Training text-to-emoji${NC}"
echo "========================================"

# Default values
EPOCHS=100
BATCH_SIZE=16
LEARNING_RATE=5e-4
MODEL_SIZE="S"
DATASET_NAME="junyeong-nero/emoji-32"
OUTPUT_DIR="checkpoints"
DEVICE="auto"
RESUME=""
USE_MIXED_PRECISION=false
USE_WANDB=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --no-mixed-precision)
            USE_MIXED_PRECISION=false
            shift
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs N           Number of training epochs (default: 100)"
            echo "  --batch-size N       Batch size (default: 16)"
            echo "  --learning-rate LR   Learning rate (default: 3e-4)"
            echo "  --model-size S|B|L   DiT model size (default: S)"
            echo "  --dataset-name NAME  Hugging Face dataset name"
            echo "  --output-dir DIR     Checkpoint output directory"
            echo "  --device DEVICE      Device: auto, cuda, mps, cpu"
            echo "  --resume PATH        Resume from checkpoint"
            echo "  --no-mixed-precision Disable mixed precision training"
            echo "  --wandb              Enable Weights & Biases logging"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Configuration:${NC}"
echo "  Epochs:           $EPOCHS"
echo "  Batch Size:       $BATCH_SIZE"
echo "  Learning Rate:    $LEARNING_RATE"
echo "  Model Size:       $MODEL_SIZE"
echo "  Dataset:          $DATASET_NAME"
echo "  Output Dir:       $OUTPUT_DIR"
echo "  Device:           $DEVICE"
echo "  Mixed Precision:  $USE_MIXED_PRECISION"
echo "  W&B Logging:      $USE_WANDB"
echo ""

# Build command
CMD="uv run python src/training/train.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --model-size $MODEL_SIZE \
    --dataset-name $DATASET_NAME \
    --output-dir $OUTPUT_DIR \
    --device $DEVICE"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

if [ "$USE_WANDB" = "true" ]; then
    CMD="$CMD --wandb"
fi

if [ "$USE_MIXED_PRECISION" = "false" ]; then
    echo -e "${YELLOW}Note: Mixed precision is enabled in config but not enforced by CLI${NC}"
fi

echo -e "${GREEN}Starting training...${NC}"
echo "Command: $CMD"
echo ""

# Run training
eval $CMD

echo ""
echo -e "${GREEN}✅ Training complete!${NC}"
echo "Checkpoints saved in: $OUTPUT_DIR"
