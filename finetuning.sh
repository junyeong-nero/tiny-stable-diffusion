#!/bin/bash
# =============================================================================
# PixMoji-Diffusion Fine-tuning Script
# =============================================================================
#
# This script runs Stage 2: Fine-tuning on emoji dataset
# IMPORTANT: Edit TRAINING_STAGE = "finetune" in main.py first!
#
# Usage:
#   ./finetuning.sh                    # Run with default settings
#   ./finetuning.sh 50                # Fine-tune for 50 epochs
#   ./finetuning.sh 100 1e-5          # Custom epochs and learning rate
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎨 PixMoji-Diffusion Fine-tuning${NC}"
echo "=========================================="

# Check if TRAINING_STAGE is set to "finetune"
if grep -q 'TRAINING_STAGE.*=.*"finetune"' main.py; then
    echo -e "${GREEN}✅ TRAINING_STAGE is set to 'finetune'${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING: TRAINING_STAGE may not be 'finetune'${NC}"
    echo -e "${YELLOW}Please edit main.py and set: TRAINING_STAGE = \"finetune\"${NC}"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted.${NC}"
        exit 1
    fi
fi

# Parse arguments
EPOCHS="${1:-100}"
LEARNING_RATE="${2:-1e-5}"
BATCH_SIZE="${3:-16}"

echo ""
echo -e "${BLUE}📊 Fine-tuning Configuration:${NC}"
echo "   Epochs: $EPOCHS"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Batch Size: $BATCH_SIZE"
echo ""

# Check for pretrained checkpoint
CHECKPOINT_FILE=$(grep -oP 'pretrain_checkpoint.*=.*"\K[^"]+' main.py 2>/dev/null || echo "checkpoints/pretrain_cifar100.pt")
if [ -f "$CHECKPOINT_FILE" ]; then
    echo -e "${GREEN}✅ Found pretrained checkpoint: $CHECKPOINT_FILE${NC}"
else
    echo -e "${YELLOW}⚠️  Pretrained checkpoint not found: $CHECKPOINT_FILE${NC}"
    echo "   Training will start from scratch."
fi

echo ""
echo "Starting fine-tuning..."
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run training with uv
uv run python main.py --train

echo ""
echo -e "${GREEN}✅ Fine-tuning complete!${NC}"
echo "Checkpoints saved to: checkpoints/"
echo "Samples saved to: samples/"
echo ""
echo "Next steps:"
echo "1. Generate images: python main.py --generate --prompt 'your prompt'"
echo "2. Run demo: python main.py --demo"
