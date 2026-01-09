#!/bin/bash
# =============================================================================
# text-to-emoji Pretraining Script
# =============================================================================
#
# This script runs Stage 1: Pretraining on CIFAR-100
#
# Usage:
#   ./pretraining.sh
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 text-to-emoji Pretraining${NC}"
echo "============================================"

echo -e "${YELLOW}📊 Pretraining on CIFAR-100 (60,000 images)${NC}"

echo ""
echo "Starting training..."
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run training with uv
# The script will use PRETRAIN_CONFIG from main.py
uv run python main.py --train

echo ""
echo -e "${GREEN}✅ Pretraining complete!${NC}"
echo "Checkpoints saved to: checkpoints/"
echo "Samples saved to: samples/"
