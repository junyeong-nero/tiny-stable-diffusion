#!/bin/bash
# Environment Setup Script
#
# Usage:
#   ./scripts/setup.sh
# Optional env:
#   INSTALL_TMUX=1   Install tmux via apt-get (Linux + apt only)
#   WANDB_LOGIN=1    Run wandb login step

set -euo pipefail

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "Syncing dependencies..."
uv sync
uv pip install git+https://github.com/openai/CLIP.git

if [ "${WANDB_LOGIN:-0}" = "1" ]; then
    uv run wandb login
fi

if [ "${INSTALL_TMUX:-0}" = "1" ]; then
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y tmux
    else
        echo "Skipping tmux install: apt-get not available"
    fi
fi
