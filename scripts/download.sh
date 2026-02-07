#!/bin/bash
# Remote checkpoint/sample sync helper (scp).
#
# Usage:
#   REMOTE_HOST=user@host REMOTE_BASE=/path/to/project ./scripts/download.sh pull
#   REMOTE_HOST=user@host REMOTE_BASE=/path/to/project ./scripts/download.sh pull 30
#   REMOTE_HOST=user@host REMOTE_BASE=/path/to/project ./scripts/download.sh push
#
# Optional env:
#   LOCAL_BASE=.
#
# If an epoch number is given (second argument), checkpoint filenames become
# diffusion_epoch_<N>.pt instead of diffusion.pt.

set -euo pipefail

MODE="${1:-pull}"
EPOCH="${2:-}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_BASE="${REMOTE_BASE:-}"
LOCAL_BASE="${LOCAL_BASE:-.}"

if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_BASE" ]; then
    echo "Error: REMOTE_HOST and REMOTE_BASE are required."
    exit 1
fi

if [ -n "$EPOCH" ]; then
    DIFFUSION_CKPT="diffusion_epoch_${EPOCH}.pt"
else
    DIFFUSION_CKPT="diffusion.pt"
fi

mkdir -p "$LOCAL_BASE/checkpoints"

case "$MODE" in
    pull)
        # scp -r "$REMOTE_HOST:$REMOTE_BASE/samples" "$LOCAL_BASE/"
        scp "$REMOTE_HOST:$REMOTE_BASE/checkpoints/vae.pt" "$LOCAL_BASE/checkpoints/vae.pt"
        scp "$REMOTE_HOST:$REMOTE_BASE/checkpoints/$DIFFUSION_CKPT" "$LOCAL_BASE/checkpoints/$DIFFUSION_CKPT"
        ;;
    push)
        scp "$LOCAL_BASE/checkpoints/vae.pt" "$REMOTE_HOST:$REMOTE_BASE/checkpoints/vae.pt"
        scp "$LOCAL_BASE/checkpoints/$DIFFUSION_CKPT" "$REMOTE_HOST:$REMOTE_BASE/checkpoints/$DIFFUSION_CKPT"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Expected: pull or push"
        exit 1
        ;;
esac
