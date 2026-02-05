#!/bin/bash
# HuggingFace helper wrapper.
#
# Usage:
#   ./scripts/hf.sh upload vae <repo-id>
#   ./scripts/hf.sh upload diffusion <repo-id>
#   ./scripts/hf.sh upload all <repo-id>
#   ./scripts/hf.sh download vae <repo-id>
#   ./scripts/hf.sh download diffusion <repo-id>
#   ./scripts/hf.sh download all <repo-id>

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <upload|download> <vae|diffusion|all> <repo-id>"
    exit 1
fi

ACTION="$1"
MODEL_TYPE="$2"
REPO_ID="$3"
shift 3

case "$ACTION" in
    upload)
        uv run python scripts/upload_to_hub.py \
            --model-type "$MODEL_TYPE" \
            --repo-id "$REPO_ID" \
            "$@"
        ;;
    download)
        uv run python scripts/download_from_hub.py \
            --model-type "$MODEL_TYPE" \
            --repo-id "$REPO_ID" \
            "$@"
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Expected: upload or download"
        exit 1
        ;;
esac
