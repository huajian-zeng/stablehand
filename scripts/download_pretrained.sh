#!/bin/bash
# Download the pretrained StableHand checkpoints (DiT + Quality Network) from
# Hugging Face into save/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python}" "$SCRIPT_DIR/download_release.py" model "$@"
