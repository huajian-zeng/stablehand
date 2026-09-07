#!/bin/bash
# Download both complete test-cache sets and the ten example videos into data/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python}" "$SCRIPT_DIR/download_release.py" dataset "$@"
