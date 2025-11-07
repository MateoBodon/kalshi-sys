#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/mateobodon/Documents/Programming/Projects/kalshi-sys"
CHANNEL_PREFIX="${POLYGON_CHANNEL_PREFIX:-A}"

cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

PYTHONPATH=src exec python -m kalshi_alpha.exec.collectors.polygon_ws \
  --symbols I:SPX,I:NDX \
  --channel-prefix "$CHANNEL_PREFIX" \
  --freshness-config configs/freshness.index.yaml
