#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/mateobodon/Documents/Programming/Projects/kalshi-sys"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

PYTHONPATH=src exec python -m kalshi_alpha.exec.collectors.polygon_ws \
  --symbols I:SPX,I:NDX \
  --channel-prefix AM \
  --freshness-config configs/freshness.index.yaml
