#!/usr/bin/env bash
set -euo pipefail
cd /Users/mateobodon/Documents/Programming/Projects/kalshi-sys
source .venv/bin/activate
PYTHONPATH=src python -m kalshi_alpha.exec.collectors.polygon_ws \
  --symbols I:SPX,I:NDX \
  --channel-prefix A \
  --freshness-config configs/freshness.index.yaml \
  --freshness-output reports/_artifacts/monitors/freshness.json
