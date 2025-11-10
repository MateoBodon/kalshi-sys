#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/mateobodon/Documents/Programming/Projects/kalshi-sys"
cd "$REPO_ROOT"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

POLYGON_CHANNEL_PREFIX="${POLYGON_CHANNEL_PREFIX:-A}"
CADENCE_LOG_THRESHOLD="${CADENCE_LOG_THRESHOLD:-30}"
VALUE_FALLBACK_SECONDS="${VALUE_FALLBACK_SECONDS:-15}"

PYTHONPATH=src exec python -m kalshi_alpha.exec.collectors.polygon_ws \
  --symbols I:SPX,I:NDX \
  --channel-prefix "$POLYGON_CHANNEL_PREFIX" \
  --freshness-config configs/freshness.yaml \
  --freshness-output reports/_artifacts/monitors/freshness.json \
  --proc-parquet data/proc/polygon_index/snapshot_live.parquet \
  --cadence-log-threshold "$CADENCE_LOG_THRESHOLD" \
  --value-fallback-seconds "$VALUE_FALLBACK_SECONDS"
