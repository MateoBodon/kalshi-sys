#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-tests/data_fixtures/index/raw}
PYTHON_BIN=${PYTHON:-python}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

SYMBOLS=("I:SPX" "I:NDX")
# Categories: gap (2024-08-05), quiet (2024-08-20), CPI (2024-09-11), FOMC (2024-09-18),
# plus the original quiet trio (2024-10-21..2024-10-23).
DATES=(
  "2024-08-05"
  "2024-08-20"
  "2024-09-11"
  "2024-09-18"
  "2024-10-21"
  "2024-10-22"
  "2024-10-23"
)
WINDOWS=("noon:11:45:00-12:05:00" "close:15:45:00-16:05:00")

mkdir -p "$OUT_DIR"

for symbol in "${SYMBOLS[@]}"; do
  symbol_slug=${symbol//:/_}
  for trading_day in "${DATES[@]}"; do
    for window_spec in "${WINDOWS[@]}"; do
      window_name=${window_spec%%:*}
      range_part=${window_spec#*:}
      start_part=${range_part%-*}
      end_part=${range_part#*-}
      output_path="$OUT_DIR/${symbol_slug}_${trading_day}_${window_name}.parquet"
      echo "Fetching $symbol $trading_day $window_name -> $output_path"
      "$PYTHON_BIN" "$SCRIPT_DIR/polygon_dump.py" \
        "$symbol" \
        "${trading_day}T${start_part}" \
        "${trading_day}T${end_part}" \
        "$output_path"
    done
  done
done
