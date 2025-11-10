#!/usr/bin/env bash
set -euo pipefail
cd /Users/mateobodon/Documents/Programming/Projects/kalshi-sys
source .venv/bin/activate
PYTHONPATH=src python scripts/run_indices_listener.py
