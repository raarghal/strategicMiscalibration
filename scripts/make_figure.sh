#!/usr/bin/env bash
# Regenerate the two-threshold figure from a toy sweep, with the DERIVED (not
# fitted) thresholds overlaid. Saves <data-dir>/toy_two_threshold.pdf.
#
# Usage:
#   scripts/make_figure.sh [DATA_DIR] [ROUND]
# If DATA_DIR is omitted, the most recent outputs/experiments/toy_*sweep* run is
# used. ROUND defaults to 1 (the round the phase-diagram prediction is about).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${1:-}"
ROUND="${2:-1}"

if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="$(ls -dt outputs/experiments/toy_*sweep* 2>/dev/null | head -n1 || true)"
  if [[ -z "$DATA_DIR" ]]; then
    echo "No outputs/experiments/toy_*sweep* run found. Run a sweep first, or pass DATA_DIR." >&2
    exit 1
  fi
  echo "Using latest sweep: $DATA_DIR"
fi

uv run python -m strategicmiscalibration.toy_plots \
  --data-dir "$DATA_DIR" \
  --round "$ROUND" \
  --save
