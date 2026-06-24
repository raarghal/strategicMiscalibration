#!/usr/bin/env bash
# Within-state mixing diagnostic (the P3 check). Holds the belief state fixed at
# (h, mu) = (0.5, 0.5) and draws many samples at temperature > 0 for a handful of
# delta straddling both thresholds, so the report frequency estimates the agent's
# WITHIN-STATE mixing probability (distinct from the cross-grid frequency the
# sweep averages). Required before making any claim about the mixing weight beta.
#
# Output: outputs/experiments/toy_mixing_oracle_scaffolded_standard_<ts>/
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m strategicmiscalibration.toy_experiment \
  --mode mixing \
  --user-kind oracle \
  --agent-prompt-style scaffolded \
  --conjecture standard \
  --temperature 0.7 \
  --num-trials 25
