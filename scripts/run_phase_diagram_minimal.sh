#!/usr/bin/env bash
# Robustness: de-scaffolded agent prompt (no backward-induction walkthrough, no
# printed kappa multiplier) vs. the oracle user. Shows the two-threshold pattern
# is not an artifact of the prompt leaking the mechanism.
#
# Output: outputs/experiments/toy_sweep_oracle_minimal_standard_<ts>/
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m strategicmiscalibration.toy_experiment \
  --mode sweep \
  --user-kind oracle \
  --agent-prompt-style minimal \
  --conjecture standard
