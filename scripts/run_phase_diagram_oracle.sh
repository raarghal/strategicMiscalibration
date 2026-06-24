#!/usr/bin/env bash
# Headline experiment: two-threshold phase diagram, clean equilibrium test.
# Strategic agent (scaffolded prompt) vs. the exact Bayes-rational ORACLE user
# (standard/trusting conjecture). This is the run that supports the figure.
#
# Output: outputs/experiments/toy_sweep_oracle_scaffolded_standard_<ts>/
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m strategicmiscalibration.toy_experiment \
  --mode sweep \
  --user-kind oracle \
  --agent-prompt-style scaffolded \
  --conjecture standard
