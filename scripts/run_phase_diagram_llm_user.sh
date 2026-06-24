#!/usr/bin/env bash
# Robustness: scaffolded agent vs. the original free-form LLM user (not the
# oracle). Shows the agent's regime behaviour survives a real LLM counterpart.
# NOTE: ~3x the API cost of the oracle arms (user decision + posterior are also
# LLM calls).
#
# Output: outputs/experiments/toy_sweep_llm_scaffolded_standard_<ts>/
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m strategicmiscalibration.toy_experiment \
  --mode sweep \
  --user-kind llm \
  --agent-prompt-style scaffolded \
  --conjecture standard
