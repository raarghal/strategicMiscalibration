#!/usr/bin/env bash
# Strategy map: elicit the agent's first-round reporting strategy sigma^A(rho)
# over the (h, mu) prior plane, under three regimes -- ORIGINAL toy prompt
# (scaffolded, action), the NEW toy prompt (minimal, action), and direct STRATEGY
# elicitation (sigma_high, sigma_low). Tests the trust watershed (no high-side
# under-reporting for h >= 1/2) and any trusted-region sandbagging.
#
# Default budget: 5x5 (h,mu) grid x 2 delta x {scaffolded:6, minimal:6, strategy:5}
#   = 1450 Together API calls, ~4.0h @ 10s/call (MEASURED; range 3.2-4.8h). Sequential.
# The runner prints the exact budget + runtime estimate before starting.
# Also computes the equilibria.py theory grid and generates all figures at the end.
#
# Preview the budget without spending anything:
#   scripts/run_strategy_map.sh --dry-run
# Override the grid / samples (passed straight through), e.g. one delta to halve it:
#   scripts/run_strategy_map.sh --deltas 0.05 --n-action 18
#
# Output: outputs/experiments/strategy_map_<ts>/strategy_map_<ts>.csv (+ config json)
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m strategicmiscalibration.toy_strategy_map \
  --versions scaffolded,minimal,strategy \
  "$@"
