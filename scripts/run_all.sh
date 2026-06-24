#!/usr/bin/env bash
# Run the full toy experiment battery sequentially, then build the headline
# figure from the oracle/scaffolded sweep. Hits the Together API for every arm
# (the LLM-user arm is the most expensive). Each arm writes to a fixed, labelled
# directory so the figure step can reference the right one.
set -euo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
BASE="outputs/experiments"
ORACLE_DIR="$BASE/toy_sweep_oracle_scaffolded_$TS"

echo ">>> [1/4] oracle user, scaffolded agent (headline)"
uv run python -m strategicmiscalibration.toy_experiment \
  --mode sweep --user-kind oracle --agent-prompt-style scaffolded \
  --conjecture standard --out "$ORACLE_DIR"

echo ">>> [2/4] oracle user, minimal agent (de-scaffold robustness)"
uv run python -m strategicmiscalibration.toy_experiment \
  --mode sweep --user-kind oracle --agent-prompt-style minimal \
  --conjecture standard --out "$BASE/toy_sweep_oracle_minimal_$TS"

echo ">>> [3/4] LLM user, scaffolded agent (real-counterpart robustness)"
uv run python -m strategicmiscalibration.toy_experiment \
  --mode sweep --user-kind llm --agent-prompt-style scaffolded \
  --conjecture standard --out "$BASE/toy_sweep_llm_scaffolded_$TS"

echo ">>> [4/4] within-state mixing diagnostic (temp>0, repeats)"
uv run python -m strategicmiscalibration.toy_experiment \
  --mode mixing --user-kind oracle --agent-prompt-style scaffolded \
  --conjecture standard --temperature 0.7 --num-trials 25 \
  --out "$BASE/toy_mixing_oracle_scaffolded_$TS"

echo ">>> figure from headline sweep"
uv run python -m strategicmiscalibration.toy_plots \
  --data-dir "$ORACLE_DIR" --round 1 --save

echo "Done. Headline figure: $ORACLE_DIR/toy_two_threshold.pdf"
