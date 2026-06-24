# Experiment runner scripts

Thin wrappers around `strategicmiscalibration.toy_experiment` (and `toy_plots`),
one per experiment arm. Each script `cd`s to the repo root and issues a single
`uv run` command, so it can be invoked from anywhere:

```bash
scripts/run_phase_diagram_oracle.sh
```

All sweeps hit the Together API (`TOGETHERAI_API_KEY` in `.env`). Results land in
`outputs/experiments/toy_<...>_<timestamp>/` as `results_*.csv` + `config_*.json`.

| Script | Arm | What it tests |
|---|---|---|
| `run_phase_diagram_oracle.sh` | oracle user, scaffolded agent, `standard` conjecture | **Headline** clean two-threshold test: agent vs. the exact Bayes-rational user. |
| `run_phase_diagram_minimal.sh` | oracle user, **minimal** agent prompt | De-scaffold robustness — the pattern isn't an artifact of the prompt leaking the mechanism. |
| `run_phase_diagram_llm_user.sh` | **LLM** user, scaffolded agent | Survives a real (free-form) LLM counterpart. ~3× the API cost. |
| `run_mixing_diagnostic.sh` | oracle user, fixed state, `temp=0.7`, 25 reps | **Within-state** mixing (P3) — distinct from cross-grid frequency; needed for any `beta` claim. |
| `make_figure.sh [DATA_DIR] [ROUND]` | — | Regenerate the two-threshold figure with **derived** thresholds overlaid. Defaults to the latest sweep, round 1. |
| `run_strategy_map.sh [--dry-run] [...]` | scaffolded + minimal (action) + **strategy** elicitation, over a 5×5 `(h,μ)` grid × 2 `δ` | Maps `σ^A(ρ⁺),σ^A(ρ⁻)` over the reputation plane to test the **trust watershed** (`thm:phase`(c)) and any trusted-region sandbagging. Also runs the `equilibria.py` theory grid + generates all figures. ~1450 calls, **~4h @ 10s/call** (`--dry-run` to preview). Driven by `toy_strategy_map`, not `toy_experiment`. See `notes/strategy-map-experiment-plan.md`. |
| `run_all.sh` | all of the above | Full battery sequentially, then the headline figure. |

## Underlying CLI

The scripts are thin; everything is driven by:

```bash
uv run python -m strategicmiscalibration.toy_experiment \
  --mode {sweep,mixing} \
  --user-kind {oracle,llm} \
  --agent-prompt-style {scaffolded,minimal} \
  --conjecture {standard,babble_up} \
  [--temperature T] [--num-trials N] \
  [--agent-model M] [--user-model M] [--out DIR]
```

- `--conjecture standard` = trusting user `(a₊,a₋)=(1,0)`; `babble_up` = `(1,1)`,
  which activates the reputation-punishment channel (the strategic type is
  conjectured to always claim high). Setting it equal to the agent's actual
  strategy would close the equilibrium fixed point.
- `--agent-model` / `--user-model` override the default Llama-3.3-70B for the
  ≥2-model-family robustness check.

To run an arm not covered by a script (e.g. a second model family), call the CLI
directly or copy one of the `.sh` files.
