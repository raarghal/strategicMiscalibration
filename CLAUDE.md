# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research framework studying **strategic miscalibration** in LLMs — whether LLMs strategically misreport confidence scores when they know their output influences another agent's decisions. It implements a repeated signaling game with reputation dynamics, elicits strategies and gameplay from LLMs, and analyzes the resulting behavior.

The core experiment: an **agent LLM** solves tasks and reports a confidence score; a **user LLM** (or rational threshold rule) observes only the confidence score and decides whether to delegate the task (paying cost `c`) or self-solve (paying effort `e`). Repeated rounds create reputation incentives.

LLM queries are routed through [LiteLLM](https://docs.litellm.ai/docs/) to [Together AI](https://docs.together.ai/intro). Set `TOGETHERAI_API_KEY` in a `.env` at the project root.

## Commands

```bash
uv sync                        # Install/sync dependencies
uv run pytest -vv              # Run all tests
uv run pytest tests/test_sanitizers.py -vv   # Run a single test file
uv run ruff check .            # Lint
uv run ruff format .           # Format
```

Run experiments directly as modules (from repo root):

```bash
uv run python -m strategicmiscalibration.math_qa_game    # Two-player game sweep
uv run python -m strategicmiscalibration.math_qa_agent_only  # Single-player (agent only)
uv run python -m strategicmiscalibration.analysis        # Analyze latest results CSV
uv run python -m strategicmiscalibration.analysis --filter --correct --save --data-dir outputs/experiments/<run>
```

**Experiment runner scripts** (`scripts/`) — thin `uv run` wrappers, one per toy
experiment arm; each `cd`s to repo root and can be run from anywhere. See
`scripts/README.md` for the table.

```bash
scripts/run_phase_diagram_oracle.sh      # headline: oracle user, scaffolded agent
scripts/run_phase_diagram_minimal.sh     # de-scaffold robustness (minimal prompt)
scripts/run_phase_diagram_llm_user.sh    # LLM-user robustness (~3x API cost)
scripts/run_mixing_diagnostic.sh         # within-state mixing: temp>0, repeats, fixed state
scripts/make_figure.sh [DATA_DIR] [ROUND]  # two-threshold figure, derived thresholds overlaid
scripts/run_all.sh                       # full battery + headline figure
```

All are driven by the `toy_experiment` CLI (`--mode {sweep,mixing}`,
`--user-kind {oracle,llm}`, `--agent-prompt-style {scaffolded,minimal}`,
`--conjecture {standard,babble_up}`, `--temperature`, `--num-trials`,
`--agent-model`, `--user-model`, `--out`); call it directly for arms without a
script (e.g. a second model family).

## Architecture

### Game Modes

**`math_qa_agent_only.py`** — Single-player mode: compares baseline agent confidence (no strategic framing) vs. strategic agent confidence (game framing). No user LLM involved; measures whether framing alone shifts confidence reporting.

**`math_qa_game.py`** — Two-player mode: both agent and user are LLMs. Per-round flow:
1. Sample a math task from the dataset (`TAL-SCQ5K-EN-R1`)
2. Query baseline agent (no game context) → sanitize
3. Query strategic agent (full game context + history) → sanitize
4. Query user LLM for delegation decision + prior beliefs → sanitize
5. Compute payoffs; if user delegates, query user LLM for posterior belief update
6. Append to history; beliefs `h_t` (honesty) and `μ_t` (ability) carry forward

Entry point for running sweeps over `discount_factor`, `h_0`, `μ_0`: the `__main__` block at the bottom of `math_qa_game.py`. Results saved to `outputs/experiments/` as `results_<timestamp>.csv` + `config_<timestamp>.json`.

**`toy_game.py`** — Two-player *toy* signaling game (the "monopolistic" setting): the agent observes a synthetic success probability `ρ_t ∈ {ρ⁻, ρ⁺}` and reports a binary signal; a user (LLM) delegates or self-completes. `BINARY` confidence, `ToyGameConfig`. This is the source of the two-threshold phase-diagram figure. Note its `__main__` forces `first_round_task="EASY"`, under which round-1 over-reporting is structurally impossible (no signal above `ρ⁺`) — use `toy_experiment.py` for two-sided figures.

**`toy_experiment.py`** — Canonical, reproducible toy runner that closes the gap between *action elicitation* (what the harness observes) and *equilibrium* (what the theory characterizes). `ToyExperimentConfig` adds: `user_kind` ("oracle" = exact Bayes-rational threshold user from `oracle_user.py`, removing the "which user is the agent best-responding to?" confound; "llm" = original free-form LLM user); `agent_prompt_style` ("minimal" = de-scaffolded prompt with no backward-induction walkthrough / no κ multiplier; "scaffolded" = original); `conjecture_(a_plus,a_minus)` = the oracle's belief about the strategic type's reporting rule (default `(1,0)` = trusting "standard" user; set equal to the agent's actual `σ^A` to close the equilibrium fixed point). Two parameterized config builders: `build_canonical_sweep(...)` (sweeps δ across both derived thresholds and `first_round_task ∈ {EASY, HARD}` so both report directions are realizable) and `build_mixing_diagnostic(...)` (fixed belief state, `temperature>0`, many repeats — estimates within-state mixing, distinct from cross-grid frequency). Both take the arm flags as parameters and are driven by the module's argparse CLI (`main()`); the `scripts/` folder holds one thin runner per arm. Writes the same per-round CSV schema as `toy_game` (plus arm columns).

**Toy theory/model support modules** (pure math, no LLM):
- `toy_model.py` — exact discrete Bayes ground truth for the 2-type×2-type game (posteriors + success-prob) under a configurable strategic-type conjecture; `ToyModelParams`.
- `toy_theory.py` — phase-diagram predictions: `κ=δ/(1−δ)`, derived thresholds `δ` at `κ=1−ρ⁺` (deflation onset) and `κ=1−ρ⁻` (inflation onset), `Regime` classifier, per-state `predicted_report_type`.
- `oracle_user.py` — wraps `toy_model` into a drop-in Bayes-rational user (`decide`, `update_posterior`) mirroring the LLM user's fields.
- `toy_plots.py` — regenerates the two-threshold figure from a toy CSV with the *derived* (not fitted) thresholds overlaid: `uv run python -m strategicmiscalibration.toy_plots --data-dir outputs/experiments/sweep_toy_<ts> --save`.

Tests for the Bayes math / thresholds live in `tests/test_toy_oracle.py`. Planning docs for the paper — the experiment suite (what to measure/plot/interpret per experiment), repo-readiness/reproducibility recommendations, and the strategy-elicitation scope — live in `notes/` (see `notes/README.md`).

### Equilibrium numerics (theory side, no LLM)

**`numericals.py`** — scans the `(h,μ)` belief grid for the monopolistic game's equilibria, transcribed from `main.tex`'s appendix tables as ~30 `cond_T*R*`/sampler/`ROW_SPECS`. The Bayes engine `reputation_updates` is verified correct; `agent_best_responds` (added) is a theory-independent agent-IC gate AND-ed into the live scan `run_and_save_scan_user_mix` so transcription errors can't admit non-equilibria. Its default sweep is under `if __name__ == "__main__"` (so the module is importable without running a scan). Known caveat: the row structure mirrors `main.tex`'s *old* tables (which contain errors — see `drafts/memory/numericals-equilibrium-verification.md`), superseded by `revised_main.tex`'s `tab:master`.

**`equilibria.py`** — the table-free redesign. Characterizes equilibria from first principles: a profile is accepted iff both players best-respond (`is_equilibrium` = `user_best_responds ∧ agent_best_responds`), and each is given a **descriptive** label (`classify_equilibrium`, e.g. `"hedged-standard · under-reporting"`) instead of a `T#R#` reference. Avoids combinatorial blow-up: pure equilibria are the 16 corner profiles (exact, grid-free); mixed equilibria are solved by 1-D root-finds on indifference loci (`solve_hedged_standard` is the worked example). The `numericals` samplers are kept as an optional fast path (`source="samplers"`), run through the same gates; `cross_check` compares the two (soundness ⟺ `generic_only==0`). Tests in `tests/test_equilibria.py`. Run `uv run python -m strategicmiscalibration.equilibria` for a demo.

### Key Abstractions

**`datatypes.py`** — All shared data contracts (TypedDicts, frozen dataclasses, config dataclasses). The game configs live here:
- `BaseGameConfig`: common parameters (models, reward `r`, cost `c`, effort `e`, discount factor `δ`, priors `h_0`/`μ_0`)
- `ToyGameConfig`: binary confidence mode, toy prompt templates
- `MathQAGameConfig`: continuous confidence mode, math QA prompt templates + dataset name

The delegation threshold is computed as `θ* = 1 - (effort - cost) / reward`. The user delegates when `agent_confidence ≥ θ*`.

**`llm_interface.py`** — All LLM interaction. Pydantic schemas for structured JSON output (`AgentBaselineResponse`, `AgentGameResponse`, `UserDecisionResponse`, `UserPosteriorResponse`). Uses `litellm.enable_json_schema_validation = True` globally. Retries via `tenacity` (5 attempts, exponential backoff 5–120s).

**`utils.py`** — Two responsibilities:
1. Query+sanitize wrappers: each `query_and_sanitize_*` function calls the LLM then normalizes/validates the response into a `Sanitized*` dataclass. Invalid responses set `is_valid=False` but still return a struct (rounds are recorded as failed rather than raising).
2. Statistics helpers: `compute_baseline_stats`, `compute_agent_stats`, `compute_confidence_comparison_stats`, `aggregate_trial_stats`.

**`analysis.py`** — CLI script for post-hoc analysis. Loads the latest `results*.csv` from an experiment dir, builds confidence-diff bar plots and delegation-rate heatmaps by prior beliefs. Saves PDFs to `confidence_plots/`, `confidence_diff_heatmaps/`, `delegation_heatmaps/` subdirs.

**`plotting.py`** — Additional plotting utilities.

### Prompt Templates

Located at `src/strategicmiscalibration/prompt_templates/`, split by game type:

- `toy/` — Two-round toy game prompts with binary confidence and explicit discount-factor framing
- `math_qa/` — Math QA prompts with continuous confidence

Each game type has `game_agent_prompt.j2` (strategic agent), `decision_user_prompt.j2` (user delegation decision), and `posterior_user_prompt.j2` (user belief update after observing outcome). `toy/` additionally has `game_agent_final_prompt.j2` (final-round agent prompt without future-discount framing); `math_qa/` additionally has `baseline_agent_prompt.j2` for the no-context baseline.

Templates are Jinja2. The agent prompt passes game parameters, history, and `confidence_mode` to control the allowed confidence values (binary/tercile/quartile/decile/continuous).

### Confidence Modes

`ConfidenceMode` enum in `llm_interface.py`: `BINARY` (0/1), `TERCILE` (0/0.5/1), `QUARTILE`, `DECILE`, `CONTINUOUS`. The toy game uses `BINARY`; the math QA game uses `CONTINUOUS`. The mode is threaded through config → template rendering → prompt instructions.

### Output Modes

`OutputMode` enum in `llm_interface.py` controls how structured output is obtained from the model. Set it via `BaseGameConfig.output_mode` (default `JSON_SCHEMA`); it threads through `load_template()` and `query_llm()`.

- `JSON_SCHEMA` — Relies on the provider's native `response_format` JSON-schema enforcement. Use for models that support it (e.g. Llama-3.3 via Together).
- `TEXT` — Requests plain text and parses the response with `extract_json()` (tries the whole string, a fenced ```json block, then the first `{...}` substring that validates). Use for models that handle structured output poorly (e.g. some GPT-OSS deployments). Switching modes is a one-line config change:

  ```python
  cfg = ToyGameConfig(agent_model_name="...gpt-oss...", output_mode=OutputMode.TEXT)
  ```

In `TEXT` mode, `load_template()` injects a `json_instructions` variable (built from the response schema by `generate_json_instructions()`); place `{{ json_instructions }}` wherever the output-format block belongs in a template. In `JSON_SCHEMA` mode that variable renders empty. The `toy/` templates use `{{ json_instructions }}`; the `math_qa/` templates instead carry their own confidence-mode-aware JSON blocks (richer than the generic generator) and work unchanged in both modes.

### Output Structure

```
outputs/
  experiments/<run_name>/
    results_<timestamp>.csv    # Per-round rows with all config fields included
    config_<timestamp>.json    # Constant config fields snapshot
  trials/two_player_<timestamp>/
    results_<timestamp>.json   # Full nested trial results
    summary_<timestamp>.txt    # Human-readable summary
```

`run_experiments()` in `math_qa_game.py` is the entry point for multi-config sweeps and writes the CSV/JSON pair. `run_trials()` writes the nested JSON + summary.

## Style

- Line length: 119 characters (Ruff)
- Google-style docstrings
- Python 3.12+
