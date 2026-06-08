# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research framework studying **strategic miscalibration** in LLMs — whether LLMs strategically misreport confidence scores when they know their output influences another agent's decisions. It implements a repeated signaling game with reputation dynamics, elicits strategies and gameplay from LLMs, and analyzes the resulting behavior.

The core experiment: an **agent LLM** solves tasks and reports a confidence score; a **user LLM** (or rational threshold rule) observes only the confidence score and decides whether to delegate the task (paying cost `c`) or self-solve (paying effort `e`). Repeated rounds create reputation incentives.

LLM queries are routed through [LiteLLM](https://docs.litellm.ai/docs/) to [Together AI](https://docs.together.ai/intro). Set `TOGETHERAI_API_KEY` in a `.env` at the project root.

## Commands

```bash
uv sync                        # Install/sync dependencies
uvx pytest -vv                 # Run all tests
uvx pytest tests/test_sanitizers.py -vv   # Run a single test file
uvx ruff check .               # Lint
uvx ruff format .              # Format
```

Run experiments directly as modules (from repo root):

```bash
uv run python -m strategicmiscalibration.math_qa_game    # Two-player game sweep
uv run python -m strategicmiscalibration.math_qa_agent_only  # Single-player (agent only)
uv run python -m strategicmiscalibration.analysis        # Analyze latest results CSV
uv run python -m strategicmiscalibration.analysis --filter --correct --save --data-dir outputs/experiments/<run>
```

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

Each game type has three templates: `game_agent_prompt.j2` (strategic agent), `decision_user_prompt.j2` (user delegation decision), `posterior_user_prompt.j2` (user belief update after observing outcome). `math_qa/` also has `baseline_agent_prompt.j2` for the no-context baseline.

Templates are Jinja2. The agent prompt passes game parameters, history, and `confidence_mode` to control the allowed confidence values (binary/tercile/quartile/decile/continuous).

### Confidence Modes

`ConfidenceMode` enum in `llm_interface.py`: `BINARY` (0/1), `TERCILE` (0/0.5/1), `QUARTILE`, `DECILE`, `CONTINUOUS`. The toy game uses `BINARY`; the math QA game uses `CONTINUOUS`. The mode is threaded through config → template rendering → prompt instructions.

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
