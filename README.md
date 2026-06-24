# Strategic Miscalibration

A research framework for studying **strategic miscalibration** in LLMs — whether
LLMs strategically misreport confidence when they know their output influences
another agent's decisions. It implements a repeated signaling game with
reputation dynamics, elicits strategies and gameplay from LLMs, and analyzes the
resulting behavior. It accompanies the game-theory paper *The Confidence Game*.

The core experiment: an **agent LLM** solves tasks and reports a confidence
score; a **user** (either an LLM or an exact Bayes-rational threshold rule)
observes only that score and decides whether to delegate the task (paying cost
`c`) or self-solve (paying effort `e`). Repeated rounds create reputation
incentives. LLM queries are routed through
[LiteLLM](https://docs.litellm.ai/docs/) to
[Together AI](https://docs.together.ai/intro).

## Orientation

- **`CLAUDE.md`** — the project map: architecture, modules, game modes,
  equilibrium numerics, key abstractions, and the full command reference.
  **Start here** to understand the codebase.
- **`CONTRIBUTING.md`** — development setup, code style, and how to add
  dependencies or open a pull request.
- **`scripts/README.md`** — the experiment-runner scripts (one thin wrapper per
  toy-experiment arm).

## Quickstart

Requires Python 3.12+ and the [uv](https://docs.astral.sh/uv/) package manager.

```bash
# 1. Install dependencies into a local virtualenv
uv sync

# 2. Configure your API key
cp .env.example .env        # then edit .env and set TOGETHERAI_API_KEY

# 3. Run the tests to confirm the setup
uv run pytest -vv

# 4. Reproduce the headline toy-game phase diagram (oracle user, scaffolded agent)
scripts/run_phase_diagram_oracle.sh
uv run python -m strategicmiscalibration.toy_plots --data-dir outputs/experiments/<run> --save
```

Results are written to `outputs/experiments/<run>/` (per-round CSV + config
JSON); figures are saved alongside. See `scripts/README.md` for the full battery
of experiment arms and `scripts/run_all.sh` to run them end-to-end.

## Common commands

```bash
uv sync                        # Install/sync dependencies
uv run pytest -vv              # Run the test suite
uv run ruff check .            # Lint
uv run ruff format .           # Format

# Experiments (from repo root)
uv run python -m strategicmiscalibration.toy_experiment --help   # canonical toy runner
uv run python -m strategicmiscalibration.math_qa_game            # two-player math-QA game
uv run python -m strategicmiscalibration.analysis --data-dir outputs/experiments/<run> --save
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This work is licensed under the MIT License. Copyright (c) 2025 Raghu Arghal.
