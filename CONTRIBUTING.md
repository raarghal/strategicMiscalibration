# Contributing

## Prerequisites

- Python 3.12+
- [uv package manager](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A Together AI API key

## Development Setup

1. **Install uv** following the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

2. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd strategicMiscalibration
   ```

3. **Create the virtual environment and install project dependencies**:
   ```bash
   uv sync
   ```

4. **Activate the environment (optional)**:
   ```bash
   source .venv/bin/activate
   ```

5. **Set up a .env file**

   In the project root, create a `.env` file:
   ```
   TOGETHERAI_API_KEY="<your key here>"
   ```
   Do not commit this file (already in `.gitignore`).

## Project Structure

```
├── src/strategicmiscalibration/   # Main package
│   ├── datatypes.py               # Shared TypedDicts and game config dataclasses
│   ├── llm_interface.py           # LiteLLM wrapper, Pydantic response schemas
│   ├── utils.py                   # Query+sanitize helpers, stats utilities
│   ├── math_qa_game.py            # Two-player experiment runner
│   ├── math_qa_agent_only.py      # Single-player (agent-only) experiment runner
│   ├── analysis.py                # Post-hoc analysis and plotting CLI
│   ├── plotting.py                # Theoretical trust-region plots
│   └── prompt_templates/          # Jinja2 templates for agent/user prompts
├── tests/                         # Test files
├── pyproject.toml                 # Project metadata and dependencies
└── outputs/                       # Experiment results (gitignored)
```

## Common Commands

```bash
uv sync                        # Install/sync dependencies
uv run pytest -vv              # Run full test suite
uv run pytest tests/test_sanitizers.py -vv   # Run a single test file
uv run ruff check .            # Lint
uv run ruff format .           # Format
```

## Running Experiments

```bash
# Two-player game (agent + user LLMs)
uv run python -m strategicmiscalibration.math_qa_game

# Single-player (agent only, measures confidence shift from strategic framing)
uv run python -m strategicmiscalibration.math_qa_agent_only

# Analyze results
uv run python -m strategicmiscalibration.analysis --data-dir outputs/experiments/<run> --save
```

## Adding Dependencies

```bash
uv add <package-name>          # Add runtime dependency
uv add --dev <package-name>    # Add dev-only dependency
uv lock                        # Regenerate lockfile
uv sync                        # Install locally
```

Commit both `pyproject.toml` and `uv.lock` together.

## Code Style

- Line length: 119 characters (Ruff)
- Google-style docstrings
- Python 3.12+

```bash
uv run ruff format .
uv run ruff check .
```

## Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make changes and add tests in `tests/`
3. Run `uv run pytest -vv` and `uv run ruff check .`
4. Commit with a descriptive message and open a pull request
