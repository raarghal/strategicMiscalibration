# Project Rules and Guidelines

## Package Management
- Always use `uv` for dependency management, not pip or poetry directly
- Use `uvx` (shorthand for `uv tool run`) for running development tools
- Run scripts with `uv run python script.py` to use the project environment
- After adding dependencies with `uv add`, run `uv sync` to update the environment
- Commit both `pyproject.toml` and `uv.lock` when dependencies change

## Code Quality
- Use Ruff for all linting and formatting (configured in `pyproject.toml`)
- Run `uvx ruff format .` before committing
- Run `uvx ruff check .` to check for linting issues
- Pre-commit hooks will automatically run on commit
- Line length limit: 119 characters
- Use Google-style docstrings

## Testing
- Write tests in the `tests/` directory
- Run tests with `uvx pytest -n auto -vv` for parallel execution
- Add tests for any new functionality

## Environment
- Never commit the `.env` file (it's already in `.gitignore`)
- The project requires `TOGETHER_API_KEY` in `.env`
- Python version: 3.12+ (specified in `.python-version`)

## Git Workflow
- Create feature branches for new work: `git checkout -b feature/name`
- Use descriptive commit messages following conventional commits format
- Pre-commit hooks will run automatically and may modify files

## Code Organization
- Main package code lives in `src/strategicuncertainty/`
- Keep utility functions in `utils.py`
- LLM interface code belongs in `llm_interface.py`
- Experimental outputs go in `outputs/` directory

## Dependencies
- Only add dependencies that are necessary for the project
- Prefer well-maintained packages with active communities
- Document the purpose of new dependencies in pull requests
