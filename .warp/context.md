# Strategic Uncertainty Quantification

## Project Overview
This is a research repository investigating strategic uncertainty quantification by AI systems. The project explores LLM-based control of social learning and is associated with the paper "Steering the Herd: A Framework for LLM-based Control of Social Learning" (Arghal et al., 2025).

## Tech Stack
- **Language**: Python 3.12+
- **Package Manager**: uv (Astral)
- **AI Framework**: Together AI API
- **Key Dependencies**:
  - `together` (LLM API client)
  - `datasets` (data processing)
  - `jinja2` (templating)
  - `tenacity` (retry logic)
  - `tqdm` (progress bars)

## Project Structure
```
src/strategicuncertainty/
├── __init__.py
├── llm_interface.py      # LLM API interactions
├── single_player.py      # Single-player game scenarios
├── two_player.py         # Two-player game scenarios
└── utils.py              # Utility functions

tests/                     # Test files
outputs/                   # Experimental outputs
```

## Development Workflow

### Environment Setup
```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install pre-commit hooks
uvx pre-commit install
```

### Running Code
```bash
# Run a script with the project environment
uv run python path/to/script.py

# Run as a module
uv run -m strategicuncertainty.module_name
```

### Testing
```bash
# Run tests with parallel execution
uvx pytest -n auto -vv
```

### Code Quality
```bash
# Format code
uvx ruff format .

# Lint code
uvx ruff check .

# Run pre-commit hooks manually
uvx pre-commit run --all-files
```

### Dependency Management
```bash
# Add new package
uv add <package-name>

# Add dev dependency
uv add --dev <package-name>

# Update dependencies
uv lock --upgrade

# Sync environment after changes
uv sync
```

## Environment Variables
The project requires a `.env` file in the project root with:
```
TOGETHERAI_API_KEY="<your-api-key>"
```

## Code Style
- Line length: 119 characters
- Python 3.12+ syntax
- Google-style docstrings
- Ruff for linting and formatting

## License
MIT License (Copyright 2025 Raghu Arghal)
