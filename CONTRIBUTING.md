# Contributing to flake-analysis-core

Thanks for considering a contribution!

## Development setup

```bash
git clone https://github.com/HoukJangBNL/flake-analysis-core.git
cd flake-analysis-core
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

## Code conventions

- Python 3.10+
- Logging via `flake_core._compat.msg` (stdlib `logging` shim) — never `print()`
- Type hints on all public functions
- Docstrings on all modules + public functions
- 100-character line limit (informal)

## Tests

- All new code MUST include tests under `tests/`
- Run `pytest -v` before pushing
- Tests must complete in <30 seconds total

## Pull requests

- Branch from `main`
- One logical change per PR
- Include test coverage for new code
- Squash merge preferred

## Reporting issues

GitHub Issues — please include:
- Python version + OS
- Minimal reproduction
- Expected vs actual behavior
