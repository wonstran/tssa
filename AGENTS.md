# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python package built from `src/`. The main package lives in `src/tssa/` and is split by responsibility:

- `core/`: analysis workflows such as counting, lane, stop, yield, and near-miss logic
- `engine/`: shared geometry, line, zone, and view-transform utilities
- `io/`: file parsing and field definitions
- `transform/`: data conversion and homography helpers
- `annotation/`: annotation-related helpers

Keep new modules inside the closest existing package and avoid adding top-level scripts unless they are broadly reusable.

## Build, Test, and Development Commands
Use a virtual environment with Python 3.9+.

- `python -m pip install -e .`: install the package in editable mode
- `python -m pip install -r requirements.txt`: install runtime dependencies, including OpenCV packages listed outside `pyproject.toml`
- `python -m build`: create source and wheel distributions
- `python -m pytest`: run tests if a `tests/` suite is added

There is no checked-in test suite today, so validate changes with focused module-level scripts or notebooks before opening a PR.

## Coding Style & Naming Conventions
Follow the existing style: 4-space indentation, module-level docstrings, and `snake_case` for functions, variables, and file names. Classes use `PascalCase` such as `Count`. Prefer type hints on public methods and keep packages under `tssa` narrowly scoped to one concern. If you introduce formatting or linting, use tools that respect the current code style rather than reformatting the full repository.

## Testing Guidelines
Place future tests under `tests/`, mirroring package paths where possible, for example `tests/core/test_count.py`. Name test files `test_<module>.py` and cover geometry edge cases, track-file parsing, and DataFrame schema assumptions. For bug fixes, add one regression test per behavior change.

## Commit & Pull Request Guidelines
Recent history favors short, direct commit subjects such as `add track-zone matching functions` and `updated count module 2`. Keep commits focused and use imperative, descriptive subjects under about 72 characters. PRs should include:

- a short problem/solution summary
- affected modules or input formats
- validation steps or sample data used
- screenshots only when outputs are visual

## Data & Configuration Notes
Do not commit large generated CSVs or derived artifacts unless they are required fixtures. Prefer referencing local data paths in docs or examples instead of hard-coding machine-specific locations.
