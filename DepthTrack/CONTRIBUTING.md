# Contributing

Thanks for your interest in improving the project!

## Development setup

```bash
git clone <repo-url>
cd DepthTrack
python -m venv .venv && source .venv/bin/activate
make dev          # installs the package with dev extras
```

## Workflow

1. Create a feature branch.
2. Make your change, keeping modules focused and adding type hints + docstrings.
3. Run the checks below; all must pass.
4. Open a pull request with a clear description and, where relevant, a sample
   before/after frame or clip.

## Checks

```bash
make test       # pytest
make lint       # ruff
make typecheck  # mypy
```

## Guidelines

- Keep new tunable values in `config.py`, not hard-coded in logic.
- Prefer pure, testable functions for new perception/geometry code.
- Heavy imports (torch, ultralytics, transformers) should stay lazy where it
  helps keep `--help` and tests fast.
