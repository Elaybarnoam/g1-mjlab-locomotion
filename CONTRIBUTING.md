# Contributing

Contributions are welcome through focused issues and pull requests.

## Development

```console
uv sync --group dev --extra native --extra video
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -m "not simulator and not gpu and not slow" --cov --cov-report=term-missing
uv run python scripts/check_repository.py
uv build
uv run twine check dist/*
uv run check-wheel-contents dist/*.whl
```

Install the local hooks once with `uv run pre-commit install`. The GitHub workflows rerun the
authoritative checks on Python 3.12 and 3.13; hooks are an early feedback mechanism, not a CI
replacement.

Use typed interfaces, document physical units and coordinate frames, and keep simulator imports out
of lightweight modules. A behavior change must update its versioned task/configuration and include
evaluation evidence. Never rewrite historical experiment results.

GPU tests must be explicitly selected on qualified hardware. A skipped GPU test is not evidence of
physical correctness.

By submitting a contribution, you agree that it may be distributed under this project's MIT
License and that you have the right to provide it under those terms.
