# Contributing

Contributions are welcome through focused issues and pull requests.

## Development

```console
uv sync --group dev --extra native --extra video
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Use typed interfaces, document physical units and coordinate frames, and keep simulator imports out
of lightweight modules. A behavior change must update its versioned task/configuration and include
evaluation evidence. Never rewrite historical experiment results.

GPU tests must be explicitly selected on qualified hardware. A skipped GPU test is not evidence of
physical correctness.

By submitting a contribution, you agree that it may be distributed under this project's MIT
License and that you have the right to provide it under those terms.
