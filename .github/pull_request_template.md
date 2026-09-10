## Summary

Describe the change and why it is needed.

## Validation

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run mypy`
- [ ] `uv run pytest -m "not simulator and not gpu and not slow" --cov --cov-report=term-missing`
- [ ] `uv run python scripts/check_repository.py`

List any simulator, native-backend, or hardware checks run separately.

## Research and safety impact

- [ ] Observation/action ordering and controller timing are unchanged, or documented below.
- [ ] Reward, termination, PPO, PD, or model changes are explicit and evidence-backed.
- [ ] Claims distinguish training metrics, simulated behavior, transfer parity, and hardware proof.
- [ ] No checkpoint, model export, raw artifact, secret, or private machine path is committed.

## Evidence and compatibility

Link the issue/spec and identify schema, configuration, or public API compatibility effects.
