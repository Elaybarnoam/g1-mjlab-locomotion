# Walking native inference

The walking controller has one authoritative, repository-independent native path. The ONNX graph
contains the deterministic actor mean and its frozen observation normalizer. MuJoCo remains the
only physics authority; the host maps each 29-value normalized action to the contract's joint
targets exactly once.

The current development artifact is deliberately **not a qualified walking release**. Its status is
`unqualified_development`, and loading it requires the explicit
`--allow-unqualified-development` flag. The supported development command interval is 0–0.6 m/s.

## Export an exact checkpoint

```bash
uv run --extra train g1-mjlab export-walking \
  --run RUN_DIRECTORY \
  --checkpoint model_99.pt \
  --output WALKING_BUNDLE
```

The new directory binds `policy.onnx`, `model.mjb`, `contract.json`, `host-profile.json`, and
`reference.npz` by SHA-256. Export validates `[batch, 102] -> [batch, 29]` and records the exact
checkpoint hash. Training dependencies are needed only for this export.

## Run with native CPU dependencies

```bash
uv sync --extra native
uv run g1-mjlab evaluate-native-walking \
  --policy WALKING_BUNDLE \
  --scenarios configs/walking-v1/stage19/development-scenarios-v3.json \
  --output NATIVE_RESULT \
  --allow-unqualified-development

uv run g1-mjlab play-policy \
  --policy WALKING_BUNDLE \
  --forward-speed .6 \
  --allow-unqualified-development
```

Commands outside the bundle's declared domain fail before simulation advances. Every reset uses a
serialized full `qpos`, `qvel`, and phase; the training run's incidental initial pose is not treated
as evaluation state.

## Control ordering

At control sample *t*, the actor observes the current physical sensors, previous applied action,
applied command, phase, and blend. It produces one action. The host then computes state for sample
*t+1*, maps the action to joint targets, and MuJoCo advances four 5 ms physics steps. The next actor
observation combines those new sensors with the already-advanced host state. The executable fixture
is `tests/fixtures/walking-control-timeline-v1.json`.

The P04-09 parity proof compares 100 live mjlab states field-by-field at `atol=1e-5, rtol=1e-4`
and compares PyTorch to ONNX actions at `atol=1e-4, rtol=1e-4`. Long trajectories are evaluated
statistically because CPU MuJoCo and GPU MuJoCo-Warp contact solvers are not bitwise identical.
