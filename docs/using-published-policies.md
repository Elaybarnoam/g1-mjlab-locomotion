# Using published policies

Published policies are immutable GitHub Release assets. The repository does not store large neural
network or compiled simulator binaries in Git.

## Run standing-v1

Requirements are Python 3.12 or 3.13 and `uv`. Native deterministic playback is CPU-compatible and
does not construct PPO, an optimizer, or a training environment.

```console
git clone --branch v0.2.0 https://github.com/Elaybarnoam/g1-mjlab-locomotion.git
cd g1-mjlab-locomotion
uv sync --extra native
uv run g1-mjlab install-policy standing-v1
uv run g1-mjlab play-policy --policy policies/standing-v1
```

`install-policy` downloads the release archive, verifies its pinned SHA-256, rejects unsafe archive
members, validates the internal model/ONNX/contract/checkpoint bindings, and installs atomically.
Running it again verifies and reuses the existing installation.

The installed bundle contains:

| File | Purpose |
| --- | --- |
| `checkpoints/policy.onnx` | Deterministic actor, weights, and observation normalizer |
| `model.mjb` | Exact compiled MuJoCo model used for qualification |
| `contract.json` | Ordered observations, actions, joints, gains, limits, and timing |
| `policy-bundle.json` | Cryptographic identity and qualification bindings |
| `scenarios.json` | Frozen final-test initial states for deterministic playback |

## Resume training

The separate `model_999.pt` release asset contains the full RSL-RL learner checkpoint. Treat it as
trusted executable-style serialization: download it only from this repository's release and verify
it against `SHA256SUMS.txt` before loading it. Resume creates fresh simulator worlds and random
number-generator state; it does not reproduce the original trajectory bit for bit.

```console
uv sync --extra train
uv run g1-mjlab train \
  --config configs/standing-v1/train.json \
  --resume /path/to/model_999.pt \
  --output runs/standing-v1-resumed
```

## Safety and scope

This release is a simulation research artifact. It has no hardware state estimator, hardware
communication layer, emergency-stop integration, latency qualification, actuator-temperature
protection, or real-world validation. Do not send its outputs to a physical robot.
