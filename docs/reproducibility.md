# Reproducibility

## Runtime basis

The qualified training runtime used Ubuntu 24.04 under WSL2, Python 3.13.15, mjlab 1.6.0 at commit
`8ee51fbcf806a7419189f706d9e394cbeb7790fa`, MuJoCo/MuJoCo Warp 3.11.0, Warp 1.14.0, PyTorch
2.9.0+cu128, and RSL-RL 5.5.0. The measured device was an NVIDIA RTX 5060 Laptop GPU with about
8 GB VRAM. That hardware record is evidence, not a minimum specification or capacity guarantee.

## Clean development/native setup

```console
uv sync --group dev --extra native --extra video
uv run g1-mjlab validate-config --config configs/standing-v1/train.json
uv run pytest
```

## Training setup

The `train` extra pins the exact mjlab Git revision and the qualified Torch, MuJoCo Warp, Warp, and
RSL-RL versions. CUDA wheel selection is platform-specific; confirm that the installed PyTorch
build reports `2.9.0+cu128` and supports the GPU before training. Do not install a Linux display
driver inside WSL when the Windows host supplies WSL CUDA support.

```console
uv sync --group dev --extra train
uv run g1-mjlab doctor --output artifacts/doctor-new-machine
```

Run data belongs on a fast local Linux filesystem for long training campaigns. Copy finalized,
non-executable evidence out after completion rather than training directly against synchronized
cloud storage.

## Run integrity

Each run records resolved configuration, source snapshot/hash, package versions, hardware, model
and controller contract, MDP/PPO settings, metrics, checkpoint index, evaluation scenarios, and
final status. Checkpoint names are ordered by numeric update, not lexicographically. Failed and
interrupted attempts are preserved.

Resume restores actor, critic, normalizers, optimizer, adaptive learning rate, and update counter
into freshly constructed worlds. It is not bitwise replay of environment/RNG state. Only load trusted
local PyTorch checkpoints.

Final qualification binds the selected checkpoint, ONNX graph, final backend summaries, and source
archive by SHA-256. A changed task/reward/controller is a new versioned experiment, not an overwrite
of standing-v1.
