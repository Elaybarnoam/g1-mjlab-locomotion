# Architecture

The project has five deliberate layers. Dependencies point inward toward small contracts and
artifact schemas; importing the package does not initialize a simulator.

```text
standing task + controller adapter
              │
              ▼
    mjlab vector environment ──► RSL-RL PPO runner
              │                         │
              │                         ▼
              └──────────────► run artifacts/checkpoints
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                    evaluation       reporting       ONNX export
                                                            │
                                                            ▼
                                               native MuJoCo adapter
```

## Ownership

- `config.py`, `contracts.py`, `evaluation.py`, `checkpoints.py`, and `artifacts.py` own validated
  data and pure logic.
- `standing_task.py` owns the versioned standing MDP.
- `training.py`, `runtime.py`, and `campaign.py` own simulator lifecycle, PPO orchestration, bounded
  retries, and telemetry; `diagnostics.py` owns bounded control and checkpoint probes.
- `deployment.py` owns frozen-bundle integrity, native observation/action reconstruction, ONNX
  inference, evaluation, and transfer parity. `native_media.py` owns recording and interactive
  playback.
- `reporting/` reads non-executable artifacts and renders offline HTML without importing mjlab.

## Durable seams

The controller contract records joint order, observation fields, action scales, nominal pose,
gains, force limits, sensor sources, timing, and model hash. Native inference rejects mismatched
contract/model/policy/scenario hashes. Evaluation records terminal state before reset and freezes a
trial after its first episode boundary.

Training checkpoints are trusted resume objects. ONNX is the deployment interchange format. The
two are never presented as equivalent security formats or interchangeable lifecycle artifacts.
