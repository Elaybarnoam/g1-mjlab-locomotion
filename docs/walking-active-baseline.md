# Active walking baseline: pre-style locomotion bootstrap

Status: **unqualified development baseline**. No walking policy is released or recommended for
hardware. The qualified standing-v1 release is independent.

The selected Unitree G1 checkpoint is `model_13398.pt` from the local run
`.runtime/walking-bootstrap-resume-10600-64`, SHA-256
`e3cbd37746fbfad9d6cce90bcbfc5371d851c619ffcda7b59aac8c3f6783cd64`.
The [machine-readable baseline](../configs/walking-v1/active-development-baseline.json) binds this
choice, the evaluation evidence, and the replay. Git does not contain the PyTorch checkpoint.

## What this pivot changes

- The next walking experiments start from walking-v1's nominal joint-position offset policy:
  `q_target = q_nominal + action_scale * actor_output`.
- The actor uses the existing 102-value observation and emits 29 joint actions; PPO's 114-value
  critic is training-only. The environment remains the existing `G1-Walking-Flat-v1` task.
- The human-reference and reference-residual walking-v2 experiments remain available for audit,
  but are not the active development baseline. No files or checkpoint histories are deleted.
- This is a development selection, **not** a reset of failed gates or a walking release.

## Evidence and remaining failure

The earlier frozen bootstrap evaluator reported functional stand→walk→stand success in 16/16
development trials and human-style success in 0/16. Its mean command RMS was 0.0843 m/s, and
all trials survived 15 seconds. It also measured contact chatter, shuffling, and stance slip.

The later schema-2 physical measurement re-evaluated the *same checkpoint* under stricter
contact/slip semantics: functional success 0/4 and style success 0/4. The dominant findings were
physical stance slip and physical contact chatter. These results are not interchangeable; the
stricter failures remain binding for future qualification. Local evidence is at
`.runtime/eval-bootstrap-final-13398/summary.json` and
`.runtime/plan04-measurement-bootstrap-003/summary.json`.

[Watch the 15-second deterministic MuJoCo replay](assets/walking-v1/bootstrap-pre-style-13398.mp4).
It is a render of the saved schema-2 simulator trajectory with the original G1 visual meshes,
not a learning run or a separate physics simulation. The replay is 960×720 at 30 fps. It is
provided for visual review because the WSLg live viewer is unreliable on this machine.

## Next experiment boundary

Hold the task, observation/action contract, PD model, and checkpoint fixed while measuring and
reducing raw foot-contact transitions and physical stance slip. Declare any reward, contact sensor,
reset, or curriculum change as a new experiment. Re-run both stand→walk→stand and the stricter
physical measurement before considering human-style tuning. Do not promote or distribute the
bootstrap checkpoint without passing the declared qualification gates.
