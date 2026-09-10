# Evaluation and evidence

## Protocol

Development checkpoint selection ranks strict passes, survival passes, lower median drift, lower
median tilt, and then later update. Final-test scenarios use a distinct seed and are evaluated only
after selection. The frozen final gate requires at least 95 of 100 strict 60-second trials in both
mjlab/MuJoCo Warp and native MuJoCo/ONNX.

A strict pass requires finite state, no true fall termination, continuous foot support except for at
most 0.10 seconds, root drift no greater than 0.20 m, and—after one settling second—torso tilt no
greater than 15 degrees and pelvis height at least 0.60 m. Survival is reported separately so a
motion-quality failure cannot be disguised as standing success.

Auto-reset is disabled while measuring. The terminal frame is recorded before reset, termination
wins over a simultaneous timeout, and trial state becomes immutable after its first episode ends.

## Results

Three independently trained seeds produced development-qualified policies. The selected seed-42
`model_999.pt` checkpoint passed 100/100 final trials in each backend. The two-sided 95% Wilson
interval for 100/100 is 0.9630–1.0. Median/worst mjlab root drift was 0.0215/0.0407 m; maximum torso
tilt was 3.38 degrees; minimum post-settling height was 0.701 m.

The native adapter replayed the same saved initial states with the exported ONNX actor. A separate
31-state parity corpus passed with maximum observation-field error `5.97e-8` and maximum inference
error `4.18e-7`.

[`evidence/standing-v1/summary.json`](../evidence/standing-v1/summary.json) is the small public result
record. It contains hashes for the private executable evidence but not the checkpoint, ONNX model,
compiled model, or raw scenario states themselves.

| Bound artifact | SHA-256 |
| --- | --- |
| Selected checkpoint | `9541715124c463d5008275f523b0855d6d7ba079c41e964a8bb6a2b74f1ff6ae` |
| ONNX actor and normalizer | `01373e0deb56136a7c758a7a3b341db31c4bad323d54c6657f4b9cb64d0635cb` |
| Controller contract | `c313b4e525f8b32cf111155c021a284ce794e25b62b088585da3fcaba84f4546` |
| Compiled MuJoCo model | `9294f78097cbe222d1ec06c62492c6bb4b7a18bb13ee8857e33827189006c33f` |
| Final mjlab evidence | `f2dad4d74f47caba9519d293fd3e13793c4f8bba347b7df7f5afaf3625d2db17` |
| Final native evidence | `13c9034df3617c84c86f6e67cd7fa6f74218e48166d07d1c25134e3337c1f807` |

## Interpretation

The result qualifies nominal flat-ground standing within the tested reset distribution. It does not
measure cross-engine variation, broad physics variation, disturbances, terrain, sensor estimation,
latency, actuator calibration, thermal/current limits, or real-robot safety. The deterministic video
is supporting visual evidence, not a substitute for the final trial table.
