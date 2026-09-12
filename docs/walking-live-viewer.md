# Walking development policy: live deterministic viewer

The live viewer runs one `G1-Walking-Flat-v1` world with the same mjlab environment and
deterministic actor-mean inference path used by headless evaluation. It does not construct an
optimizer, sample the Gaussian action distribution, apply domain randomization, or add observation
noise. The loaded observation normalizer is frozen.

The GPU MuJoCo-Warp state is authoritative. MJLab copies one world's state into passive CPU MuJoCo
buffers for display and calls `mj_forward` only to reconstruct visual kinematics and contacts. The
viewer never advances a second physics simulation.

## Run

From the Linux/WSL training environment, install the training extra and provide a trusted local
checkpoint:

```console
uv sync --extra train
uv run g1-mjlab play-walking \
  --config configs/walking-v1/bootstrap-train.json \
  --checkpoint .runtime/walking-bootstrap-resume-10600-64/checkpoints/model_13398.pt \
  --schedule configs/walking-v1/bootstrap-development-schedule.json \
  --seed 10042 \
  --viewer native
```

The schedule requests two seconds standing, ten seconds at 0.6 m/s, and three seconds standing.
After it completes, the viewer holds the standing command until the window closes. Add `--loop` to
reset only after the complete schedule, or `--duration-seconds N` for a bounded smoke run.

Controls:

- `Space`: pause or resume without advancing policy phase.
- `R`: explicit environment, policy-schedule, action-history, and sensor-history reset.
- `-` / `+`: reduce or increase target playback speed.
- right arrow: one policy step while paused.
- mouse: retain native MuJoCo orbit and zoom controls.

The overlay names the checkpoint and hash prefix and shows simulated time, episode/reset counter,
requested and filtered forward speed, gait phase, state, and measured real-time factor. A true
termination pauses on the terminal frame and names the reason; it is never hidden by auto-reset.

## Determinism and performance

The policy input is the documented 102-value actor vector and output is the 29 joint-position
offset action. The simulator integration test captures the first 100 physical actor inputs from a
headless rollout and verifies that the viewer adapter produces the same 100 actions within
`atol=1e-6, rtol=1e-5`. It also exercises a real sensor-history reset after those steps.

The viewer targets 50 control steps per wall-clock second. If this machine cannot sustain that
rate, MJLab preserves policy-step order and reports a real-time factor below 1.0; it does not skip
policy steps. WSLg may use software OpenGL for display even while MuJoCo-Warp physics runs on CUDA,
which can make the interactive path substantially slower than headless evaluation.

Only load checkpoints you created or otherwise trust. PyTorch checkpoint files are executable
serialization, not a safe public interchange format.
