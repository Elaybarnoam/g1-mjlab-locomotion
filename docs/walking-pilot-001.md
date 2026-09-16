# Walking pilot 001 — diagnosed, not qualified

The first walking pilot completed 1,000 PPO updates: 256 robots, 24 steps per rollout, and 6,144,000
total transitions. Optimization remained finite and produced checkpoints every 100 updates. This is
a completed training run, but it is **not a working walking policy**.

Training telemetry appeared to improve because mean episode length rose and
`Metrics/twist/command_error_m_s` became small. W08 proved that this old name measured only the
requested-versus-rate-limited command state, not the robot's achieved velocity. The field is now
renamed `command_filter_error_m_s`, and future training logs also include physical base-velocity
tracking error.

Deterministic stand→walk→stand evaluation gave:

| Checkpoint | Mean first-episode survival | Best counted steps | Achieved-speed RMS error | Both gates |
| --- | ---: | ---: | ---: | ---: |
| 100 | 1.46 s | 0 | unavailable before fall | 0/4 |
| 300 | 1.94 s | 0 | unavailable before fall | 0/4 |
| 500 | 15.00 s | 5 | 1.061 m/s | 0/4 |
| 700 | 15.00 s | 0 | 1.062 m/s | 0/4 |
| 900 | 15.00 s | 4 | 1.066 m/s | 0/4 |
| 999 | 15.00 s | 0 | 1.066 m/s | 0/4 |

Checkpoint 500 is retained only as the best *development diagnostic*, not a release candidate. Its
recorded video trial survived 15 seconds and stopped cleanly, but its six contact events had a median
step length of only 0.0225 m, cadence 0.544 steps/s, zero alternation, and 100% tiny steps. It failed
command tracking, cadence, alternation, and shuffling gates.

<video controls src="assets/walking-v1/pilot-001-shuffling.mp4"></video>

The run demonstrates collapse to standing/stabilization rather than forward walking. More unchanged
training is not justified. W09 should run one-variable experiments that first make physical forward
progress learnable, then strengthen reference pose/contact timing without sacrificing balance. The
reviewed numerical record is
[`evidence/walking-v1/pilot-001.json`](../evidence/walking-v1/pilot-001.json).
