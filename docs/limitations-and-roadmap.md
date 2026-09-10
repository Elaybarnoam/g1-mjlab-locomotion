# Limitations and roadmap

## Current limitations

- mjlab/MuJoCo Warp and native MuJoCo share the MuJoCo model family; transfer is not cross-engine.
- Final tests use flat ground, modest reset variation, and no controlled pushes.
- Dynamics, contact, mass, friction, latency, sensor noise, and actuator properties are not broadly
  randomized.
- The actor consumes simulated pelvis-site linear velocity unavailable directly from a real IMU.
- Hardware gain mapping, current/thermal limits, communications, emergency stop, and fall-safety
  infrastructure are outside the project.
- Standing-v1 is balance from an upright reset, not get-up behavior.
- Walking and walk-to-stand transitions are not implemented.

## Ordered roadmap

1. Add a separately versioned push-recovery suite and controlled disturbance curriculum.
2. Add physical/domain randomization with held-out parameter distributions.
3. Replace pelvis velocity with a deployable estimator or train under estimator outputs.
4. Qualify another physics engine to establish true cross-engine sim-to-sim transfer.
5. Introduce walking-v1 commands while retaining zero-command standing as a regression gate.
6. Add walk-to-stand and stand-to-walk transition suites.
7. Build a hardware safety/control interface only after estimator, actuator, latency, and emergency-
   stop requirements are independently reviewed.

Every stage receives new development and untouched final suites. Passing one stage does not inherit
qualification for the next.
