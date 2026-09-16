# Walking-v1 status and reproduction

Walking-v1 is the Unitree G1 forward-locomotion research task. It uses PPO with an asymmetric
actor–critic: the deployed actor receives 102 values and emits 29 normalized joint actions; the
training-only critic receives 114 values, adding privileged foot/contact state.

## Active development baseline

The active baseline is the **pre-style nominal-action bootstrap**, checkpoint `model_13398.pt`
(`e3cbd37746fbfad9d6cce90bcbfc5371d851c619ffcda7b59aac8c3f6783cd64`). Its local run is
`.runtime/walking-bootstrap-resume-10600-64`; those private artifacts are not distributed in Git.
The selection and the two non-equivalent evaluation results are recorded in
[`active-development-baseline.json`](../configs/walking-v1/active-development-baseline.json).

The earlier functional evaluator passed 16/16 stand→walk→stand trials, while a later stricter
physical-contact/slip measurement passed 0/4. Both evaluations failed their style gates. The
[15-second deterministic replay](assets/walking-v1/bootstrap-pre-style-13398.mp4) shows what this
checkpoint actually does. It is **not** proof of a qualified gait.

For new development, retain the nominal joint-offset controller and first address foot-contact
chatter and physical stance slip. Do not initialize a new experiment from Stage 18/19 style-tuned
checkpoints or the walking-v2 reference-residual policy unless the experiment explicitly declares
that different baseline. Standing-v1 and all historical walking evidence remain unchanged.

## Later style-tuning result (historical)

Stage 19 evaluated three bounded mechanics/reward hypotheses for 800 PPO updates and 1,228,800
transitions. The best observed Arm C checkpoint passed walking function in 16/16 frozen development
scenarios, but passed combined function and human-style gait in 0/16. It is therefore an
**unqualified development artifact**, not a walking release. Further reward-only PPO training and
the hidden final suite are blocked by the predeclared decision rules.

Exact checkpoint SHA-256:
`0865d076a7820aba3bdd1d8d97d0d60194c7989e92919b510514ffde17f929e9`.

## Architecture

- Actor: `102 → 512 → 256 → 128 → 29`, ELU, frozen empirical input normalization at inference.
- Critic: `114 → 512 → 256 → 128 → 1`, ELU, used only during PPO training.
- Control frequency: 50 Hz; each action is held for four 5 ms MuJoCo physics steps.
- Actuation: position targets with the model's joint-specific Kp, Kd, armature, and force limits.
- Host state: acceleration-limited command, hysteretic stand/walk mode, blend, and speed-scaled gait
  phase. Requested command is set before inference; the actor sees current applied state.

See [walking-theory.md](walking-theory.md), [walking-policy-spec.md](walking-policy-spec.md),
[walking-evaluation.md](walking-evaluation.md), and
[walking-native-inference.md](walking-native-inference.md).

## Reproduce the retained development artifact

Training/export requires the pinned GPU environment. Native evaluation and playback require only
the `native` extra after a bundle has been exported.

```bash
uv run --extra train g1-mjlab export-walking \
  --run RUN_DIRECTORY \
  --checkpoint model_99.pt \
  --output WALKING_BUNDLE

uv run g1-mjlab evaluate-native-walking \
  --policy WALKING_BUNDLE \
  --scenarios configs/walking-v1/stage19/development-scenarios-v3.json \
  --output NATIVE_OUTPUT \
  --allow-unqualified-development
```

The opt-in flag is intentional. No documentation or finite-rollout result changes qualification.
