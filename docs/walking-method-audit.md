# Walking method audit

`g1-mjlab audit-walking-method` diagnoses the unqualified walking-v1 policy before
walking-v2 is constructed. It never updates policy weights. Every input is a
repository-relative path paired with its SHA-256 digest; missing, changed, duplicate,
absolute, or parent-traversing inputs fail before simulation.

```bash
g1-mjlab audit-walking-method \
  --manifest .runtime/plan05-p03-method-audit-manifest.json \
  --output .runtime/plan05-p03-method-audit-004
```

The command emits:

- `reference-dynamics.json`: timing, derivative, seam, quaternion, contact-duty, sole,
  and legacy damping/armature proxy evidence. The proxy is explicitly not constrained
  inverse dynamics.
- `phase-dependence.json` and `phase-sweep-actions.npz`: 1,024 trace-derived physical
  states, each evaluated at 16 phase angles with only actor indices 99–100 changed.
- `observation-adequacy.json`: actor/critic dimensions, empirical variance, frozen
  normalizer variance, phase-unit-circle checks, and expected fixed command channels.
- `reward-return.json` and `reward-return-samples.npz`: 10 fresh stochastic frozen-policy
  batches of 64 environments × 24 steps. The raw archive includes reward, critic value,
  GAE return, normalized advantage, done flags, raw terms, weighted reward rates, term
  weights, and seeds. No optimizer update is executed.
- `method-audit.json`: tri-state hypotheses and hashes of every result.
- `method-decision.json`: the selected primary method and unresolved uncertainties.

Correlations with advantage are associations, not causal evidence. Frozen-phase and
phase-shuffle probes are distribution-shift diagnostics, not candidate evaluation. The
audit deliberately leaves constrained inverse dynamics and human visual naturalness to
later Plan 05 gates.
