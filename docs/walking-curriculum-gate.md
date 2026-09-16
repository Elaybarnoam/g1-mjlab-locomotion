# Walking speed-curriculum prerequisite

The speed curriculum is fail-closed. It may start only after the Stage 19 experiment table names a
development-qualified 0.6 m/s checkpoint and a human owner approves normal-speed video of that
exact checkpoint. Approval is a strict JSON record bound to the checkpoint SHA-256; approval of a
different model is rejected.

```bash
g1-mjlab check-walking-curriculum-prerequisite \
  --experiment-table .runtime/plan04-p07-experiment-table-001/experiment-table.json \
  --visual-approval .runtime/visual-approval.json \
  --output .runtime/curriculum-prerequisite.json
```

The command exits with status 2 when blocked and writes the reasons atomically. In the current
Stage 19 result there is no qualified checkpoint and no visual approval. Speed-curriculum,
seed-replication, and candidate-freeze training are therefore forbidden. This is an intentional
scientific stop, not an infrastructure failure; bypassing it would turn a failed 0.6 m/s hypothesis
into an unsupported multi-speed claim.
