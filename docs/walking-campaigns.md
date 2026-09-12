# Walking campaign execution and recovery

The Stage 19 campaign harness turns a long PPO experiment into bounded, inspectable processes. It
does not select a policy or claim gait improvement. Selection rules and the A/B/C experiment matrix
belong to the next development phase.

## Manifest contract

`g1-mjlab run-walking-campaign` accepts one strict JSON manifest. Every checkpoint, run config,
walking profile, reward profile, PPO profile, and scenario set is identified by both path and
SHA-256. Loading fails before GPU work if any byte stream changed. The manifest also fixes the
baseline commit, hypothesis, seed, environment count, checkpoint interval, total update budget,
segment size, smoke size, metric schema, and recovery policy.

The checked-in `configs/walking-v1/stage19/campaign.json` is a four-update local harness proof. Its
source checkpoint lives under the ignored `.runtime` tree and is not a public walking policy. A
fresh clone must obtain the exact checkpoint named by the manifest or author a new manifest with
the intended trusted checkpoint and hash.

## Lifecycle and artifacts

The state machine is:

```text
planned -> smoke_running -> smoke_passed -> training -> evaluating
                                                ^            |
                                                |------------|
                                                             -> completed
```

Any active state can terminate as `failed` or `stopped` where allowed. `state.json` is atomically
replaced and `events.jsonl` is append-only, flushed, and fsync'd. The output directory must not
exist; an old campaign is never overwritten.

Each segment gets its own generated config, log, complete training run, checkpoint index, and
schema-2 physical evaluation. Training and evaluation are blocking subprocesses, so two GPU
processes never overlap. The standard budget is 100 updates per segment; smaller values are for
harness proof only. A strict resume restores actor, critic, normalizers, optimizer, learned action
standard deviation, and effective learning rate, then begins at the completed index plus one in
fresh environments. It is not exact trajectory replay.

Every saved upstream checkpoint is copied through a temporary file and atomically promoted while
training is still active. Its index records iteration, completed updates, transitions, byte size,
SHA-256, source lineage, and `complete` validity. Hidden temporary files from an interrupted copy do
not match the checkpoint catalog pattern and cannot be selected.

Training evidence includes raw loss, reward, error, termination, throughput, effective learning
rate and action-standard-deviation records. `optimization.json` adds actor/critic deltas, finite
gradient checks, final gradient norm, and every PPO minibatch KL sample. `normalizer-state.json`
stores exact final actor and critic normalization tensors. `memory.json` records allocator and GPU
headroom evidence.

## CUDA recovery

CUDA 700 or 719 permits at most the manifest's single recovery attempt. The harness preserves the
failed output, selects only the latest indexed `complete` checkpoint, verifies that trusted
checkpoint in a fresh CPU process, and runs a fresh 64-world smoke in another process. The smoke is
discarded from the learning lineage. The failed segment is then continued from its last complete
iteration, with exact remaining-budget arithmetic. A repeated CUDA failure, any non-finite learner
state, an invalid checkpoint, or any other process failure ends the campaign with evidence.

## Commands

```bash
g1-mjlab run-walking-campaign \
  --manifest configs/walking-v1/stage19/campaign.json \
  --output .runtime/my-new-campaign

g1-mjlab evaluate-walking-checkpoints \
  --manifest configs/walking-v1/stage19/campaign.json \
  --output .runtime/my-new-campaign

python scripts/verify_walking_campaign.py \
  --campaign .runtime/my-new-campaign \
  --output .runtime/my-new-campaign/verification.json
```

The second command evaluates the source and each unique complete checkpoint in separate sequential
processes. Existing successful checkpoint evaluations are idempotently reused by checkpoint hash.
