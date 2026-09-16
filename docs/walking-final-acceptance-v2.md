# Walking final acceptance v2

Walking-v2 qualification is calculated from raw, pre-reset trial measurements. Development
`functional_passed`, `style_passed`, or externally supplied `accepted` values are never consumed by
the final reducer.

Each backend result is bound to the same checkpoint, ONNX graph, MuJoCo model, controller, policy
contract, reference bank, host profile, evaluator, acceptance profile, scenario set, trial IDs,
seeds, categories, and initial states. The suite contains exactly 100 trials: 25 slow, 25 medium,
25 fast, and 25 stand-walk-stop trials. A backend passes with at least 95 accepted trials overall
and at least 23 in every category. Both mjlab and native MuJoCo must pass independently.

The numerical thresholds and measurement conventions are frozen in
[`final-acceptance-v2.json`](../configs/walking-v2/final-acceptance-v2.json) and mapped to their
implementation and tests in [`criteria-map.json`](../configs/walking-v2/criteria-map.json). Missing
measurements, inadequate walk/contact samples, non-finite state, early termination, or forbidden
ground contact fail closed.

The historical walking-v1 evaluator remains available for development reproduction. Its looser
windows and stored pass flags cannot be converted into walking-v2 final evidence.

Human review uses visual-review schema 2. It records one `accepted` or `rejected` decision and binds
the exact checkpoint, model, reference, contract, and ordered videos. The explicit legacy adapter
maps only schema-1 `approved` to schema-2 `accepted`; arbitrary truthy values are rejected.
