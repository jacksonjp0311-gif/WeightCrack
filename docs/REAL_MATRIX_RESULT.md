# Real-matrix probe result (dense traces)

Date: 2026-11-10. Branch `dev/verified-projection-repair`.

## Path verified

The probe now runs end-to-end on the extracted Qwen2.5-0.5B down-projection when supplied with dense failure traces.

- Matrix: `experiments/real_matrix/mlp_down_proj.npz` (896 × 4864).
- Traces: 12 synthetic failure vectors sharing a recoverable direction (`experiments/real_matrix/traces_dense.npz`).
- Direction: top PCA component of the failure activations.
- Columns selected: 4441 and 671.
- Candidate routed through `construct_candidate`. Host gate applied. Status `probe_complete`.
- Touched coordinates: 1792. Max absolute stored delta: 0.0038.

## What this closes

The code path that was missing (require traces, consume them, apply the gate) is now exercised on a real matrix shape. No holdout was opened because the traces are synthetic and there is no behavioral score.

## Remaining gap

A practice set of real looping generations (at least 8–10 under the 4-word × 3 rule, or the Qwen3.5-0.8B MATH loop case) is still required before a zero-loss score and a sealed holdout can be opened. The matrix and the gated probe path are ready.
