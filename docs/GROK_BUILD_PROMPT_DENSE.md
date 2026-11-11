# Prompt for Grok Build — denser traces and gated real-matrix run

You are working in the WeightCrack repository on branch `dev/verified-projection-repair` (head `9568580` or later).

## Current state

- Matrix extracted: `experiments/real_matrix/mlp_down_proj.npz` from `Qwen/Qwen2.5-0.5B` layer 23 `mlp.down_proj.weight` (shape 896×4864).
- Probe updated: requires `traces.npz` with a `fail` array, computes PCA direction from those traces, ranks columns by write magnitude, and routes the candidate through `construct_candidate` so the host gate is applied. It refuses random-vector fallback when a traces path is given.
- Prior practice set had only 1 looping prompt out of 18. That is not dense enough for a reliable direction or a zero-loss score.

## Goal

Produce a policy-admitted edit on the real matrix that moves held-out behavior on a dense failure mode, or document why the gate rejected every candidate.

## Steps

1. Expand the practice set until at least 8–10 prompts produce loops under the same 4-word × 3 rule used previously, or switch to the Qwen3.5-0.8B MATH loop case documented in `docs/MIT_EXP8_TARGET.md`. Save the failure activations as `experiments/real_matrix/traces.npz` with key `fail` (shape `(n_fail, hidden)`). Optional: also save `success`.

2. Run the probe:
   ```
   python -m weightcrack.evaluation.real_matrix_probe \
     experiments/real_matrix/mlp_down_proj.npz \
     experiments/real_probe \
     experiments/real_matrix/traces.npz
   ```

3. If the gate admits a candidate on practice, lock the finalist hash, open a sealed holdout once, and record the behavioral numbers plus the receipt in `docs/REAL_MATRIX_RESULT.md`.

4. If the gate rejects every candidate, record the rejection reasons and the number of failure traces used. Do not open holdout.

5. Commit the trace generator (or the traces themselves if small), the probe report, and the updated result note. Push to the same branch.

## Constraints

- Do not expand synthetic ranking instruments.
- Do not bypass `construct_candidate`.
- If the model or traces cannot be obtained, document the exact error in `docs/REAL_MATRIX_STATUS.md` and stop.
- Success criterion: a policy-admitted edit on a real matrix that moves held-out behavior on a dense failure mode, with the receipt written.

Report the final commit hash and the contents of `docs/REAL_MATRIX_RESULT.md`.
