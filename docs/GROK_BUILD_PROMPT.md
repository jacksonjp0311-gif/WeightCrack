# Prompt for Grok Build — real-matrix measurement

You are working in the WeightCrack repository on branch dev/verified-projection-repair.

Goal: produce the first host-gated real-matrix measurement.

## Steps

1. Extract one MLP down-projection from a public Qwen checkpoint that has dense repetition/loop failures on math prompts. Prefer Qwen3.5-0.8B or Qwen2.5-0.5B. Save it as experiments/real_matrix/mlp_down_proj.npz with key "weight".

2. Capture residual or hidden-state activations on a short practice set of math prompts that produce looping vs non-looping generations. Save them as experiments/real_matrix/traces.npz with arrays "fail" (required) and "success" (optional).

3. Run the existing probe:
   ```
   python -m weightcrack.evaluation.real_matrix_probe experiments/real_matrix/mlp_down_proj.npz experiments/real_probe experiments/real_matrix/traces.npz
   ```

4. If a candidate qualifies under the zero-loss gate on the practice set, lock the finalist hash, open a sealed holdout once, and record the behavioral numbers plus the receipt. Write the result to docs/REAL_MATRIX_RESULT.md.

5. Commit the matrix extraction script, the traces (or a generator that reproduces them), the probe report, and the result note. Push to the same branch.

## Constraints

- Do not expand the synthetic ranking instruments.
- Do not bypass the host gate. Route any proposal through construct_candidate.
- If download fails, document the exact error in docs/REAL_MATRIX_STATUS.md and stop.
- The success criterion is a policy-admitted edit on a real matrix that moves held-out behavior on a dense failure mode.

Report the final commit hash and the contents of docs/REAL_MATRIX_RESULT.md.
