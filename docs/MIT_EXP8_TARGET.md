# Real-matrix target: MIT Exp 8 loop fix

The closest published sparse surgical result is the MIT CSAIL weight-space programming edit (Astra) on Qwen3.5-0.8B that reduced generation loops on MATH-500.

## Reported result

- ~131k weights changed across two layers (layers 15 and 19 down_proj, 64 columns each in the original report).
- MATH-500 holdout (n=20, temp 0.3): base 1/20 → edit 14/20; random-direction placebo 0/20.
- Replicated bit-exact by FiveTechSoft/weights (rig_runs/holdout-08b-*.json).

## Relevance to WeightCrack

The operation is a sparse projection-style edit on MLP down-projections. It is the same algebraic family as `project_remove`. The failure mode (repetition / looping) is dense enough that localization instruments have a chance.

## What is still missing

A local extraction of one of those matrices plus the actual failure and success residual or hidden-state traces that were used to rank the columns. The probe now accepts an optional traces.npz containing `fail` (and optionally `success`) arrays. Until those files exist, the probe remains structural.

## Next concrete step

1. Extract `model.layers.15.mlp.down_proj.weight` (or the equivalent) from Qwen3.5-0.8B into a .npz.
2. Capture residual states on a short practice set of looping vs non-looping math prompts.
3. Run:
   ```
   python -m weightcrack.evaluation.real_matrix_probe matrix.npz out/ traces.npz
   ```
4. If a candidate qualifies under the gate on practice, lock the hash and open a sealed holdout once.

This is the measurement that changes the claim from “works on dense synthetic faults” to “works on a real matrix under the host gate.”
