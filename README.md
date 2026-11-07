# WeightCrack

Host-gated sparse projection edits on NumPy matrices, with coordinate-level accounting and exclusive artifact creation.

## What is implemented

- `project_remove` on a single matrix stored as `weight` in an `.npz`.
- Working precision float64; storage dtype of the source is preserved.
- Strengths in (0, 2]; values above 1 reverse the projected component.
- Policy enforces `max_entries` and `max_fraction` on the union of touched coordinates, and `max_abs_delta` on the realized stored difference after casting.
- Exclusive output bundles: no overwrite of existing candidates or receipts; source/output aliases are rejected.
- Receipts bind source content hash, candidate file hash, candidate content hash, patch, policy, and accounting.
- Practice-only localization instruments (mean-difference direction, column ranking by absolute projection) and a shared trial path that scores the same rounded candidate that would be saved.

## What is not implemented

- Safetensors / Hugging Face loading.
- Multiple tensors, low-rank updates, or the full preservation-memorization solver.
- Real language-model evaluation.
- Process-level isolation beyond ordinary path separation.

## Bounded experiment

A single synthetic 8-16-4 classifier with one fixed rank-one fault on two columns of W. Practice-only mean-difference direction, heuristic vs random column selection, eight doses, held-out evaluation after locking.

Command:

```
python -m weightcrack.evaluation.synthetic_experiment experiments/synthetic_v1
```

On the pre-registered seed the result was NO_QUALIFIED_CANDIDATE for both methods. That is a completed measurement, not a failure of the harness.

## Evidence boundaries

External MATH/GSM/IFEval numbers and WeightBridge LoRA studies are not measurements of this repository. The earlier phase tests exercised matrix-norm callbacks and suite hashes; they did not demonstrate sector improvement.
