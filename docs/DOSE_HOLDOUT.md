# Dose holdout

Date: 2026-11-10. Branch `dev/verified-projection-repair`.

Strengths 1.0, 1.5, and 2.0 were admitted. Each sealed holdout was opened once. None of them moved held-out loops. The success criterion is not met.

## Source

The matrix and the eight failure traces are the blobs already stored in commit `591e289`. They were still present at `bf5e7bc`, so nothing was re-captured. Fail shape is `(8, 1024)`. The matrix is `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`, tensor `model.language_model.layers.15.mlp.down_proj.weight`, float32 shape `(1024, 3584)`, content SHA-256 `b75ed374dddf0d5c9443f12b5d7d7af05d5234d264965911419ef3616fb0144d`.

Columns stay 2056 and 1309. The direction is the float64 vector in `experiments/real_probe/finalist.wcbundle/receipt.json` at `normalized_patch.direction_f64_hex`. It was not recomputed from a new ranking.

The strength-0.5 holdout was not opened again. Its recorded baseline is 4 of 6, with labels true, false, true, true, false, true. That baseline was not regenerated.

## Gate

Each strength was built with `construct_candidate` and the probe policy: `allow_tensors` `weight`, `max_entries` 10000, `max_fraction` 0.05, `max_abs_delta` 1.0. No strength was rejected. Each candidate hash was written to `experiments/real_probe_dose/dose_holdout_started.json` before any holdout prompt was tokenized. `apply_patch` then published a new bundle. Touched coordinates are 2048, which is 1/1792 of the matrix.

| strength | candidate content SHA-256 | max abs stored delta | bundle |
| --- | --- | --- | --- |
| 1.0 | `1d62852a6a025be8c69a58aa0de7f9bb69b8b6ff694a84640f6aea1fc8558a22` | 0.006556929787620902 | `experiments/real_probe_dose/strength-1.0.wcbundle` |
| 1.5 | `6a8b15ba00a25afc7d9a94584e5b446a6eca41a5bd2fd668af2f9de8e64bf630` | 0.009835394834226463 | `experiments/real_probe_dose/strength-1.5.wcbundle` |
| 2.0 | `bcd7928103bf3901ab938e931550ded594c573931f66c5479ceb205cc085e50f` | 0.013113859808072448 | `experiments/real_probe_dose/strength-2.0.wcbundle` |

## Holdout

Opened once per finalist. Six sealed prompts. Thinking enabled. Greedy decoding, 384 new tokens. The 4-word rule is unchanged. Only the edited matrix was generated. Baseline labels are copied from `experiments/real_probe/holdout.json`.

| id | baseline | strength 1.0 | strength 1.5 | strength 2.0 |
| --- | --- | --- | --- | --- |
| h01 | true | true | true | true |
| h02 | false | false | false | false |
| h03 | true | true | true | true |
| h04 | true | true | true | true |
| h05 | false | false | false | false |
| h06 | true | true | true | true |

Each edited pass looped on 4 of 6 prompts. Every prompt kept its baseline label. Held-out behavior did not move at any of the three strengths.

Details: `experiments/real_probe_dose/dose_holdout.json`. Runner: `experiments/real_probe_dose/dose_holdout.py`.

`docs/DOSE_GRID.md` records a separate synthetic Qwen2.5-0.5B admission grid. It is not this measurement.

## Criterion

The success criterion is a policy-admitted edit that moves held-out loops. The gate admitted all three edits and the receipts are written. Held-out loops did not move, so the criterion is not met.
