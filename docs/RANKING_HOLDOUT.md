# Ranking holdout

Date: 2026-11-11. Branch `dev/verified-projection-repair`.

Strengths 0.5 and 1.0 were admitted. Each sealed holdout was opened once. No label flipped. The success criterion is not met.

## Ranking

Eight failure traces from the Qwen3.5-0.8B layer-15 measurement, shape `(8, 1024)`. Columns were ranked by the sum of absolute write magnitudes onto the top two PCA components of those traces. The selected columns stay 1309, 2633, 1775, and 1022.

The edit direction is the top PCA component of the same eight centered traces, normalized by its norm plus `1e-12`. That is the direction whose strength-0.5 and strength-1.0 deltas match `experiments/real_probe_rank/ranking.json`.

## Gate

Each strength was built with `construct_candidate` and the probe policy: `allow_tensors` `weight`, `max_entries` 10000, `max_fraction` 0.05, `max_abs_delta` 1.0. No strength was rejected. Each candidate hash was written to `experiments/real_probe_rank/ranking_holdout_started.json` before any holdout prompt was tokenized. `apply_patch` then published a new bundle. Touched coordinates are 4096, which is 1/896 of the matrix.

| strength | candidate content SHA-256 | max abs stored delta | bundle |
| --- | --- | --- | --- |
| 0.5 | `ef2601d185d53009b8f74979c17b49c4685909af603e15361e6cf54cafad1597` | 0.002945561893284321 | `experiments/real_probe_rank/strength-0.5.wcbundle` |
| 1.0 | `cf7a3f2a895bd0c7b94c1564cc931a203faa3601dba608ef29c7020262b6b44a` | 0.005891123786568642 | `experiments/real_probe_rank/strength-1.0.wcbundle` |

## Holdout

Opened once per finalist. Six sealed prompts. Thinking enabled. Greedy decoding, 384 new tokens. The 4-word rule is unchanged. Only the edited matrix was generated. Baseline labels are copied from `experiments/real_probe/holdout.json` and were not regenerated. The recorded baseline is 4 of 6.

| id | baseline | strength 0.5 | strength 1.0 |
| --- | --- | --- | --- |
| h01 | true | true | true |
| h02 | false | false | false |
| h03 | true | true | true |
| h04 | true | true | true |
| h05 | false | false | false |
| h06 | true | true | true |

Each edited pass looped on 4 of 6 prompts. Every prompt kept its baseline label. No label flipped at either strength.

Details: `experiments/real_probe_rank/ranking_holdout.json`. Runner: `experiments/real_probe_rank/ranking_holdout.py`.

## Criterion

The success criterion is a policy-admitted edit that moves held-out loops. The gate admitted both edits and the receipts are written. No label flipped, so the criterion is not met.
