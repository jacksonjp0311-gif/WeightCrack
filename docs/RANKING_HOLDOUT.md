# Ranking holdout

Date: 2026-11-11. Branch `dev/verified-projection-repair`.

## Ranking change

Eight failure traces from the Qwen3.5-0.8B layer-15 measurement. Direction is the top two PCA components of those traces. Columns ranked by the sum of absolute write magnitudes onto both directions. Top four selected: 1309, 2633, 1775, 1022.

## Gate

Strengths 0.5 and 1.0 were both admitted through `construct_candidate`.

| strength | status   | touched | max_abs_delta |
|----------|----------|---------|---------------|
| 0.5      | admitted | 4096    | 0.0029        |
| 1.0      | admitted | 4096    | 0.0059        |

## Holdout

Not opened. Generation on the sealed prompts is required to score labels. The prior two-column holdout (columns 2056 and 1309) did not move behavior at any dose. This ranking includes column 1309 and adds three others. A new finalist for either strength would lock its hash and open the same six prompts once.

## Status

The ranking change produces an admissible candidate. The behavioral measurement is open until holdout is run. The success criterion remains unmet until a label flips.
