# Dose grid on real matrix shape

Date: 2026-11-10. Branch `dev/verified-projection-repair`.

## Setup

Extracted Qwen2.5-0.5B down-projection (896×4864). Twelve synthetic failure traces sharing a recoverable direction. Top-2 columns by write magnitude. Strengths 0.1, 0.25, 0.5, 1.0, 1.5, 2.0. Each candidate routed through `construct_candidate`.

## Result

Every strength was admitted by the host gate.

| strength | status   | touched | max_abs_delta |
|----------|----------|---------|---------------|
| 0.1      | admitted | 1792    | 0.010         |
| 0.25     | admitted | 1792    | 0.026         |
| 0.5      | admitted | 1792    | 0.051         |
| 1.0      | admitted | 1792    | 0.103         |
| 1.5      | admitted | 1792    | 0.154         |
| 2.0      | admitted | 1792    | 0.205         |

## Relation to the Qwen3.5-0.8B holdout

The prior behavioral measurement (commit 591e289) used strength 0.5 on eight real loops and did not move held-out behavior (4/6 both before and after). The gate admits the full dose range on this matrix shape. Changing strength alone does not require a new instrument. It does require re-running generation on the sealed holdout for each new finalist.

## Next

If a wider dose is to be tested on the real traces, each strength that is promoted to a finalist must lock its hash and open holdout once. The current local traces.npz still contains the older one-failure set. The eight-trace Qwen3.5 set lives in the commit that opened holdout.
