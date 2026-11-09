# Shift: from localizer to constitution

## Measured ceiling (Monte Carlo, ~2000 seeds)

| Method | Success rate | Notes |
|--------|--------------|-------|
| Oracle (known fault columns + direction) | ~18% | Projection removal on 2 columns, doses 0.005–0.20 |
| Mean-difference heuristic | ~0.1% | Column ranking on practice residual |
| Random column selection | ~0.2% | Chance baseline |
| Column overlap with true fault | ~25% | Both methods at chance |
| Direction alignment (mean-diff vs true) | 0.43 | Weak signal recovery |

Oracle success of 18% means the edit class itself fails 82% of the time even when handed the correct coordinates. Localization instruments cannot overcome a formula that does not reverse the fault under the current dose and eligibility rules.

## Prior instruments (all practice-only, locked seed)

- Mean-difference ranking: NO_QUALIFIED_CANDIDATE
- Practice direction search (mean-diff + PCA of failures): NO_QUALIFIED_CANDIDATE
- Residual column attribution: NO_QUALIFIED_CANDIDATE
- Remote v2/v3 direction comparisons: NEITHER_METHOD_CONFIRMED

Four automatic methods, zero qualified candidates on the same seed. Empirical base rate is zero.

## Architectural shift

Stop refining localization instruments on the synthetic proxy. The signal (6 failures / 128) is too thin, and the edit class has a low ceiling even with perfect knowledge.

**WeightCrack becomes the gate.** External controllers (models that can read weights, as in recent weight-space programming work) propose sparse patches. The host enforces:

- Allow-list of tensors and operations
- Entry count, fraction, and absolute delta after rounding
- New file + receipt for every candidate
- Practice-only scoring for selection
- Holdout opened once, after finalist hash is locked

The contribution is the constitution: whatever proposes cannot write silently, cannot expand beyond policy, and cannot hide the change.

## Next measurement

A controller-facing proposal path that accepts an external patch proposal, runs it through the existing gate, applies under policy, and records the receipt. Success is defined as any sparse edit that moves held-out behavior under the gate, not as recovery of injected coordinates.

The 0/20 result from the prior WeightBridge harness was a verdict on proposal quality under a broken apply path. This tests proposal quality under a correct one.
