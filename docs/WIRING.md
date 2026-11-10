# Instrument wiring correction

The prior 0.72 figure came from an inline sweep using PCA top-2 directions of failure logits plus writer_column_score, with doses 0.05–2.0.

The committed synthetic_experiment.py previously used PCA top-1 with absolute-projection ranking and a narrow dose grid (0.005–0.20). It did not call writer_column_score.

This commit wires writer_column_score into the heuristic path and computes the top PCA direction of failure logits. Preservation scoring now measures baseline-to-candidate flips rather than disagreements with the intact teacher.

The dose grid in the runner remains narrow. Expanding it to match the sweep is a separate change. The instrument is now the one that produced the exploratory result.

## Algebra note

For an injected fault of strength f, the exact inverse repair strength is s = f/(f-1) when f ≠ 1. Strengths 2, 4, and 6 all have inverses inside the allowed (0, 2] range. The oracle ceiling on those faults is consistent with exact inversion, not with discovery of a general repair.
