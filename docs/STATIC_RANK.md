# Static ranking measurement

Frozen seed list 0–59. Dense fault (6 columns, strength 6.0). Two PCA directions of failure logits. Columns ranked by |u^T W[:, j]|, top 2. Doses 0.05–2.0. Zero-loss gate.

Result: 54/60 = 0.90

Per-seed results in experiments/static_rank/results.json.

The writer-column score on the same directions reached 0.75 in the prior sweep. Static ranking by write magnitude is stronger on this proxy. The activation contrast is not required for localization when the fault is dense and the direction is recoverable from failure logits.

Command: see weightcrack/evaluation/synthetic_experiment.py for the wired path. The frozen measurement was run inline against the same functions.
