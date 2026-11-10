# PCA direction on failure logits

The top principal direction of the failure logits, paired with the writer-column score, raised zero-loss success from 0.57 to 0.64 on the dense fault (6 columns, strength 6.0, 100 seeds).

Mean-difference direction: 0.57
PCA direction on failures: 0.64

synthetic_experiment.py now uses the PCA direction for the heuristic path. The mean-difference direction remains available for comparison.

This is the strongest practice-only configuration measured on the dense proxy under the current gate.
