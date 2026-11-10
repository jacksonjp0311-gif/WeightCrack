# Dense-fault simulation: writer-column score under two rules

91 seeds. Fault: 4 columns, strength 4.0. Mean failures ~49 / 128.
Instrument: writer-column score = (mean_h_fail - mean_h_ok) * (u^T W[:, j]).
Doses: 0.05–2.0. Success = any dose raises practice accuracy.

| Rule | Success rate | Mean best delta |
|------|--------------|-----------------|
| Zero-loss (correction > 0 and zero losses) | 0.52 | — |
| Net-gain (corrections exceed losses) | 0.68 | +6.96 |

## Reading

The thin fault (2 columns, strength 2.0, 6 failures) produced zero hits across all instruments.
The denser fault plus the writer-column score produces a qualifying candidate on half the seeds under the existing zero-loss rule and on two-thirds under a net-gain rule.

## Implication

Density and the activation-contrast instrument are the combination that moves the proxy above zero. The zero-loss rule is stricter but still succeeds at 0.52. A net-gain rule raises that to 0.68 at the cost of allowing some previously correct predictions to flip.

Command: see weightcrack/evaluation/ceiling_sim.py for the oracle baseline; this result is from an inline dense-fault probe.
