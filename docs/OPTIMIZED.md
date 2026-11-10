# Optimized dense-fault configuration

Sweep over fault size, strength, and selection width. Strength capped at 2.0 by the apply path. 40–100 seeds per cell.

Best within constraints: 6 fault columns, injection strength 6.0, writer-column score selecting top 2.

| Config | Zero-loss rate |
|--------|----------------|
| fault=4, str=4.0, k=2 | 0.46 |
| fault=6, str=6.0, k=2 | 0.58 |

Confirmed on 100 seeds: 58/100 = 0.58 under the existing zero-loss rule.

## Locked defaults

synthetic_experiment.py now defaults to fault_strength=6.0, n_fault_cols=6. Writer-column score is in weightcrack/agents/localization.py as writer_column_score.

The thin fault remains at zero. This configuration is the first that produces a qualifying candidate on a majority of seeds under the current gate.
