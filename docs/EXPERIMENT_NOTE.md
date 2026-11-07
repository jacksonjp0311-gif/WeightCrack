## Stronger-fault synthetic (seed 0, fault_strength=2.0)

Pre-registered. Same populations and selection rules as the prior run.

- Practice failures: 6 / 128
- Heuristic columns: [0, 14]
- Random columns: [3, 8]
- Both methods: NO_QUALIFIED_CANDIDATE

Oracle check (not used for selection): applying the known fault direction and the known fault columns [3, 7] at strength 1.0 recovered 4 of the 6 practice failures with zero losses of previously correct predictions. The edit can reverse the injected fault when the coordinates are known. The mean-difference ranking did not select those coordinates.

Command:
```
python -c "
from pathlib import Path
from weightcrack.evaluation.synthetic_experiment import run_experiment
run_experiment(Path('experiments/synthetic_strong'), seed=0, fault_strength=2.0)
"
```
