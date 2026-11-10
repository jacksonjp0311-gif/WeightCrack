# Ceiling simulation: edit class on known coordinates

200 seeds. Oracle columns and direction known. Dose grid 0.05–2.0. Success = any dose raises practice accuracy above the damaged baseline.

| Edit | Columns | Success rate |
|------|---------|--------------|
| project_remove | 2 (true fault) | 1.00 |
| project_remove | 4 (true + 2 random) | 0.885 |
| project_remove | 8 (true + 6 random) | 0.755 |
| rank-1 write | 2 | 0.475 |
| rank-1 write | 4 | 0.495 |

## Reading

On this proxy, projection removal on the true fault columns succeeds when the dose grid is wide enough. Adding random columns or switching to a rank-1 write lowers the rate. The earlier ~18% ceiling was under tighter dose and eligibility constraints, not an intrinsic limit of the formula.

## Implication

The edit class is not the bottleneck on this proxy when coordinates are known. Localization remains the open problem: practice-only instruments have not selected the true columns under the eligibility rules. Expanding the allow-list to rank-1 writes does not raise the ceiling here.

Command: `python -m weightcrack.evaluation.ceiling_sim experiments/ceiling`
