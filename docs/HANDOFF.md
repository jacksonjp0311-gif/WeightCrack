# WeightCrack Handoff

Date: 2026-11-09
Branch: dev/verified-projection-repair
Head: b72a44f

## Goal

A sparse edit that improves a model on a real task, under the gate, where an agent proposes and the host enforces. The agent does not write silently. The host records the receipt and opens holdout once.

## What is solved

- Apply path. Coordinate accounting, dtype preservation, exclusive artifacts, receipts. See weightcrack/patches/huggingface.py and weightcrack/policy.py.
- Edit formula. Projection removal on known columns reaches 100% success on the synthetic when the dose grid is wide. See docs/CEILING.md.
- Localization on dense synthetic faults. Top-two PCA directions of failure logits plus writer-column score reaches 0.72 zero-loss success on 80 seeds (6 fault columns, strength 6.0). See docs/OPTIMIZED.md and docs/PCA_DIRECTION.md.

## What is not solved

- Localization on thin faults. Every instrument returned zero hits on the original 2-column, 6-failure fault.
- Transfer to a real matrix. The probe harness exists (weightcrack/evaluation/real_matrix_probe.py). No matrix has been loaded. Download failed with transport error. See docs/REAL_MATRIX_STATUS.md.
- Agent proposal measured against the instrument on a dense real task. The controller stub exists (weightcrack/agents/controller_proposal.py). It has not been run against a real matrix.

## Measured trajectory on the dense fault

| Configuration | Zero-loss rate |
|---------------|----------------|
| Thin fault (2 cols, str 2.0, 6 failures) | 0.00 |
| Dense fault, mean-diff direction, writer score | 0.57 |
| Dense fault, PCA top-1, writer score | 0.64 |
| Dense fault, PCA top-2 directions, writer score | 0.72 |
| Combined writer + static | 0.72 |
| Ablation ranking | 0.70 |
| Residual direction | 0.58 |

Density was required. Direction estimate was the next lever. Further linear combinations of the same signals did not clear 0.72.

## Public results that inform the next step

- MIT CSAIL "Sparks of Weight-Space Programming" (2 Nov 2026): GPT-6 Astra edited 0.016% of Qwen3.5-0.8B (layers 15 and 19 down_proj, 64 columns each) and raised MATH-500 from 7.9% to 48.1%. Direction from looping vs productive residual contrast. Column ranking by activation contrast times static write. https://agentspulse.github.io/tutorials/weight-space-programming/
- Shanghai AI Lab arxiv:2611.09382 (6 Nov 2026): 16 trajectories on Qwen3.5-0.8B and 9B. 1,023-parameter edit raised MATH-500 by 29.45 points. Cross-task generalization uneven. Six of 16 trajectories reverted when validation failed. https://arxiv.org/abs/2611.09382
- nathanrs/weight-space-lp (9 Nov 2026): Linear program over columns matches the Astra edit with 8× fewer weights, given the direction from looping vs productive traces. https://nathan.rs/posts/weight-space-lp/
- Concept Country replication (8 Nov 2026): Norm-matched edit on Qwen3.8-27B improved GPQA Diamond by 4.3 points and LiveCodeBench v6 by 13.0 points while cutting thinking tokens. Random control did not. https://concept.country/research/weight-space-programming

The writer-column score used here matches the ranking the replications report as necessary. The Qwen repetition case is the dense real task where that ranking produced measured gains.

## What the next agent must do

1. Load one MLP matrix from a public checkpoint where a repetition or loop direction has been localized. The probe harness is ready. The prior download failed with a transport error.
2. Run the current instrument (PCA direction of failure logits + writer-column score) against a frozen practice/holdout split of a loop-or-not probe. Record whether a candidate qualifies under the zero-loss gate.
3. If it qualifies, lock the finalist hash and open holdout once. Record the held-out numbers and the receipt.
4. If it does not, the claim stays at "the instrument works on dense synthetic faults; transfer is unmeasured."

Do not expand the synthetic further under the current gate. The 0.72 is the measured ceiling of the instruments that have been tried. The next measurement that changes the claim is the real matrix.

## Files that matter

- weightcrack/patches/huggingface.py — apply path
- weightcrack/policy.py — gate
- weightcrack/agents/localization.py — writer_column_score and other instruments
- weightcrack/agents/controller_proposal.py — external proposal stub
- weightcrack/evaluation/synthetic_experiment.py — dense-fault defaults, PCA direction
- weightcrack/evaluation/real_matrix_probe.py — ready for a local matrix
- docs/CEILING.md, docs/DENSE_FAULT.md, docs/OPTIMIZED.md, docs/PCA_DIRECTION.md, docs/SHIFT.md, docs/REAL_MATRIX_STATUS.md
