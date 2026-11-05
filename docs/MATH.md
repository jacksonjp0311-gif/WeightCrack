# Refined localization and edit math

## Core observation

On the Qwen3.5-0.8B pilots, the edits that cleared the locality gate were not dense fine-tunes. They were removals of a small number of directions written into specific MLP output columns (notably layers 15 and 19 in the published-style reconstruction). Changing ~0.016 % of weights by subtracting a single direction from 64 columns produced large shifts on math and instruction-following probes while leaving sentinels intact.

This matches the broader locate-then-edit literature (ROME/MEMIT/EMMET) and the recent weight-space programming results: the useful intervention is often a low-rank update or an explicit projection removal, not a coordinate-wise scatter of values.

## Preservation-memorization objective

For a weight matrix \( W \) and a set of key vectors we wish to preserve (\( K_0 \)) and a set we wish to rewrite (\( K_e \)):

\[
\min_{\hat{W}} \| \hat{W} K_0 - W K_0 \|_F^2 + \lambda \| \hat{W} K_e - V_e \|_F^2
\]

ROME solves the equality-constrained form for a single edit via a closed-form rank-one update. MEMIT relaxes to least-squares and distributes the update across layers. EMMET restores the equality constraint for the batched case.

For the current WeightCrack target (sector repair rather than fact insertion), the practical reduction is simpler: identify a direction \( u \) that is active on failing practice examples and weakly active on sentinels, then subtract a controlled multiple of its outer product from a small set of columns.

## Direction-removal formula (first-class)

Given a matrix \( W \in \mathbb{R}^{d \times n} \), a unit vector \( u \in \mathbb{R}^{d} \), a set of columns \( C \), and a strength \( s \in (0, 2] \):

\[
W'_{:, c} = W_{:, c} - s \, u \, (u^\top W_{:, c}) \quad \forall c \in C
\]

This is exactly the operation already supported in the adapted patch path. The host can execute it under the existing policy bounds (max entries, max fraction, max abs delta after rounding).

## Localization instruments that stay practice-only

1. **Activation difference**  
   Mean residual or MLP output on failing practice examples minus mean on sentinel or correct examples. Top singular vector of the difference is a candidate \( u \).

2. **Finite-difference sensitivity**  
   Host-executed: add a small probe along a candidate direction, measure practice score change, return the delta. Agent never sees the raw weights.

3. **Column ranking**  
   For a candidate \( u \), rank columns by \( |u^\top W_{:, c}| \). The top-k columns are the natural target set for the removal.

These three instruments turn the agent into a selector over host-computed candidates rather than a free-form weight writer. That is the gap the current pilots still show (direct transfer 0/20, constrained pilot 1/3).

## What remains under the existing invariants

- Practice suite only for all localization and trial steps.
- Holdout opened once, after candidate selection, by the host.
- Policy gate still enforces allow-list, entry count, fraction, and delta.
- Every candidate is a new file + receipt.

The math that has already produced measured gains fits inside those constraints. The next concrete step is to expose the activation-difference and column-ranking reports as first-class, bounded tools the agent can call.
