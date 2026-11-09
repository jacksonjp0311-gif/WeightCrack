# Direction comparison v2

Study id: `proxy_direction_v2`.

Question: does a direction informed by supervised practice loss produce a better compensating edit than the existing difference between mean incorrect and mean correct logits?

This file is the frozen protocol. The official candidate trials are a later commit. They are not rerun to change this protocol.

## Reused material

The damaged model and the practice arrays are the byte-identical v1 files:

| Identity | SHA-256 |
| --- | --- |
| `practice_export.npz` | `43564c2ce9e4a973a517d4eb887bd75c377d72f0785ab4e2692e14cc659d4535` |
| `damaged_source.npz` | `9bb3173af038aa8d58070fd28c3c5c6811e5afe0dbc2ef309dd44c3aebad2728` |
| Damaged `W` content | `0f264954eccef5288d489cbfba305a80a0f8635d476e30b0739d35e338c59e5e` |
| Intact `W` content | `c8da464c0dc7412cc928e10fd61b00f544a7897a65d669b226c6997fcf2bc5a7` |

v1 constants are not replaced. This study has its own config and no random-control arm. v1 compared column selection under one shared direction. v2 compares two direction constructions.

## Methods

1. `mean_logit_difference`: the existing normalized mean-incorrect-minus-mean-correct practice-logit direction.
2. `loss_sensitive_projection`: the direction below.

Both methods rank columns by absolute projection and keep the top two. Policy, precision, norm doses, matching tolerance, and the practice selection rule are the v1 values. Doses are `0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20` times the damaged `W` Frobenius norm. Match tolerance is `1e-3`. The policy allows eight coordinates, fraction `0.125`, and stored absolute delta `2.0`. Strength stays in `(0, 2]`. Storage is the source dtype after one cast from float64.

Each method has one direction, one column pair, and at most eight candidate slots. Sixteen slots is the maximum, plus the shared no-edit control. An infeasible dose consumes its slot. Columns are not resampled and bounds are not enlarged.

Practice qualification requires positive target corrections, zero losses of previously correct targets, and zero preservation flips. Ties prefer more corrections, then a smaller stored update norm, then an earlier dose. Otherwise the finalist status is `NO_QUALIFIED_CANDIDATE`.

## Loss-sensitive direction

Use the 128 practice target inputs, their labels, and the damaged stored parameters, in float64. Examples are rows and `N = 128`.

```
H = tanh(X @ A.T + b)
Z = H @ W.T + c
P = stable_softmax(Z)
Y = one_hot(practice_target_labels)
G = (P - Y).T @ H / N
M = (G @ W.T + W @ G.T) / 2
```

`G` is the gradient of mean cross-entropy with respect to `W`. Eigenpairs come from `numpy.linalg.eigh(M)`. The tolerance is `tau = 1e-12 * max(1, frobenius_norm(M))`. All intermediates must be finite, the largest eigenvalue must exceed `tau`, and the gap to the second-largest eigenvalue must exceed `tau`. Otherwise the status is `NO_UNIQUE_POSITIVE_PROJECTION_DIRECTION`. There is no fallback direction. That status is not evidence that no useful sparse direction exists.

The saved direction is the unit eigenvector for the largest eigenvalue. Its sign makes the largest-magnitude coordinate positive. Equal magnitudes use the lowest index.

For the full-matrix step `ΔW = -s * u u.T @ W`, the first-order loss derivative at `s = 0` is `-u.T @ M @ u`. The candidate still edits only `top_two(abs(u.T @ W))`. The full-matrix derivative does not guarantee descent after that restriction, improvement at a finite dose, or preservation on unseen inputs.

The selected-support diagnostic is

```
-sum((u.T @ G[:, c]) * (u.T @ W[:, c]) for c in selected_columns)
```

It is recorded and does not choose a direction, a column pair, or an extra candidate.

## Confirmation

Seed `1112` draws 256 target rows and then 256 preservation rows with `numpy.random.default_rng`, `standard_normal`, and one cast to float32. Labels are the intact teacher's argmax. v1's held-out set is not used to select or qualify candidates.

The bank is frozen, and both practice finalists or failure states are locked, before either method is scored on it. Each locked finalist is scored once. Confirmation requires positive target correction, zero losses of previously correct targets, and zero preservation flips. Preservation changes are also counted as corrections, regressions, or changes between incorrect answers.

If both methods confirm, more target corrections decides. An equal count is `NO_CONFIRMATION_WINNER`. If neither confirms, the comparison is `NEITHER_METHOD_CONFIRMED`. This reading does not create candidates.

No seed rerolls, extra attempts, new formulas, changed thresholds, or dose adjustments are part of the protocol after confirmation is opened.

## Process boundary

The proposer is a subprocess. Its arguments are the practice export, the public config, the damaged source, the candidates directory, and a deadline. The confirmation bank is not an argument. Process separation is not a filesystem sandbox. A modified proposer could open sibling files.

The original teacher and fault are already disclosed. A confirmation result is a result on fresh inputs for this known damaged model. It is not a fresh blinded fault-localization replication.

The strength-one fault deleted a component. The permitted projection is nonexpansive in exact arithmetic, so exact restoration of the intact weights is not the claim. The experiment tests behavioral compensation.

The official run's wall-time ceiling is 60 seconds, including candidate construction, selection, confirmation, and the in-run replay. Implementation, dependency setup, and ordinary unit tests are outside that ceiling. A harness failure is preserved and reported. The budget is not silently reset.

No real language-model checkpoint is loaded. `docs/REAL_MATRIX_STATUS.md` still records that limit.

## Commands

```
python -m weightcrack.evaluation.direction_experiment experiments/proxy_direction_v2
python -m weightcrack.evaluation.verify_direction experiments/proxy_direction_v2
```

The first command refuses a nonempty directory and refuses the archived v1 directory. The official result is not in the protocol commit.
