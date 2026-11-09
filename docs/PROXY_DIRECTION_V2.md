# Direction comparison v2

Study id: `proxy_direction_v2`.

Question: does a direction informed by supervised practice loss produce a better compensating edit than the existing difference between mean incorrect and mean correct logits?

The protocol sections below were frozen before the official trial. The measured result is a later section. The trial is not rerun to change the protocol.

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

The first command refuses a nonempty directory and refuses the archived v1 directory.

The first launch, preserved as `experiments/proxy_direction_v2_harness_failure`, exited 1 before any candidate slot. The proposer raised `KeyError: 'shapes'` because the public config omitted the matrix shapes. Confirmation was not opened and no bundle was written. The shapes are the v1 matrices, `W` included as `[4, 16]`. That omission is a harness defect, not a direction result. The protocol commit was not amended. The official candidate trial is `experiments/proxy_direction_v2`.

## Measured result

Protocol commit `f82334f8dd3ca962d0532d74211ab3860ceda469` predates the trial. Commit `7c66e8cda391de1084c86985c207ac0ad89d2782` added the omitted shapes and preserved the harness failure. The official runner then exited 0. `verify_direction` printed:

```
status=verified study_status=complete wall_seconds=0.7181493999669328 comparison=NEITHER_METHOD_CONFIRMED
```

The run used 0.7181493999669328 seconds of the 60-second ceiling. `exceeded` is false. Sixteen slots were applied and zero slots were rejected, plus the shared no-edit control. `value_exact_reconstruction` is true. On this Windows runtime all sixteen current resave flags are true. Python 3.12.10 and NumPy 2.5.4.

### Directions

`mean_logit_difference` has direction SHA-256 `c760c2cc907e2fdde44eed24faca8f698ac4bdc75d90c65f94175acc09c2df03`, the same identity as the v1 practice direction. Ranked columns are `[13, 4]`. The selected-support diagnostic is `0.06240743190157001`.

`loss_sensitive_projection` passed the eigenvalue gate. The largest eigenvalue is `0.052331545871127066`, the second is `-0.009201665831084175`, and the gap is `0.06153321170221124`. `tau` is `1e-12`. The Frobenius norm of `M` is `0.2396278921008862`. Direction SHA-256 is `5c18c0525562990638d9730acc265e52cd08fc67b91884798f535d3a281d4410`. Ranked columns are `[4, 10]`. The full-matrix derivative is `-0.052331545871127094`. The selected-support diagnostic is `0.02018349511452412`.

Both diagnostics have `diagnostic_affects_selection` false. The negative full-matrix derivative is the first-order change for an unrestricted projection. The positive selected-support values are the first-order change after the edit is restricted to the two ranked columns. Those diagnostics did not create another direction or another candidate.

### Practice

The damaged practice baseline is 118 correct of 128 targets.

`loss_sensitive_projection` applied every dose and qualified none:

| Dose | Corrections | Losses | Preservation flips | Candidate correct |
| --- | ---: | ---: | ---: | ---: |
| 0.005 | 0 | 0 | 0 | 118 |
| 0.01 | 0 | 0 | 0 | 118 |
| 0.02 | 0 | 1 | 1 | 117 |
| 0.04 | 0 | 1 | 1 | 117 |
| 0.08 | 0 | 2 | 1 | 116 |
| 0.12 | 0 | 4 | 3 | 114 |
| 0.16 | 0 | 4 | 3 | 114 |
| 0.20 | 0 | 5 | 3 | 113 |

Its finalist status is `NO_QUALIFIED_CANDIDATE`.

`mean_logit_difference` applied every dose. One slot qualified:

| Dose | Corrections | Losses | Preservation flips | Eligible | Candidate correct |
| --- | ---: | ---: | ---: | --- | ---: |
| 0.005 | 0 | 0 | 0 | false | 118 |
| 0.01 | 0 | 0 | 0 | false | 118 |
| 0.02 | 1 | 0 | 0 | true | 119 |
| 0.04 | 1 | 1 | 1 | false | 118 |
| 0.08 | 1 | 3 | 5 | false | 116 |
| 0.12 | 1 | 8 | 11 | false | 111 |
| 0.16 | 1 | 14 | 17 | false | 105 |
| 0.20 | 1 | 18 | 20 | false | 101 |

The finalist is `mean_logit_difference-02.wcbundle`, dose index 2, columns `[13, 4]`, strength `0.0960031127352528`, stored update Frobenius norm `0.14652793881663081`. Bundle SHA-256 `b103897223d276f9d1084a954fc83fe1d2002803c779699d148929afb9b52447`. Candidate content SHA-256 `7b19e3b74f796feaf972dbb68904593a8696464c898534bbc764fcd8731ef62f`.

### Confirmation

Seed `1112` drew the bank before either finalist was scored. Bank SHA-256 `05a62b0c7b813b152e4513574462de846dec5c85990af1ee7c79c9a255a7277d`. Lock SHA-256 `9ac583e37e2c3a76fddf0625844b76967108238d4596a5ca1614f8d0c7469c33`. `confirmation_opened` is true and `used_for_selection` is false.

The loss-sensitive confirmation record is `NO_QUALIFIED_CANDIDATE` because that method had no locked finalist.

On 256 confirmation targets the mean-logit finalist moved 246 correct answers to 245. Corrections are 0 and losses are 1. Preservation flips are 2, both correct-to-incorrect regressions. Preservation corrections and incorrect-to-incorrect changes are 0. `behavioral_repair` is false. The no-edit control stayed at 246 correct targets and 0 preservation flips.

The comparison is `NEITHER_METHOD_CONFIRMED`.

The intact teacher and the original fault were already disclosed. This is confirmation on fresh inputs for that known damaged model. The proposer subprocess receives the practice export, public config, damaged source, candidates directory, and deadline. The confirmation bank is not an argument. That process boundary is not a filesystem sandbox.

The strength-one fault deleted a component. Exact restoration of the intact weights is outside the claim. The comparison tests behavioral compensation under the frozen two-column budget.

### Completion checks

```
python -m pytest tests -q --tb=line
python -m weightcrack.evaluation.verify_proxy experiments/proxy_fault_v1
python -m weightcrack.evaluation.verify_direction experiments/proxy_direction_v2
```

`pytest` reported 34 passed and 1 warning in 3.09 seconds. The warning is the existing `PytestReturnNotNoneWarning` from `tests/test_phase3.py::test_suite_pins_and_scoring`. The v1 archive check printed `status=verified`, `study_status=complete`, and wall time `0.9394292000215501`. Its saved `behavioral_repair` value remains false. The pre-edit manifest of the four archived experiment directories still matches all 76 files, with 0 missing and 0 changed.

### Reproduce

`experiments/proxy_direction_v2` is nonempty, so the runner refuses that path. A fresh destination is:

```
python -m weightcrack.evaluation.direction_experiment experiments/proxy_direction_v2_replay
python -m weightcrack.evaluation.verify_direction experiments/proxy_direction_v2_replay
```

The official artifacts are verified with:

```
python -m weightcrack.evaluation.verify_direction experiments/proxy_direction_v2
```

### Separate claims

The verifier repairs, the shapes fix, the tests, and this replay are engineering completion. The scientific outcome on this synthetic model is `NEITHER_METHOD_CONFIRMED`. No real language-model checkpoint was loaded. `docs/REAL_MATRIX_STATUS.md` remains the blocker for that work.
