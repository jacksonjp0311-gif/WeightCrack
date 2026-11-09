# Support comparison v3

Study id: `proxy_support_v3`.

Question: does selecting the two columns with the largest positive predicted loss reduction produce a better compensating edit than selecting the two columns with the largest projection magnitude, when both use the same loss-sensitive direction?

The sections through Commands are the protocol. They were written before the official candidate trial. The measured result is a later section. The trial is not rerun to change the protocol.

## What v2 showed

On the v2 damaged model, the loss-sensitive direction had full-matrix derivative `-0.052331545871127094`. After the edit was restricted to columns `[4, 10]`, the selected-support derivative was `+0.02018349511452412`. The unrestricted projection was a predicted descent. The two-column edit chosen by absolute projection was a predicted ascent. v2 did not confirm either method. This study asks whether the column choice, not a new direction, is the gap.

## Shared direction

Both methods use the v2 loss-sensitive direction and no other direction.

```
H = tanh(X @ A.T + b)
Z = H @ W.T + c
P = stable_softmax(Z)
Y = one_hot(practice_target_labels)
G = (P - Y).T @ H / N
M = (G @ W.T + W @ G.T) / 2
```

`N` is 128. Eigenpairs come from `numpy.linalg.eigh`. The tolerance and the sign rule are the v2 rules. A missing unique leading eigenvalue is `NO_UNIQUE_POSITIVE_PROJECTION_DIRECTION` for both methods. There is no fallback direction.

For the permitted edit

```
W'[:, j] = W[:, j] - s * u * (u.T W[:, j])
```

the first-order practice-loss derivative on a column set `C` is

```
dL/ds at s=0 = -sum_{j in C} (u.T G[:, j]) * (u.T W[:, j])
```

Define `a_j = (u.T G[:, j]) * (u.T W[:, j])`. Positive `a_j` is a descent contribution at strength zero. The summed derivative is recorded after the columns are chosen. It does not choose a direction, a third column, or an extra dose.

## Methods

1. `magnitude_projection`: rank `r_j = |u.T W[:, j]|`. Keep the two largest scores. Equal scores keep the smaller column index.
2. `signed_loss_contribution`: keep columns with `a_j > tau`, where `tau = 1e-12 * max(1, max |a_j|)`. Among those, keep the two largest `a_j`. Equal values keep the smaller column index. Fewer than two columns above `tau` is `INSUFFICIENT_DESCENT_SUPPORT`. The support is not expanded, and another ranking is not substituted.

`rank_columns` remains the v2 absolute-projection helper. Its `argsort` order is not this study's tie-break. The magnitude method does not call it.

Both methods use the same new damaged model, the same practice arrays, the same direction, the same projection, two columns, the v1 dose grid, and the v1 policy. The only experimental variable is the column rule.

Doses are `0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20` times the damaged `W` Frobenius norm. Match tolerance is `1e-3`. The policy allows eight coordinates, fraction `0.125`, and stored absolute delta `2.0`. Strength stays in `(0, 2]`. Storage is the source dtype after one cast from float64. Sixteen slots is the maximum, plus the shared no-edit control. An infeasible dose consumes its slot.

Practice qualification requires positive target corrections, zero losses of previously correct targets, and zero preservation flips. Ties prefer more corrections, then a smaller stored update norm, then an earlier dose. Otherwise the finalist status is `NO_QUALIFIED_CANDIDATE`.

## One new case

The classifier is `f(x) = argmax(W tanh(Ax + b) + c)` with shapes 8, 16, and 4. The intact model is one `standard_normal` draw cast to float32. The fault is one `project_remove` of strength 1.0 on two columns drawn without replacement, using the same generator order as v1. v1 seeds `1108` through `1111` and the v2 confirmation seed `1112` are not reused.

Practice is 128 target rows and then 128 preservation rows from the data seed. Confirmation is a separate generator: 256 target rows and then 256 preservation rows, cast once to float32. Labels are the intact teacher's argmax. v1 and v2 confirmation banks are not read.

The seed integers, the realized fault columns, the intact teacher, and the confirmation labels live in `host_only/`. `prepare` writes those files and `public_commitment.json`, which stores their SHA-256 values. That commitment is the protocol record. `prepare` does not build candidates and does not score them. `run` refuses to redraw the committed files.

The protocol commit contains the commitment and the proposer inputs. It does not contain `host_only/`. The seed integers are therefore not in that commit. They are disclosed with `host_only/` after the lock and the confirmation score. Anyone with the disclosed seeds can redraw the teacher. The proposer arguments still omit `host_only/`. Process separation is not a filesystem sandbox.

## Confirmation

The lock is written before either finalist is scored on the confirmation bank. Each locked finalist is scored once. Confirmation requires positive target correction, zero losses of previously correct targets, and zero preservation flips. Preservation changes are also counted as corrections, regressions, or changes between incorrect answers.

If both methods confirm, more target corrections decides. An equal count is `NO_CONFIRMATION_WINNER`. One confirmed method is `SIGNED_SUPPORT_CONFIRMED` or `MAGNITUDE_SUPPORT_CONFIRMED`. If neither confirms, the comparison is `NEITHER_METHOD_CONFIRMED`. This reading does not create candidates.

For every applied candidate the record includes the selected columns, the direction hash, the signed contributions, the selected-support derivative, the first-order prediction at the stored strength, the measured practice cross-entropy change, the practice corrections and losses, the preservation flips, and the stored-update accounting. Sign agreement is a diagnostic. It does not open another search.

## Commands

```
python -m weightcrack.evaluation.support_experiment prepare experiments/proxy_support_v3 --seeds-file SEEDS.json
python -m weightcrack.evaluation.support_experiment run experiments/proxy_support_v3
python -m weightcrack.evaluation.verify_support experiments/proxy_support_v3
```

`SEEDS.json` stays outside the repository until `host_only/` is disclosed. `prepare` refuses a nonempty directory and refuses the archived experiment names. `run` refuses an existing selection or report. The official `run` ceiling is 60 seconds and does not include `prepare` or unit tests.

No real language-model checkpoint is loaded. `docs/REAL_MATRIX_STATUS.md` still records that limit.

## Measured result

The protocol commit is `a03425c6f2349136bbbf0e38dac5dc185fbac28a`. It contains the proposer inputs and `public_commitment.json`. It does not contain `host_only/`. The official `run` then wrote the lock, opened the confirmation bank, and wrote `experiment_report.json`. This section was written after that run. The trial was not repeated.

Disclosed host configuration, from `host_only/host_config.json`:

- Model seed `1113`, data seed `1114`, fault seed `1115`, confirmation seed `1116`.
- Fault columns `[3, 0]`, fault strength `1.0`, eight changed coordinates.
- Python `3.12.10`, NumPy `2.5.4`.

`verify_support` reported `status=verified`, `study_status=complete`, wall `0.7759591999929398` seconds, comparison `NEITHER_METHOD_CONFIRMED`. The run itself returned status code 0 with that same comparison. The wall clock is under the 60-second ceiling. Sixteen slots were applied and zero were rejected. Every applied bundle reconstructed its stored values. The resave of each applied candidate matched the saved bytes on this machine.

The loss-sensitive direction passed the existing uniqueness gate. Its content hash is `82697467246233f47dceb3e5e361afda17999fda2b299eb51290ecfbf8411688`. The largest eigenvalue is `0.15391471415070682`, the second is `0.06598281873605817`, and the gap is `0.08793189541464864`. The tolerance is `1e-12`. The full-matrix derivative is `-0.15391471415070684`.

Both rankings selected columns `[0, 6]`. Magnitude scores were `1.6485727103898251` and `1.5114685068783975`. Signed contributions on those columns were `0.21284183531471276` and `0.15166652915083903`. Ten columns had a strictly positive contribution. The signed rule kept two of them. The selected-support derivative on `[0, 6]` is `-0.36450836446555185`.

The eight doses produced the same stored matrix under both methods. Each pair of bundles has the same candidate content hash and the same candidate file hash. The experimental variable therefore did not create two different edits on this draw. The locked finalist for each method is dose `0.005`, bundle index `00`, strength `0.017956306007532792`, stored update norm `0.040160910798450605`, eight touched coordinates, fraction `0.125`, and maximum absolute stored delta `0.02596902847290039`. The shared candidate content hash is `966d7885a1f35f15bfd538385c2c3f036972a31e421eb177ce00891b6a90f140`.

Practice, 128 target rows: the damaged model had 102 correct. The dose `0.005` edit had 103 correct, one correction, zero losses, and zero preservation flips, so it qualified. The seven larger doses also reduced practice cross-entropy and increased target corrections, and each of them changed at least two preservation predictions, so they did not qualify. From dose `0.08` upward they also lost previously correct target predictions.

Confirmation, 256 fresh target rows: the no-edit control stayed at 211 correct, with zero corrections, zero losses, and zero preservation flips. Each locked finalist moved that count from 211 to 212: one target correction, zero target losses, and two preservation flips. Both flips were changes from an incorrect preservation answer to the teacher label. The preregistered rule still requires zero preservation flips, so `behavioral_repair` is false for both methods. The comparison is `NEITHER_METHOD_CONFIRMED`.

On all 16 applied slots the sign of the first-order prediction matches the sign of the measured practice-target cross-entropy change. Both are negative. At dose `0.005` the prediction is `-0.006545223734648741` and the measured change is `-0.006463649848455821`. At dose `0.20` the prediction is `-0.2618083606177329` and the measured change is `-0.1200677152725444`. The local derivative's sign survived these sparse, finite, stored edits. The linear magnitude is a poorer description at the larger strengths. That diagnostic was not used to choose another column or another dose.

## Interpretation

On this one case, selecting support by positive predicted loss reduction produced the same two columns as selecting support by projection magnitude. The shared edit lowered practice cross-entropy and met the practice gate at the smallest dose. It did not meet the confirmation gate, because two preservation predictions changed. The signed rule therefore shows no advantage over the magnitude rule here. One synthetic draw does not establish a general editing method, and this repository has not compared the working label "support–direction compatibility" with the literature.

## Checks

- `python -m pytest tests/test_support_v3.py -q --tb=short`: 8 passed in 2.05 seconds.
- `python -m pytest -q --tb=line`: 42 passed, 1 warning, in 4.92 seconds. The warning is the existing `tests/test_phase3.py::test_suite_pins_and_scoring` return value.
- `prepare` returned status code 0 before the protocol commit.
- `run` returned status code 0. Wall time `0.7759591999929398` seconds.
- `python -m weightcrack.evaluation.verify_support experiments/proxy_support_v3` returned `status=verified`.
- The pre-edit manifest `docs/support_v3_archive_integrity.json` still matches the six archived experiment directories: 156 files, 0 missing, 0 changed. That comparison was repeated after the official run.
