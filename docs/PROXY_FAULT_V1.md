# Proxy fault study v1

One intact teacher, one damaged copy, one practice-derived direction, two column-selection methods. No training and no second task.

## Protocol

Model: `f(x) = argmax(W tanh(Ax + b) + c)` with input 8, hidden 16, classes 4. Parameters are drawn with `numpy.random.Generator.standard_normal` and cast once to float32. Labels are the intact teacher's argmax predictions. Logits used for scoring and for the direction are computed in float64 from those stored parameters.

Frozen constants in `weightcrack/evaluation/proxy_experiment.py`:

| Item | Value |
| --- | --- |
| Model / data / fault / control seeds | 1108 / 1109 / 1110 / 1111 |
| Fault | `project_remove` at strength 1 on two fault-seed columns |
| Practice target / preservation | 128 / 128 |
| Held-out target / preservation | 256 / 256 |
| Doses | 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20 times the damaged `W` Frobenius norm |
| Match tolerance | 1e-3 relative stored update norm |
| Policy | 8 entries, fraction 0.125, absolute stored delta 2 |
| Wall ceiling | 60 seconds |

The fraction denominator is 64, the number of entries in `W`. Two columns touch 8 coordinates, which is exactly the entry cap and the fraction cap.

The direction site is `post_linear_logits` on the damaged model: normalized mean logit on incorrect practice targets minus the mean on correct ones. Its length is 4, matching `W.shape[0]`. `top_k` does not change it. Both methods use this direction. The heuristic takes the two largest absolute projections. The random arm draws two columns without replacement from `default_rng(1111)`. Strengths are calibrated from that geometry before scores are computed. An invalid dose would have consumed its slot; none did.

Practice qualification requires at least one target correction, no loss of a previously correct target, and zero preservation flips. Ties prefer more corrections, then a smaller stored update norm, then an earlier dose. The no-edit control is the damaged matrix.

The proposer subprocess receives `proposal_inputs/practice_export.npz`, `public_config.json`, `damaged_source.npz`, the candidates directory, and a deadline. It does not receive `host_only/`. The host writes `proposal_lock.json` and an event line before it opens the secret. Process separation is not a filesystem sandbox.

## Result

Runtime: Python 3.12.10, NumPy 2.5.4. Wall time 0.9394292000215501 seconds. Status `complete`. Verify replay exited 0.

The fault changed columns `[10, 7]` (8 stored coordinates). The practice direction was nonzero. Practice targets: 10 wrong, 118 correct.

| Method | Columns | Practice finalist | Held-out corrections | Held-out losses | Preservation flips | Repair |
| --- | --- | --- | --- | --- | --- | --- |
| No edit | none | damaged model, 118/128 correct | 0 | 0 | 0 | false |
| Heuristic | 13, 4 | dose index 2, strength 0.0960031127352528 | 2 | 1 | 3 | false |
| Random | 12, 13 | `NO_QUALIFIED_CANDIDATE` | not scored | not scored | not scored | false |

Held-out baseline accuracy of the damaged model is 238/256. The heuristic finalist is 239/256: the 2 corrections and 1 loss net one additional correct target, and 3 preservation predictions changed. The repair rule requires a gain, no loss, and no preservation flip, so the flag stays false.

All 16 dose slots applied. Each touched 8/64 coordinates. The heuristic finalist's stored update norm is 0.14652793881663081 against a target of 0.1465279351220506 (relative error 2.5214169721366263e-08). Its max stored absolute delta is 0.08100330829620361. `value_exact_reconstruction` is true for all 16 bundles. This run's resave byte-identity flags are also true; that flag is not the reconstruction claim.

The selected columns are not the injected columns. A practice pass of one correction did not survive the held-out rule. This supports only a statement about this fault, this direction, and this dose grid.

## Commands

```
python -m weightcrack.evaluation.proxy_experiment experiments/proxy_fault_v1
python -m weightcrack.evaluation.verify_proxy experiments/proxy_fault_v1
```

Machine-readable output is `experiments/proxy_fault_v1/experiment_report.json`. The host secret, lock, receipts, and source matrix are in that directory. Rollback is `proposal_inputs/damaged_source.npz`, whose file hash is `9bb3173af038aa8d58070fd28c3c5c6811e5afe0dbc2ef309dd44c3aebad2728`.
