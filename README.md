# WeightCrack

Host-gated sparse projection edits on NumPy matrices. This repository implements one operation, enforces a coordinate budget, and records one synthetic classification study.

## Implemented

`project_remove` updates selected columns of one named matrix:

```
W'[:, c] = W[:, c] - s * u * (u^T W[:, c])
```

`u` is the unit direction. `s` is in `(0, 2]`. Strength 1 removes the projected component. Strengths above 1 reverse it: the stored coefficient along `u` is `(1 - s)` times the original coefficient. Arithmetic uses float64, then one cast back to the source dtype. Supported storage dtypes are float32 and float64.

The edited tensor must be named. An in-memory array is bound with an explicit `tensor_name`. An `.npz` source is addressed by a real key. Arbitrary labels do not all point at one anonymous matrix.

Budgets count distinct scalar coordinates of that tensor. `max_entries` and `max_fraction` use the union of those coordinates. The fraction denominator is the edited tensor's scalar count. Receipts also report the operation count and how many stored values actually changed. `max_abs_delta` is the maximum absolute difference between stored source values and stored candidate values.

A candidate is a new `.wcbundle` directory. `np.savez` receives an exclusively opened file handle, so it cannot append `.npz` to the caller's path. Existing candidate, receipt, and bundle paths are rejected. The completion manifest is written last. Readers reject a directory without a valid manifest. A failed attempt deletes only the directory that attempt created.

Receipts are `weightcrack.receipt.v2`. Disposition `applied` means the file write passed the policy gate. It does not mean a behavioral pass. `verify_bundle` checks the receipt, the policy, the source identity, and exact stored-value reconstruction. Full-file byte identity of a resave is measured separately.

`Session` takes a frozen host config: task id, tensor name, sector, policy, and output root. `propose` and `trial` select columns and strength only. Trials score the same policy-admitted stored candidate that application would save.

The localization direction is the normalized mean difference of two practice activation sets. `top_k` is recorded and does not change that vector. Column ranking uses absolute projection onto that one direction. A large score maximizes removal at a fixed strength. It does not show that the column caused an error.

## Not implemented

- A preservation-memorization solver. The objective in `docs/MATH.md` is motivation.
- Safetensors or Hugging Face loading, and any real language-model edit.
- More than one change, or operations other than `project_remove`.
- A filesystem sandbox. The proxy proposer is a subprocess with a fixed argument list. A modified proposer could still open sibling files.

## Reproduce the proxy study

From this repository:

```
python -m weightcrack.evaluation.proxy_experiment experiments/proxy_fault_v1
python -m weightcrack.evaluation.verify_proxy experiments/proxy_fault_v1
```

The first command refuses to overwrite an existing directory. The study is `f(x) = argmax(W tanh(Ax + b) + c)` with shapes 8, 16, and 4. Seeds, the two-column fault, the dose grid, and the budgets are constants in `weightcrack/evaluation/proxy_experiment.py`. Details and the measured result are in `docs/PROXY_FAULT_V1.md`.

Bundle verification reconstructs through the same application gate as a normal edit. A fresh resave boolean is recorded separately from the historical one. `docs/VERIFIER_REPAIR.md` has the commands and the current-machine observation.

The direction comparison on the same damaged model is recorded in `docs/PROXY_DIRECTION_V2.md`. Its comparison is `NEITHER_METHOD_CONFIRMED`. The loss-sensitive arm had no practice-qualified candidate. The mean-logit arm qualified on practice and, on the fresh confirmation bank, made zero target corrections, one loss, and two preservation regressions. `behavioral_repair` is false. The study keeps the v1 constants and has no random arm.

```
python -m weightcrack.evaluation.verify_direction experiments/proxy_direction_v2
```

A second run needs an empty directory, for example `experiments/proxy_direction_v2_replay`. The official directory refuses an overwrite.

The support comparison is `experiments/proxy_support_v3`, recorded in `docs/PROXY_SUPPORT_V3.md`. Both column rules selected columns `[0, 6]` and produced the same stored edits. The comparison is `NEITHER_METHOD_CONFIRMED`. The fresh-bank finalist made one target correction and two preservation flips. `behavioral_repair` is false.

```
python -m weightcrack.evaluation.verify_support experiments/proxy_support_v3
```

On this machine the study finished in 0.9394292000215501 seconds. The absolute-projection heuristic did not meet the held-out repair rule. The random control had no practice-qualified candidate. That is the result.

## Tests

```
python -m pytest tests -q --tb=line
```

The run on Python 3.12.10 after the support-comparison tests was 42 passed and 1 warning, in 4.92 seconds. `tests/test_phase3.py` returns `True`; pytest warns and the assertions still ran. Phase 3 checks hash lengths of pinned text. Phase 5 checks the same session gate under a second sector name and those hash lengths. Neither test is a coding or math improvement.

## Evidence that is not a WeightCrack measurement

- The 0/20 direct-proposal cohort used a pinned local Qwen controller on synthetic ToyMLP targets. It finalized no candidates. See WeightBridge `docs/AGENT_TOOLS.md` at commit `baa336e7c517139876a1a5d6309af7199ea1e661`.
- WeightBridge Phase P, same commit, `docs/PHASE_III_P.md`: a reconstructed published edit moved 3/48 target presentations to 8/48, with six corrections, one regression, and 24/24 non-target controls retained. Zero-regression qualification failed.
- WeightBridge Phase Q and Phase R, same commit, are supervised LoRA studies. They do not test this column-ranking heuristic.
- External authors' math and instruction numbers are their claims.

Historical WeightCrack runs under `experiments/synthetic_v1`, `experiments/synthetic_strong`, and `experiments/synthetic_search` are unchanged. `docs/REAL_MATRIX_STATUS.md` records that no real matrix was loaded.
