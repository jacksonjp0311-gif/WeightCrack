# Verification repair

Three checks were repaired without rewriting `experiments/proxy_fault_v1`, `experiments/synthetic_v1`, `experiments/synthetic_strong`, or `experiments/synthetic_search`.

The archived v1 outcome is unchanged. Heuristic columns are `[13, 4]`. Injected fault columns are `[10, 7]`. Practice had one correction, zero losses, and zero preservation flips. Held-out targets had two corrections and one loss. Held-out preservation had three flips, all correct-to-incorrect regressions. `behavioral_repair` is false. The random arm has no practice-qualified candidate.

Across the combined 512 held-out examples, the saved candidate moved 482 correct answers to 480. That combined count is descriptive context. The original scoring rule, target and preservation counted separately, remains the v1 result.

## What changed

`verify_bundle` now reconstructs through `construct_candidate`. A bundle must agree on the supported operation, the single change, tensor and operation permissions, the submitted base identity, input and budget limits, the regenerated normalized patch, source and candidate manifests, untouched tensor names and dtypes, accounting, and the duplicated receipt and event identities. Exact stored values are still required.

A fresh `np.savez` boolean is an environment observation. Original artifact hashes and exact tensor reconstruction remain mandatory. The historical resave boolean list does not have to equal the current machine's list. The v1 report was not rewritten.

`verify_study` for `proxy_fault_v1` treats the lock as the selection record. It recomputes practice and held-out counts, population sizes, prediction hashes, repair flags, applied-bundle counts, and stored update norms from the reconstructed candidate. The expected method set is the v1 pair, heuristic and random. A later study does not inherit that pair. Repair flags must be real booleans.

Unsigned file hashes do not authenticate a rewritten archive or show who authorized it. These tests check semantic consistency of a bundle whose completion manifest was refreshed.

## Commands

From this repository, after the repair:

```
python -m pytest tests -q --tb=line
python -m weightcrack.evaluation.verify_proxy experiments/proxy_fault_v1
```

`pytest` reported 34 passed and 1 warning in 3.31 seconds. The warning is `PytestReturnNotNoneWarning` from `tests/test_phase3.py::test_suite_pins_and_scoring`, which returns a bool. That test was left as it was.

The fresh-process archive check printed:

```
status=verified study_status=complete wall_seconds=0.9394292000215501
current_resave_byte_identical=[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]
historical_resave_byte_identical=[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]
resave_matches_historical=true
python=3.12.10 numpy=2.5.4
```

On this Windows runtime the sixteen fresh flags happen to match the historical flags. That match is not the acceptance condition. `value_exact_reconstruction` is true, and the saved behavioral repair flag remains false.

The pre-edit hash manifest of the four archived experiment directories still matches all 76 files.

After the direction study, the same archive command printed the same verified line and the same wall time. A later `pytest` run reported 34 passed and the same phase-3 warning in 3.09 seconds. The 76-file manifest still had 0 missing and 0 changed files. The archived report's `behavioral_repair` value remains false.
