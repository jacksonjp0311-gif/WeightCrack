# Controller and final-layer interval repair

This note records an engineering repair. It does not replace `docs/FEASIBILITY.md`. That file still describes the earlier 200-point grid on seeds 300–319: 14 feasible and 14 matched after storage.

## What changed

The controller calls `construct_candidate` with the patch tensor name and unpacks the candidate, the accounting, and the normalized edit. Ordinary `EditRejected` and `PolicyViolation` results stay rejections. A programming exception propagates. The controller does not write a bundle.

The feasibility replay now solves strength intervals for the fixed-hidden-layer classifier `argmax(W tanh(Ax+b)+c)`. Hidden activations stay fixed, so logit differences are affine in the strength. The permitted window is `(0, 2]`. The same 200-point grid is only a comparison sample. This calculation is exact for that final-layer proxy. It is not an exact solver for a transformer layer whose later activations depend on the edited weights.

Verifier repairs from `dev/support-direction-v3` commit `f82334f` were copied as three files: `weightcrack/verify.py`, `weightcrack/evaluation/verify_proxy.py`, and `tests/test_verify_repairs.py`. The support-direction branch, its protocol, and `experiments/proxy_support_v3` were not merged and were not edited.

## Fresh measurement

Command:

```
python -m weightcrack.evaluation.feasibility_replay experiments/feasibility_interval_check
```

Python 3.12.10 and NumPy 2.5.4. Seeds 300–319. The report is `experiments/feasibility_interval_check/report.json`, SHA-256 `d65be78961a833e7e2bac435a203c78056a389f0f3397806c638b8e1d0f12cf5`.

- 16 seeds had a nonempty policy-clipped interval.
- All 16 selected strengths were reconstructed through `construct_candidate` and matched an independent `apply_project_remove`.
- All 16 stored float32 candidates kept previously correct targets and corrected at least one incorrect target.
- Seeds 302, 305, 307, and 314 had an empty behavioral interval. The comparison grid also found no feasible strength for those seeds.
- Analytic intervals and the 200-point grid disagreed at 0 sampled strengths away from interval endpoints.
- 240 sampled strengths were behaviorally feasible and outside the stored-delta cap. Seed 303 is one case: the behavioral interval reaches about 0.802, and the cap cuts the admitted interval at about 0.152. The selected strength is the midpoint of the admitted interval.

The selected strength is the only stored candidate reconstructed for each seed. The float64 cap is not a substitute for the float32 gate at every boundary point.

## Archived studies

`git diff --name-only -- experiments docs` was empty before this note and the new report were added. `python -m weightcrack.evaluation.verify_proxy experiments/proxy_fault_v1` returned `status=verified`, `study_status=complete`, and `wall_seconds=0.9394292000215501`. All 16 current and historical resaves were byte-identical.

## Checks in this repair

`python -m pytest -q --tb=short` returned 38 passed and 1 warning. The warning is the existing `PytestReturnNotNoneWarning` from `tests/test_phase3.py::test_suite_pins_and_scoring`.

## Next experiment

A real-model test is still blocked. `docs/REAL_MATRIX_STATUS.md` records `DOWNLOAD_FAILED` for Qwen/Qwen2.5-0.5B. No matrix was loaded, and no probe ran on real weights.

When a local `.npz` with one `weight` array is available, run the existing `weightcrack/evaluation/real_matrix_probe.py` once, through `construct_candidate`, on the practice path only. Do not start a training job or another synthetic seed sweep from this repair. The proxy comparison on `dev/support-direction-v3` remains `NEITHER_METHOD_CONFIRMED`.
