# WeightCrack plan

## Objective

Apply one sparse projection edit under a host policy and measure whether that edit changes predictions. The objective is not a general sector agent.

## What the code does

- `project_remove` on one named float32 or float64 matrix, in float64, with one cast back to the source dtype.
- Coordinate budgets, stored-delta checks, and exclusive `.wcbundle` publication.
- One trial path shared with application. A trial cannot skip the policy gate.
- A host `SessionConfig`. The caller does not supply a path, a policy, or a sector at propose time.
- Verification that reconstructs stored values and checks source identity.
- One preregistered synthetic classifier study, `experiments/proxy_fault_v1`.

## What the exit criteria still require

Phase labels below are the original plan. "Done" is not used where the stated exit criterion is unmet.

### Phase 1 — runnable direction removal

The NumPy formula, receipt, and reload check exist (`tests/test_phase1.py`). The exit criterion also asked for a pinned Qwen apply. No real matrix has been edited. Safetensors loading is not implemented.

### Phase 2 — practice-only localization

The direction is a normalized mean difference of two nonempty practice sets. It is one vector. `top_k` does not change it. Column ranking is absolute projection. `direction_trial` and `try_direction` call the same gated constructor. The exit criterion said the caller would not see raw weights. The session and the proxy proposer both receive the damaged matrix, because ranking reads it.

### Phase 3 — sector suites

`weightcrack/evaluation/suites.py` pins math and coding example text and exposes a string-match stub. The test checks hash length. It does not score a model.

### Phase 4 — governed session

`Session.propose` and `Session.trial` use the host config. The phase test checks that the saved array matches the trial. It does not measure holdout target gain. The proxy study is the behavioral measurement, and it did not meet the repair rule.

### Phase 5 — second sector

The same session accepts a host sector name other than a hardcoded `math_loop`. The coding suite hash is distinct from the math suite hash and is 64 hex characters. That is not a coding-performance result. The earlier phase-five test did not show a coding improvement; it exercised a matrix-norm callback and suite hashes.

## Constraints the implementation keeps

- Unsupported operations are rejected.
- Nonfinite inputs and nonfinite results are rejected. Invalid numbers are not clamped.
- Existing artifacts are not replaced.
- Held-out labels in the proxy study are outside the proposer argument list. The host writes `proposal_lock.json` before it opens `host_only/`. The event log records that order. This is not a hardened filesystem sandbox.
- A successful bundle is disposition `applied`. Behavioral qualification is a separate field in the study report. The receipt leaves it null. The verifier rejects a receipt that claims a behavioral pass.

## Next work that this iteration did not do

Real-model loading, training, and any second fault or second dataset. `docs/REAL_MATRIX_STATUS.md` still records the failed Qwen download.
