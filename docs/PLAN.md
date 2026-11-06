# WeightCrack Plan

## Objective

A host-gated meta-algorithm that can take a model, localize weights responsible for a target sector (math, coding, refusal, style, etc.), propose a sparse edit, have the host apply it under policy, and measure a performance upgrade on a frozen holdout while preserving sentinel behavior.

## Current state

- Schemas, policy gate, receipt path, and evaluation loop contracts exist.
- Direction-removal formula and preservation-memorization framing are documented.
- Tensor-level apply and practice-only localization instruments are still stubs.
- Prior WeightBridge pilots show direction removal can work (1/3 constrained cases, published-style reconstructions) while free-form proposal does not (0/20 direct transfer).

## Phase 1 — Make the working edit runnable

**Status:** Done (NumPy proxy).

Direction-removal formula is implemented and verified by round-trip test (`tests/test_phase1.py`). Receipt is emitted. Safetensors loader can be layered on the same apply logic.

1. Finish the safetensors load / apply / write path in `weightcrack/patches/huggingface.py`.
   - Support the closed-form column removal already specified in `docs/MATH.md`.
   - Enforce policy bounds before any write.
   - Emit a complete receipt with base hash, patch hash, policy hash, and changed-parameter count.
2. Add a minimal CLI entry point: `weightcrack apply --base ... --patch ... --policy ... --out ...`.
3. Verify round-trip: apply, reload, confirm hashes, confirm the mathematical operation matches the formula.

**Exit criterion:** A known direction-removal patch can be applied to the pinned Qwen3.5-0.8B (or a smaller proxy) and the resulting candidate reloads with matching receipts.

## Phase 2 — Practice-only localization instruments

**Status:** Done (NumPy proxy).

Activation-difference singular vector, column ranking by projection, and host-executed direction trial are implemented and verified (`tests/test_phase2.py`). All operate on caller-supplied practice data only.

1. Activation-difference report (practice examples only).
   - Mean residual or MLP output on failing examples minus mean on correct/sentinel examples.
   - Return top singular vector as candidate direction \( u \).
2. Column-ranking report.
   - For a candidate \( u \), score columns by \( |u^\top W_{:, c}| \) and return top-k.
3. Direction-trial primitive.
   - Host applies the removal formula under the current policy budget, scores practice, returns the delta.
   - No permanent write.
4. Wire these as agent tools that never receive holdout paths or values.

**Exit criterion:** An agent (or scripted caller) can request a localization report, select a direction and column set, run a trial, and receive a practice delta without ever seeing raw weights or holdout data.

## Phase 3 — Sector probe suites

**Goal:** Frozen, hash-pinned practice and holdout suites for at least one sector.

1. Define a narrow first sector (math repetition / loop reduction is the strongest existing signal).
2. Pin practice examples, holdout examples, and sentinel controls with hashes.
3. Implement the evaluation loop so it reports target gain and sentinel flips.
4. Require both a minimum target gain and zero (or bounded) sentinel flips for a pass.

**Exit criterion:** A candidate can be scored against the frozen suite and the result is recorded with the same hashes on every run.

## Phase 4 — Agent loop under governance

**Goal:** Close the propose → gate → evaluate loop with an agent that only sees practice signals.

1. Session controller that exposes only the Phase 2 tools and the practice suite.
2. Agent proposes a `weightcrack.patch.v1`.
3. Host gates, applies (or rejects), evaluates on holdout once, and appends a ledger record.
4. Measure success rate on the induced-fault or loop-reduction suite.

**Exit criterion:** At least one agent-proposed patch clears the full gate (policy, reload, target gain, sentinel preservation) on a frozen suite.

## Phase 5 — Second sector and robustness

**Goal:** Show the same loop works beyond the first sector without relaxing invariants.

1. Add a second sector probe (coding precision or refusal strength).
2. Confirm practice/holdout separation and receipt audit still hold.
3. Document failure modes and any policy adjustments required.

## Constraints that do not change

- Agent never receives direct write access or holdout data.
- Every candidate is a new file + receipt.
- Policy gate remains the sole authority for application.
- Practice and holdout stay separated by process boundary.
- All claims remain scoped to the measured suite and model.

## Immediate next step

Finish the tensor-level apply path for the direction-removal formula (Phase 1). That is the smallest change that turns the documented math into a runnable candidate.
