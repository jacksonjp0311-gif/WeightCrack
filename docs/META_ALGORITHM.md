# Meta-algorithm as implemented

## Scope

One host-owned edit of one matrix, then an optional behavioral score. Sectors such as math accuracy or coding precision are names a host may put in `SessionConfig.sector`. The engine does not treat the name as evidence of that skill.

## Steps that exist

1. The host freezes the tensor name, the source bytes, the policy, the sector, and the output directory.
2. A proposal may choose columns and a strength inside that policy. It cannot replace the path or the policy.
3. The host builds one `project_remove` candidate in float64, casts it to the source dtype, and rejects it when a coordinate budget or the stored absolute delta fails.
4. A trial uses that same candidate. Application writes it only into a new `.wcbundle`, manifest last.
5. `verify_bundle` recomputes the candidate from the normalized patch and compares stored values. A resave's byte identity is a separate flag.
6. Behavioral status is recorded by the study harness. `applied` stays the file disposition. Invalid scores, missing manifests, and hash mismatches fail closed.

## Localization that exists

The practice direction is a normalized mean difference at a named activation site. On the proxy study that site is `post_linear_logits`, and its width is the row count of `W`. Ranking sorts columns by absolute projection. `top_k` does not create extra spectral components of the direction.

The preservation-memorization objective is not a step in this loop.

## Proxy comparison

The host derives one direction from damaged-model practice logits. Two methods share it: the top two ranked columns, and two columns drawn without replacement from a fixed control generator. Each method gets the same eight update-norm doses. Strengths are calibrated from source geometry before any candidate is scored. Selection uses practice corrections, practice losses, and preservation flips. Held-out labels are read after `proposal_lock.json` is on disk.

The measured outcome is in `docs/PROXY_FAULT_V1.md`. The heuristic's held-out repair flag is false. The random arm did not qualify on practice.

## Not in this loop

Model training, a general agent framework, multi-tensor edits, and a real checkpoint. Suite files under `weightcrack/evaluation/suites.py` pin example text. Their hash checks do not score a model.
