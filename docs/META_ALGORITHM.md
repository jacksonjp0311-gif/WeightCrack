# Meta-algorithm sketch

## Objective

Given a model and a target sector (math accuracy, coding precision, refusal strength, style consistency, etc.), produce a sparse weight edit that improves performance on that sector while preserving behavior on sentinel controls.

## Loop

1. **Inspect**  
   Header + bounded statistics + optional activation probes on practice examples only.

## Localization (refined)

Host-executed, practice-only instruments:

- Activation difference: mean residual / MLP output on failing practice examples minus mean on correct or sentinel examples. Top singular vector is a candidate direction \( u \).
- Column ranking: for a candidate \( u \), score columns by \( |u^\top W_{:, c}| \) and return the top-k.
- Direction trial: host applies the closed-form removal
  \[
  W'_{:, c} = W_{:, c} - s \, u \, (u^\top W_{:, c})
  \]
  under the current policy budget, scores practice, and returns the delta. No permanent write.

Agent proposes a `weightcrack.patch.v1` that names the tensor, the direction, the columns, and the strength. Host gates and, if accepted, writes the candidate + receipt.

3. **Propose**  
   Agent emits a `weightcrack.patch.v1` grounded in the localization report and practice scores.

4. **Gate**  
   Host enforces policy (allow-list, max entries, fraction, abs delta, rank).

5. **Apply**  
   New candidate checkpoint + immutable receipt.

6. **Evaluate**  
   Frozen holdout: sector targets must improve; sentinel controls must not flip.

7. **Record**  
   Append-only ledger entry with hashes, practice/holdout separation proof, and disposition.

## Next instruments to add

- Bounded activation/attention probe report (practice-only).
- Host-executed direction trial primitive (`try_direction` that returns practice delta under policy budget).
- Sector-specific probe suites (hash-pinned).
- Receipt verification and rollback path.
