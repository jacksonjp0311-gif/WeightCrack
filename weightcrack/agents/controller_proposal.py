"""Controller-facing proposal path.

Accepts an external patch proposal (from a controller that can read weights),
runs it through the existing policy gate, and returns the accounting if admissible.
Does not write files. The host remains the only writer.

This is the shift documented in docs/SHIFT.md: WeightCrack is the constitution,
not the localizer.
"""

from __future__ import annotations

from typing import Any

from ..patches.huggingface import apply_project_remove
from ..policy import gate_patch, enforce_realized, PolicyViolation
from ..schema import WeightCrackPatch, WeightCrackPolicy


def propose_and_gate(
    patch: WeightCrackPatch,
    policy: WeightCrackPolicy,
    tensors: dict[str, Any],
) -> dict[str, Any]:
    """
    Gate an external proposal and realize it if admissible.
    Returns accounting or a rejection reason.
    Does not write. Host decides whether to persist.
    """
    try:
        gate_patch(patch, policy, base_param_count=sum(t.size for t in tensors.values()))
    except PolicyViolation as exc:
        return {"status": "rejected", "reason": str(exc)}

    change = patch.changes[0]
    if change.op != "project_remove":
        return {"status": "rejected", "reason": f"op {change.op} not yet realized"}

    matrix = tensors.get(change.tensor)
    if matrix is None:
        return {"status": "rejected", "reason": f"tensor {change.tensor} not supplied"}

    try:
        candidate, accounting = apply_project_remove(
            matrix,
            change.direction,
            change.index or [],
            change.strength or 0.0,
        )
        enforce_realized(accounting, policy, matrix.size)
    except Exception as exc:
        return {"status": "rejected", "reason": str(exc)}

    return {
        "status": "admissible",
        "accounting": accounting,
        "candidate_shape": list(candidate.shape),
        "note": "Host must persist and write receipt. This function does not write.",
    }
