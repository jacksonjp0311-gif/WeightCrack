"""Controller-facing proposal path.

Accepts an external patch proposal (from a controller that can read weights),
runs it through the existing policy gate, and returns the accounting if admissible.
Does not write files. The host remains the only writer.

This is the shift documented in docs/SHIFT.md: WeightCrack is the constitution,
not the localizer.
"""

from __future__ import annotations

from typing import Any

from ..patches.huggingface import construct_candidate
from ..policy import gate_patch, PolicyViolation
from ..schema import WeightCrackPatch, WeightCrackPolicy


def propose_and_gate(
    patch: WeightCrackPatch,
    policy: WeightCrackPolicy,
    tensors: dict[str, Any],
) -> dict[str, Any]:
    """
    Gate an external proposal through the full construct_candidate path.
    Returns accounting or a rejection reason.
    Does not write. Host decides whether to persist.
    """
    change = patch.changes[0]
    matrix = tensors.get(change.tensor)
    if matrix is None:
        return {"status": "rejected", "reason": f"tensor {change.tensor} not supplied"}

    try:
        gate_patch(patch, policy, base_param_count=sum(t.size for t in tensors.values()))
        candidate, accounting, _ = construct_candidate(matrix, patch, policy, change.tensor)
    except (PolicyViolation, Exception) as exc:
        return {"status": "rejected", "reason": str(exc)}

    return {
        "status": "admissible",
        "accounting": accounting,
        "candidate_shape": list(candidate.shape),
        "source_sha256": patch.base_sha256,
        "note": "Host must persist and write receipt. This function does not write.",
    }
