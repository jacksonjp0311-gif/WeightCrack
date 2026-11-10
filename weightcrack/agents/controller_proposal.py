"""Controller-facing proposal path.

Accepts an external patch proposal, runs it through ``construct_candidate``,
and returns the accounting when the edit is admissible. This function does
not write a bundle. The host remains the only writer.

Programming errors propagate. An ordinary rejection is only ``EditRejected``
or ``PolicyViolation`` from the shared gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..canonical import array_content_sha256
from ..errors import EditRejected, PolicyViolation
from ..patches.huggingface import construct_candidate
from ..schema import WeightCrackPatch, WeightCrackPolicy


def propose_and_gate(
    patch: WeightCrackPatch,
    policy: WeightCrackPolicy,
    tensors: dict[str, Any],
) -> dict[str, Any]:
    """Gate one external proposal through the shared candidate constructor.

    The returned record is not a saved bundle. A source array is left unchanged.
    """
    if type(patch) is not WeightCrackPatch:
        return {"status": "rejected", "reason": "patch is not a WeightCrackPatch"}
    if type(policy) is not WeightCrackPolicy:
        return {"status": "rejected", "reason": "policy is not a WeightCrackPolicy"}
    if not isinstance(tensors, dict):
        return {"status": "rejected", "reason": "tensors were not supplied"}
    if len(patch.changes) != 1:
        return {"status": "rejected", "reason": "this iteration supports exactly one change"}
    change = patch.changes[0]
    if not isinstance(change.tensor, str) or not change.tensor:
        return {"status": "rejected", "reason": "tensor name is required"}
    if change.tensor not in tensors:
        return {"status": "rejected", "reason": f"tensor {change.tensor} not supplied"}
    matrix = tensors[change.tensor]
    snapshot = np.array(matrix, copy=True) if isinstance(matrix, np.ndarray) else None
    try:
        candidate, accounting, _normalized = construct_candidate(
            matrix, patch, policy, change.tensor
        )
    except (EditRejected, PolicyViolation) as exc:
        return {"status": "rejected", "reason": str(exc)}
    if snapshot is not None and (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != snapshot.shape
        or matrix.dtype != snapshot.dtype
        or not np.array_equal(matrix, snapshot)
    ):
        raise RuntimeError("source tensor was mutated")
    return {
        "status": "admissible",
        "accounting": accounting,
        "candidate_shape": list(candidate.shape),
        "candidate_content_sha256": array_content_sha256(candidate),
        "source_sha256": patch.base_sha256,
        "note": "Host must persist and write receipt. This function does not write.",
    }
