"""Policy-gated direction-removal and sparse edits.

Phase 1 implementation: closed-form column removal on NumPy matrices.
Safetensors loader can be layered on top of the same apply logic.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import WeightCrackPatch, WeightCrackPolicy


class EditRejected(Exception):
    pass


def _remove_direction(
    matrix: np.ndarray,
    direction: np.ndarray,
    columns: list[int],
    strength: float,
) -> np.ndarray:
    """Apply W'[:, c] = W[:, c] - s * u * (u^T W[:, c]) for selected columns."""
    if matrix.ndim != 2:
        raise EditRejected("direction removal requires a 2D matrix")
    u = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(u)
    if not math.isfinite(norm) or norm < 1e-12:
        raise EditRejected("direction has zero or invalid norm")
    u = u / norm
    out = matrix.astype(np.float64).copy()
    for c in columns:
        if not 0 <= c < matrix.shape[1]:
            raise EditRejected(f"column {c} out of range")
        col = out[:, c]
        proj = float(np.dot(u, col))
        out[:, c] = col - strength * u * proj
    return out


def apply_patch(
    base: str | Path | np.ndarray,
    patch: WeightCrackPatch,
    policy: WeightCrackPolicy,
    output: str | Path,
) -> dict[str, Any]:
    """
    Apply a gated patch.
    `base` may be a path to a .npz or a NumPy array (for the Phase 1 proxy).
    Writes a new file and returns a receipt.
    """
    out_path = Path(output)
    if out_path.exists():
        raise EditRejected(f"output path exists: {out_path}")

    # Load
    if isinstance(base, np.ndarray):
        matrix = base
        base_hash = hashlib.sha256(matrix.tobytes()).hexdigest()
    else:
        base_path = Path(base)
        data = np.load(base_path)
        # Expect a single matrix under key 'weight' for the proxy
        matrix = data["weight"]
        base_hash = hashlib.sha256(matrix.tobytes()).hexdigest()

    if patch.base_sha256 != base_hash:
        raise EditRejected("base hash mismatch")

    # Gate
    from ..policy import gate_patch
    gate_patch(patch, policy, matrix.size)

    # Apply
    result = matrix.copy()
    records = []
    for change in patch.changes:
        if change.op == "project_remove":
            if change.direction is None or change.index is None or change.strength is None:
                raise EditRejected("project_remove requires direction, index (columns), strength")
            result = _remove_direction(result, change.direction, change.index, change.strength)
            records.append({
                "tensor": change.tensor,
                "op": "project_remove",
                "columns": change.index,
                "strength": change.strength,
            })
        else:
            raise EditRejected(f"Phase 1 supports project_remove; got {change.op}")

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, weight=result)

    receipt = {
        "schema": "weightcrack.receipt.v1",
        "base_sha256": base_hash,
        "patch_hash": patch.compute_hash(),
        "sector": patch.sector,
        "reason": patch.reason,
        "policy_hash": hashlib.sha256(policy.model_dump_json().encode()).hexdigest(),
        "changed_entries": len(records),
        "records": records,
        "status": "applied",
    }
    receipt_path = out_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return receipt
