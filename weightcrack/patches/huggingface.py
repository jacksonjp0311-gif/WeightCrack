"""Policy-gated projection removal with strict artifact and accounting rules.

Working precision is float64. Storage dtype of the source is preserved.
Strengths above 1 reverse the projected component.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import WeightCrackPatch, WeightCrackPolicy


class EditRejected(Exception):
    pass


WORKING_DTYPE = np.float64


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_content_hash(arr: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _is_finite_array(arr: np.ndarray) -> bool:
    return bool(np.isfinite(arr).all())


def validate_project_remove(
    matrix: np.ndarray,
    direction: np.ndarray | list[float],
    columns: list[int],
    strength: float,
) -> tuple[np.ndarray, list[int], float]:
    """Validate inputs for project_remove. Returns normalized direction, columns, strength."""
    if matrix.ndim != 2:
        raise EditRejected("project_remove requires a 2D matrix")
    if not _is_finite_array(matrix):
        raise EditRejected("source matrix contains nonfinite values")

    u = np.asarray(direction, dtype=WORKING_DTYPE)
    if u.ndim != 1 or u.shape[0] != matrix.shape[0]:
        raise EditRejected("direction length must match matrix rows")
    if not _is_finite_array(u):
        raise EditRejected("direction contains nonfinite values")
    norm = float(np.linalg.norm(u))
    if not math.isfinite(norm) or norm < 1e-12:
        raise EditRejected("direction has zero or invalid norm")
    u = u / norm

    if not isinstance(columns, list) or not columns:
        raise EditRejected("columns must be a nonempty list")
    if len(set(columns)) != len(columns):
        raise EditRejected("duplicate columns")
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in columns):
        raise EditRejected("columns must be integers")
    if any(c < 0 or c >= matrix.shape[1] for c in columns):
        raise EditRejected("column out of range")

    if not isinstance(strength, (int, float)) or isinstance(strength, bool):
        raise EditRejected("strength must be a real number")
    if not math.isfinite(strength):
        raise EditRejected("strength is nonfinite")
    if not 0 < strength <= 2:
        raise EditRejected("strength must be in (0, 2]")

    return u, columns, float(strength)


def apply_project_remove(
    matrix: np.ndarray,
    direction: np.ndarray | list[float],
    columns: list[int],
    strength: float,
    storage_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply the formula and return the candidate in the original storage dtype
    together with accounting.
    """
    u, columns, strength = validate_project_remove(matrix, direction, columns, strength)
    storage_dtype = np.dtype(storage_dtype or matrix.dtype)

    working = matrix.astype(WORKING_DTYPE, copy=True)
    touched: set[tuple[int, int]] = set()
    for c in columns:
        col = working[:, c]
        proj = float(np.dot(u, col))
        working[:, c] = col - strength * u * proj
        for r in range(matrix.shape[0]):
            touched.add((r, c))

    if not _is_finite_array(working):
        raise EditRejected("projection produced nonfinite values")

    candidate = working.astype(storage_dtype)
    if not _is_finite_array(candidate):
        raise EditRejected("casting to storage dtype produced nonfinite values")

    # Realized stored deltas
    stored_src = matrix.astype(storage_dtype)
    delta = np.abs(candidate.astype(WORKING_DTYPE) - stored_src.astype(WORKING_DTYPE))
    max_delta = float(np.max(delta)) if delta.size else 0.0
    changed_mask = delta > 0
    changed_coords = int(np.sum(changed_mask))

    accounting = {
        "operation_count": 1,
        "touched_coordinates": len(touched),
        "changed_coordinates": changed_coords,
        "max_abs_stored_delta": max_delta,
        "columns": columns,
        "strength": strength,
    }
    return candidate, accounting


def _exclusive_output_bundle(output: Path) -> Path:
    """Return an exclusive path for the candidate .npz. Rejects existing destinations."""
    if output.suffix == ".npz":
        candidate = output
    else:
        candidate = output.with_suffix(".npz")
    receipt = candidate.with_suffix(".receipt.json")
    manifest = candidate.with_suffix(".manifest.json")
    for path in (candidate, receipt, manifest):
        if path.exists() or path.is_symlink():
            raise EditRejected(f"destination already exists: {path}")
    return candidate


def apply_patch(
    base: str | Path | np.ndarray,
    patch: WeightCrackPatch,
    policy: WeightCrackPolicy,
    output: str | Path,
) -> dict[str, Any]:
    """
    Apply a gated project_remove patch.
    Writes an exclusive candidate bundle and receipt. Never overwrites.
    """
    out_path = Path(output)
    candidate_path = _exclusive_output_bundle(out_path)

    # Load source
    if isinstance(base, np.ndarray):
        matrix = base
        source_path = None
    else:
        source_path = Path(base)
        if source_path.resolve() == candidate_path.resolve():
            raise EditRejected("source and output alias")
        data = np.load(source_path)
        if "weight" not in data:
            raise EditRejected("source npz missing 'weight' array")
        matrix = data["weight"]

    if not _is_finite_array(matrix):
        raise EditRejected("source contains nonfinite values")

    base_hash = _array_content_hash(matrix)
    if patch.base_sha256 != base_hash:
        raise EditRejected("base hash mismatch")

    if len(patch.changes) != 1 or patch.changes[0].op != "project_remove":
        raise EditRejected("this iteration supports exactly one project_remove")

    change = patch.changes[0]
    if change.tensor not in policy.allow_tensors:
        raise EditRejected(f"tensor {change.tensor} not allowlisted")

    # Shared construction path (also used by trials)
    candidate, accounting = apply_project_remove(
        matrix,
        change.direction,
        change.index or [],
        change.strength or 0.0,
        storage_dtype=matrix.dtype,
    )

    # Policy on realized values
    from ..policy import enforce_realized
    enforce_realized(accounting, policy, matrix.size)

    # Stage write
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".npz", dir=candidate_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "wb") as f:
            np.savez(f, weight=candidate)
        # Publish
        os.replace(tmp_path, candidate_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    candidate_hash = _sha256_bytes(candidate_path.read_bytes())
    receipt = {
        "schema": "weightcrack.receipt.v1",
        "base_sha256": base_hash,
        "candidate_sha256": candidate_hash,
        "candidate_content_hash": _array_content_hash(candidate),
        "patch_hash": patch.compute_hash(),
        "policy_hash": hashlib.sha256(policy.model_dump_json().encode()).hexdigest(),
        "sector": patch.sector,
        "reason": patch.reason,
        "accounting": accounting,
        "storage_dtype": str(matrix.dtype),
        "status": "applied",
    }
    receipt_path = candidate_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
