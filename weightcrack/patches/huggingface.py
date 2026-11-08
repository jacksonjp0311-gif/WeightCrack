"""Policy-gated projection removal.

Working precision is float64. The source storage dtype is preserved by one
cast after that arithmetic. Strength 1 removes the projected component.
Strengths above 1 reverse it: the stored component is (1 - strength) times
the original component. The declared range is (0, 2], and it is not pure removal.

This path follows the column-removal formula inspected in WeightBridge
``core.apply_patch`` (remove_column_direction). It does not import WeightBridge.
The supported operation in this iteration is ``project_remove`` only.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..bundle import publish_bundle
from ..canonical import (
    array_content_sha256,
    canonical_sha256,
    f64_hex,
    file_sha256,
    model_payload,
    model_sha256,
    tensor_manifest,
)
from ..errors import EditRejected
from ..policy import enforce_realized, gate_patch
from ..schema import WeightCrackPatch, WeightCrackPolicy

WORKING_DTYPE = np.float64
SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
STRENGTH_RANGE = (0.0, 2.0)  # exclusive lower bound, inclusive upper bound
SUPPORTED_OPERATION = "project_remove"


def _sha256_text(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _check_destination(source: Path | None, output: Path) -> None:
    if output.suffix != ".wcbundle":
        raise EditRejected(
            "output must be a new .wcbundle directory; extensionless names and .npz paths are rejected"
        )
    if output.exists() or output.is_symlink():
        raise EditRejected(f"destination already exists: {output}")
    if source is None:
        return
    if source.is_symlink():
        raise EditRejected("source symlinks are not accepted")
    planned = [
        output,
        output / "candidate.npz",
        output / "receipt.json",
        output / "events.jsonl",
        output / "manifest.json",
    ]
    for candidate in planned:
        if _same_path(candidate, source):
            raise EditRejected("source and output alias")
    source_abs = Path(os.path.abspath(source))
    output_abs = Path(os.path.abspath(output))
    if source_abs == output_abs or source_abs in output_abs.parents:
        raise EditRejected("source and output alias")
    try:
        output_abs.relative_to(source_abs)
    except ValueError:
        return
    raise EditRejected("output is nested in the source")


def _require_matrix(matrix: np.ndarray) -> np.ndarray:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise EditRejected("project_remove requires a 2D matrix")
    if matrix.dtype not in SUPPORTED_DTYPES:
        raise EditRejected(f"unsupported storage dtype: {matrix.dtype}")
    if matrix.size < 1 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise EditRejected("matrix has no editable coordinates")
    if not np.isfinite(matrix).all():
        raise EditRejected("source contains nonfinite values")
    return matrix


def _columns(columns: list[int], width: int) -> list[int]:
    if not isinstance(columns, list) or not columns:
        raise EditRejected("columns must be a nonempty list")
    parsed: list[int] = []
    for column in columns:
        if type(column) is not int:
            raise EditRejected("columns must be integers")
        if column < 0 or column >= width:
            raise EditRejected("column out of range")
        parsed.append(column)
    if len(set(parsed)) != len(parsed):
        raise EditRejected("duplicate columns")
    return parsed


def _strength(strength: float) -> float:
    if type(strength) not in {int, float}:
        raise EditRejected("strength must be a real number")
    if not math.isfinite(strength) or not STRENGTH_RANGE[0] < float(strength) <= STRENGTH_RANGE[1]:
        raise EditRejected("strength must be finite and in (0, 2]")
    return float(strength)


def _direction(direction, rows: int) -> np.ndarray:
    if isinstance(direction, np.ndarray):
        if direction.ndim != 1 or not np.issubdtype(direction.dtype, np.floating):
            raise EditRejected("direction length must match matrix rows")
        values = np.array(direction, dtype=WORKING_DTYPE, copy=True)
    elif isinstance(direction, list):
        if not direction or any(type(item) not in {int, float} for item in direction):
            raise EditRejected("direction must be a list of real numbers")
        values = np.asarray(direction, dtype=WORKING_DTYPE)
    else:
        raise EditRejected("direction must be a list of real numbers")
    if values.ndim != 1 or values.shape[0] != rows:
        raise EditRejected("direction length must match matrix rows")
    if not np.isfinite(values).all():
        raise EditRejected("direction contains nonfinite values")
    return values


def validate_project_remove(matrix, direction, columns, strength):
    """Validate inputs. Returns the original float64 direction, columns, and strength."""
    matrix = _require_matrix(np.asarray(matrix) if not isinstance(matrix, np.ndarray) else matrix)
    original = _direction(direction, matrix.shape[0])
    norm = float(np.linalg.norm(original))
    if not math.isfinite(norm) or norm < 1e-12:
        raise EditRejected("direction has zero or invalid norm")
    return original, _columns(columns, matrix.shape[1]), _strength(strength)


def apply_project_remove(
    matrix: np.ndarray,
    direction,
    columns: list[int],
    strength: float,
    storage_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the stored candidate and coordinate accounting. Does not write a file."""
    matrix = _require_matrix(matrix)
    original, columns, strength = validate_project_remove(matrix, direction, columns, strength)
    if storage_dtype is not None and np.dtype(storage_dtype) != matrix.dtype:
        raise EditRejected("storage dtype does not match the source")
    unit = original / float(np.linalg.norm(original))
    working = np.array(matrix, dtype=WORKING_DTYPE, copy=True)
    for column in columns:
        projection = float(np.dot(unit, working[:, column]))
        working[:, column] = working[:, column] - strength * unit * projection
    if not np.isfinite(working).all():
        raise EditRejected("projection produced nonfinite values")
    with np.errstate(over="ignore", invalid="ignore"):
        candidate = working.astype(matrix.dtype, copy=True)
    if candidate.dtype != matrix.dtype or not np.isfinite(candidate).all():
        raise EditRejected("casting to storage dtype produced nonfinite values")
    stored = np.array(matrix, dtype=matrix.dtype, copy=True)
    untouched = np.ones(matrix.shape, dtype=bool)
    for column in columns:
        untouched[:, column] = False
    if not np.array_equal(candidate[untouched], stored[untouched]):
        raise EditRejected("untouched stored values changed")
    delta = np.abs(candidate.astype(WORKING_DTYPE) - stored.astype(WORKING_DTYPE))
    changed = np.not_equal(candidate, stored)
    touched = matrix.shape[0] * len(columns)
    accounting = {
        "operation_count": 1,
        "touched_coordinates": int(touched),
        "changed_coordinates": int(changed.sum()),
        "fraction_denominator": int(matrix.size),
        "fraction_denominator_definition": "edited_tensor_scalar_count",
        "touched_fraction": touched / int(matrix.size),
        "max_abs_stored_delta": float(delta.max()) if delta.size else 0.0,
        "stored_update_frobenius": float(np.linalg.norm(candidate.astype(WORKING_DTYPE) - stored.astype(WORKING_DTYPE))),
        "columns": columns,
        "strength": strength,
    }
    return candidate, accounting


def _bind_arrays(base, patch: WeightCrackPatch, tensor_name: str | None):
    change = patch.changes[0]
    if not isinstance(change.tensor, str) or not change.tensor:
        raise EditRejected("tensor name is required")
    if isinstance(base, np.ndarray):
        if tensor_name is None:
            raise EditRejected("ndarray source requires an explicit tensor_name binding")
        if change.tensor != tensor_name:
            raise EditRejected("patch tensor does not match the bound matrix")
        matrix = _require_matrix(base)
        return {tensor_name: np.array(matrix, copy=True)}, tensor_name, None
    source = Path(base)
    if source.is_symlink():
        raise EditRejected("source symlinks are not accepted")
    if not source.is_file():
        raise EditRejected("source file is missing")
    if tensor_name is not None and tensor_name != change.tensor:
        raise EditRejected("tensor_name does not match the patch")
    with np.load(source) as loaded:
        keys = list(loaded.files)
        if change.tensor not in keys:
            raise EditRejected(f"tensor {change.tensor} is not in the source")
        arrays = {}
        for key in keys:
            value = loaded[key]
            if not isinstance(value, np.ndarray):
                raise EditRejected(f"unsupported source entry: {key}")
            arrays[key] = np.array(value, copy=True)
    _require_matrix(arrays[change.tensor])
    return arrays, change.tensor, source


def _normalized_patch(tensor_name: str, original: np.ndarray, columns: list[int], strength: float, matrix: np.ndarray) -> dict[str, Any]:
    return {
        "op": SUPPORTED_OPERATION,
        "tensor": tensor_name,
        "columns": columns,
        "strength": strength,
        "strength_f64_hex": f64_hex(np.asarray([strength], dtype=np.float64)),
        "direction_f64_hex": f64_hex(original),
        "working_dtype": "float64",
        "storage_dtype": str(matrix.dtype),
        "rounding": "cast_once_after_working_precision",
        "strength_range": "(0, 2]",
        "strength_above_one": "reverses the projected component",
    }


def construct_candidate(
    matrix: np.ndarray,
    patch: WeightCrackPatch,
    policy: WeightCrackPolicy,
    tensor_name: str,
):
    """Build the policy-admitted stored candidate. Trials and saves both use this."""
    matrix = _require_matrix(matrix)
    if len(patch.changes) != 1:
        raise EditRejected("this iteration supports exactly one change")
    change = patch.changes[0]
    if change.op != SUPPORTED_OPERATION:
        raise EditRejected(f"unsupported operation: {change.op}")
    if change.value is not None or change.rank is not None:
        raise EditRejected("project_remove does not accept value or rank")
    if not isinstance(patch.reason, str) or not patch.reason:
        raise EditRejected("reason is required")
    if not isinstance(patch.sector, str) or not patch.sector:
        raise EditRejected("sector is required")
    if not _sha256_text(patch.base_sha256):
        raise EditRejected("base_sha256 must be a lowercase SHA-256 hex digest")
    if change.tensor != tensor_name:
        raise EditRejected("patch tensor does not match the bound matrix")
    if change.direction is None or change.index is None or change.strength is None:
        raise EditRejected("project_remove requires direction, columns, and strength")
    gate_patch(patch, policy, int(matrix.size))
    if patch.base_sha256 != array_content_sha256(matrix):
        raise EditRejected("base hash mismatch")
    original, columns, strength = validate_project_remove(
        matrix, change.direction, change.index, change.strength
    )
    candidate, accounting = apply_project_remove(matrix, original, columns, strength)
    enforce_realized(accounting, policy, int(matrix.size))
    normalized = _normalized_patch(tensor_name, original, columns, strength, matrix)
    return candidate, accounting, normalized


def apply_patch(
    base,
    patch: WeightCrackPatch,
    policy: WeightCrackPolicy,
    output,
    tensor_name: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Apply one gated project_remove into a new bundle. Never replaces an existing path."""
    if task_id is not None and (not isinstance(task_id, str) or not task_id):
        raise EditRejected("task_id must be a non-empty string when supplied")
    output_path = Path(output)
    arrays, bound_name, source_path = _bind_arrays(base, patch, tensor_name)
    _check_destination(source_path, output_path)
    matrix = arrays[bound_name]
    candidate, accounting, normalized = construct_candidate(matrix, patch, policy, bound_name)
    published = dict(arrays)
    published[bound_name] = candidate
    source_manifest = tensor_manifest(bound_name, matrix)
    candidate_manifest = tensor_manifest(bound_name, candidate)
    other_manifests = [
        tensor_manifest(name, value) for name, value in arrays.items() if name != bound_name
    ]
    receipt: dict[str, Any] = {
        "schema": "weightcrack.receipt.v2",
        "status": "applied",
        "disposition": "applied",
        "behavioral_qualification": None,
        "task_id": task_id,
        "sector": patch.sector,
        "reason": patch.reason,
        "source_manifest": source_manifest,
        "candidate_manifest": candidate_manifest,
        "untouched_tensor_manifests": other_manifests,
        "base_sha256": source_manifest["content_sha256"],
        "candidate_content_sha256": candidate_manifest["content_sha256"],
        "source_file_sha256": file_sha256(source_path) if source_path is not None else None,
        "patch": model_payload(patch),
        "patch_sha256": model_sha256(patch),
        "normalized_patch": normalized,
        "normalized_patch_sha256": canonical_sha256(normalized),
        "policy": model_payload(policy),
        "policy_sha256": model_sha256(policy),
        "precision": {
            "working_dtype": "float64",
            "storage_dtype": str(matrix.dtype),
            "rounding": "cast_once_after_working_precision",
            "strength_range": "(0, 2]",
            "strength_above_one": "reverses the projected component",
        },
        "accounting": accounting,
    }
    event = {
        "schema": "weightcrack.event.v1",
        "status": "applied",
        "disposition": "applied",
        "behavioral_qualification": None,
        "task_id": task_id,
        "tensor": bound_name,
        "patch_sha256": receipt["patch_sha256"],
        "policy_sha256": receipt["policy_sha256"],
    }
    published_receipt = publish_bundle(output_path, published, receipt, event)
    published_receipt["bundle"] = str(output_path)
    return published_receipt
