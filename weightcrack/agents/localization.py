"""Practice-only localization instruments.

The activation direction is the normalized mean difference between two
practice activation sets. ``top_k`` is recorded and does not change that
vector. Column ranking scores absolute projection onto one direction. A large
score maximizes removal magnitude at a fixed strength; it does not show that
the column caused an error.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import EditRejected
from ..patches.huggingface import construct_candidate
from ..policy import PolicyViolation
from ..schema import PatchChange, WeightCrackPatch, WeightCrackPolicy
from ..canonical import array_content_sha256


def _finite_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise EditRejected(f"invalid practice score ({label})")
    number = float(value)
    if not math.isfinite(number):
        raise EditRejected(f"invalid practice score ({label})")
    return number


def activation_difference(
    fail_activations: np.ndarray,
    success_activations: np.ndarray,
    top_k: int = 1,
) -> dict[str, Any]:
    """Normalized mean(fail) - mean(success). One vector, not a spectral basis."""
    if fail_activations.ndim != 2 or success_activations.ndim != 2:
        raise ValueError("activations must be 2D (examples x dim)")
    if fail_activations.shape[1] != success_activations.shape[1]:
        raise ValueError("activation widths must match")
    if fail_activations.shape[0] < 1 or success_activations.shape[0] < 1:
        raise ValueError("activation sets must be nonempty")
    delta = fail_activations.mean(axis=0) - success_activations.mean(axis=0)
    norm = float(np.linalg.norm(delta))
    report = {
        "top_k": top_k,
        "top_k_affects_direction": False,
        "components": 1,
        "norm": norm,
    }
    if not math.isfinite(norm) or norm < 1e-12:
        report["direction"] = np.zeros(delta.shape, dtype=np.float64).tolist()
        report["status"] = "NO_LOCALIZATION_SIGNAL"
        return report
    report["direction"] = (delta / norm).tolist()
    report["status"] = "ok"
    return report


def rank_columns(
    matrix: np.ndarray,
    direction: list[float] | np.ndarray,
    top_k: int = 8,
) -> dict[str, Any]:
    """Rank columns by absolute projection. This is not a causal attribution."""
    unit = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("direction has zero or invalid norm")
    unit = unit / norm
    if matrix.ndim != 2 or matrix.shape[0] != unit.shape[0]:
        raise ValueError("direction length must match matrix rows")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    scores = np.abs(matrix.astype(np.float64).T @ unit)
    count = min(top_k, int(scores.shape[0]))
    order = np.argsort(scores)[::-1][:count]
    return {
        "columns": [int(column) for column in order.tolist()],
        "scores": [float(scores[column]) for column in order],
    }


def direction_trial(
    matrix: np.ndarray,
    direction: list[float] | np.ndarray,
    columns: list[int],
    strength: float,
    practice_score_fn,
    policy: WeightCrackPolicy,
    tensor_name: str,
) -> dict[str, Any]:
    """Score the same policy-admitted stored candidate that application would save."""
    if not isinstance(policy, WeightCrackPolicy):
        raise EditRejected("a host policy is required")
    if not isinstance(tensor_name, str) or not tensor_name:
        raise EditRejected("tensor_name is required")
    if isinstance(direction, np.ndarray):
        direction_list = np.asarray(direction, dtype=np.float64).tolist()
    else:
        direction_list = direction
    parsed_columns = []
    for column in columns:
        if isinstance(column, np.integer) and type(column) is not np.bool_:
            column = int(column)
        if type(column) is not int:
            raise EditRejected("columns must be integers")
        parsed_columns.append(column)
    patch = WeightCrackPatch(
        base_sha256=array_content_sha256(matrix),
        reason="direction trial",
        sector="trial",
        changes=[
            PatchChange(
                tensor=tensor_name,
                op="project_remove",
                index=parsed_columns,
                direction=direction_list,
                strength=strength,
            )
        ],
    )
    candidate, accounting, _normalized = construct_candidate(matrix, patch, policy, tensor_name)
    before = _finite_score(practice_score_fn(matrix), "before")
    after = _finite_score(practice_score_fn(candidate), "after")
    return {
        "practice_before": before,
        "practice_after": after,
        "delta": after - before,
        "columns": parsed_columns,
        "strength": float(strength),
        "accounting": accounting,
        "candidate": candidate,
        "content_sha256": array_content_sha256(candidate),
        "admitted": True,
    }


def practice_direction_search(
    matrix: np.ndarray,
    fail_activations: np.ndarray,
    success_activations: np.ndarray,
    practice_score_fn,
    policy: WeightCrackPolicy,
    tensor_name: str,
    strength: float = 0.5,
    max_directions: int = 4,
) -> dict[str, Any]:
    """Practice-only search retained from the earlier branch. It is not the proxy study."""
    candidates = []
    diff = activation_difference(fail_activations, success_activations)
    if diff["status"] != "ok":
        return {"status": "NO_DIRECTION"}
    candidates.append(np.array(diff["direction"]))
    if fail_activations.shape[0] >= 2:
        centered = fail_activations - fail_activations.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        for row in vh[: max(0, max_directions - 1)]:
            candidates.append(row)

    best = None
    for unit in candidates:
        try:
            ranked = rank_columns(matrix, unit, top_k=2)
            trial = direction_trial(
                matrix,
                unit,
                ranked["columns"],
                strength,
                practice_score_fn,
                policy,
                tensor_name,
            )
        except (EditRejected, PolicyViolation, ValueError):
            continue
        if best is None or trial["delta"] > best["delta"]:
            best = {
                "direction": np.asarray(unit, dtype=np.float64).tolist(),
                "columns": ranked["columns"],
                "delta": trial["delta"],
                "trial": {key: value for key, value in trial.items() if key != "candidate"},
                "status": "ok",
            }
    return best or {"status": "NO_DIRECTION"}
