"""Practice-only localization instruments.

Activation difference, column ranking, and direction trial.
All operate on caller-supplied practice statistics or matrices.
No holdout access.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def activation_difference(
    fail_activations: np.ndarray,
    success_activations: np.ndarray,
    top_k: int = 1,
) -> dict[str, Any]:
    """
    Compute the top singular vector of (mean_fail - mean_success).
    Returns a candidate direction u.
    """
    if fail_activations.ndim != 2 or success_activations.ndim != 2:
        raise ValueError("activations must be 2D (examples x dim)")
    delta = fail_activations.mean(axis=0) - success_activations.mean(axis=0)
    # SVD on the difference vector treated as a rank-1 matrix
    u = delta / (np.linalg.norm(delta) + 1e-12)
    return {
        "direction": u.tolist(),
        "norm": float(np.linalg.norm(delta)),
        "top_k": top_k,
    }


def rank_columns(
    matrix: np.ndarray,
    direction: list[float] | np.ndarray,
    top_k: int = 8,
) -> dict[str, Any]:
    """
    Rank columns by absolute projection onto the direction.
    Returns the top-k column indices and scores.
    """
    u = np.asarray(direction, dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    if matrix.shape[0] != u.shape[0]:
        raise ValueError("direction length must match matrix rows")
    scores = np.abs(matrix.T @ u)
    order = np.argsort(scores)[::-1][:top_k]
    return {
        "columns": order.tolist(),
        "scores": scores[order].tolist(),
    }


def direction_trial(
    matrix: np.ndarray,
    direction: list[float] | np.ndarray,
    columns: list[int],
    strength: float,
    practice_score_fn,
) -> dict[str, Any]:
    """
    Apply the removal formula temporarily, score practice, return the delta.
    Does not write a permanent candidate.
    """
    from ..patches.huggingface import _remove_direction
    edited = _remove_direction(matrix, direction, columns, strength)
    before = practice_score_fn(matrix)
    after = practice_score_fn(edited)
    return {
        "practice_before": before,
        "practice_after": after,
        "delta": after - before,
        "columns": columns,
        "strength": strength,
    }
