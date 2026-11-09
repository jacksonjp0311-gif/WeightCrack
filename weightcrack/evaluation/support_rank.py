"""Two ways to choose two columns for one loss-sensitive direction.

``a_j = (u.T G[:,j]) * (u.T W[:,j])``. The selected-support derivative is
``-sum_j a_j``. A positive ``a_j`` is a descent contribution at strength zero.
Magnitude ranking uses ``|u.T W[:,j]|``. Both rankings break ties by ascending
column index. They do not build a new direction and they do not add doses.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .loss_direction import orient_unit, selected_support_derivative

INSUFFICIENT_DESCENT_SUPPORT = "INSUFFICIENT_DESCENT_SUPPORT"
CONTRIBUTION_TOLERANCE_SCALE = 1e-12
LOSS_SIGN_TOLERANCE = 1e-12
SUPPORT_SIZE = 2


def _as_f64(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def contribution_tolerance(contributions: np.ndarray) -> float:
    """``1e-12 * max(1, max |a_j|)``. Fixed before any official trial."""
    peak = float(np.max(np.abs(contributions))) if contributions.size else 0.0
    if not math.isfinite(peak):
        raise ValueError("column contributions are nonfinite")
    return CONTRIBUTION_TOLERANCE_SCALE * max(1.0, peak)


def column_products(
    direction: np.ndarray,
    gradient: np.ndarray,
    weight: np.ndarray,
) -> dict[str, Any]:
    """Per-column magnitude and signed contribution for one unit direction."""
    unit = orient_unit(direction)
    gradient = _as_f64(gradient)
    weight = _as_f64(weight)
    if gradient.shape != weight.shape or gradient.ndim != 2:
        raise ValueError("gradient and weight shapes disagree")
    if unit.shape != (weight.shape[0],):
        raise ValueError("direction length must match matrix rows")
    weight_projection = weight.T @ unit
    gradient_projection = gradient.T @ unit
    contributions = gradient_projection * weight_projection
    magnitudes = np.abs(weight_projection)
    if not np.isfinite(contributions).all() or not np.isfinite(magnitudes).all():
        raise ValueError("column contributions are nonfinite")
    tolerance = contribution_tolerance(contributions)
    return {
        "contributions": contributions,
        "magnitudes": magnitudes,
        "weight_projection": weight_projection,
        "gradient_projection": gradient_projection,
        "tau": tolerance,
    }


def _descending_then_index(scores: np.ndarray) -> list[int]:
    return sorted(range(int(scores.shape[0])), key=lambda index: (-float(scores[index]), index))


def rank_magnitude(
    weight: np.ndarray,
    direction: np.ndarray,
    support_size: int = SUPPORT_SIZE,
) -> dict[str, Any]:
    """Largest ``|u.T W[:,j]|``, then the smallest column index."""
    if type(support_size) is not int or support_size < 1:
        raise ValueError("support size must be a positive integer")
    unit = orient_unit(direction)
    weight = _as_f64(weight)
    if unit.shape != (weight.shape[0],):
        raise ValueError("direction length must match matrix rows")
    magnitudes = np.abs(weight.T @ unit)
    if not np.isfinite(magnitudes).all():
        raise ValueError("projection magnitudes are nonfinite")
    if int(magnitudes.shape[0]) < support_size:
        raise ValueError("matrix does not have the requested support")
    order = _descending_then_index(magnitudes)
    columns = order[:support_size]
    return {
        "status": "ok",
        "columns": columns,
        "scores": [float(magnitudes[column]) for column in columns],
        "magnitudes": magnitudes,
    }


def rank_signed_contributions(
    direction: np.ndarray,
    gradient: np.ndarray,
    weight: np.ndarray,
    support_size: int = SUPPORT_SIZE,
) -> dict[str, Any]:
    """Exactly ``support_size`` columns with the largest strictly positive ``a_j``.

    Ties keep the smallest column index. Columns above the tolerance that are
    not in that prefix are not edited. Fewer than ``support_size`` columns
    above the tolerance is ``INSUFFICIENT_DESCENT_SUPPORT``. There is no
    substitute ranking and the support is not expanded.
    """
    if type(support_size) is not int or support_size < 1:
        raise ValueError("support size must be a positive integer")
    products = column_products(direction, gradient, weight)
    contributions = products["contributions"]
    tolerance = float(products["tau"])
    eligible = [
        index
        for index, value in enumerate(contributions.tolist())
        if float(value) > tolerance
    ]
    eligible.sort(key=lambda index: (-float(contributions[index]), index))
    report = {
        "tau": tolerance,
        "positive_count": len(eligible),
        "contributions": contributions,
        "magnitudes": products["magnitudes"],
    }
    if len(eligible) < support_size:
        report["status"] = INSUFFICIENT_DESCENT_SUPPORT
        report["columns"] = []
        report["selected_contributions"] = []
        return report
    columns = eligible[:support_size]
    report["status"] = "ok"
    report["columns"] = columns
    report["selected_contributions"] = [float(contributions[column]) for column in columns]
    return report


def loss_sign(value: float, tolerance: float = LOSS_SIGN_TOLERANCE) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("loss change is not a real number")
    number = float(value)
    if not math.isfinite(number):
        return "nonfinite"
    if abs(number) <= tolerance:
        return "zero"
    if number < 0.0:
        return "negative"
    return "positive"


def sign_agreement(predicted: float, actual: float) -> dict[str, Any]:
    """Whether the first-order prediction matches the measured loss change.

    A zero or nonfinite side is indeterminate. The result is not a selection rule.
    """
    predicted_sign = loss_sign(predicted)
    actual_sign = loss_sign(actual)
    indeterminate = predicted_sign in {"zero", "nonfinite"} or actual_sign in {"zero", "nonfinite"}
    agrees = None if indeterminate else predicted_sign == actual_sign
    status = "indeterminate" if indeterminate else ("agree" if agrees else "disagree")
    return {
        "predicted_sign": predicted_sign,
        "actual_sign": actual_sign,
        "sign_agreement": agrees,
        "sign_status": status,
        "diagnostic_affects_selection": False,
    }


def predicted_support_change(
    direction: np.ndarray,
    gradient: np.ndarray,
    weight: np.ndarray,
    columns: list[int],
    strength: float,
) -> dict[str, Any]:
    """First-order change ``(dL/ds) * s`` on the columns the edit touches."""
    derivative = selected_support_derivative(direction, gradient, weight, columns)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError("strength is not a real number")
    change = derivative * float(strength)
    if not math.isfinite(change):
        raise ValueError("predicted loss change is nonfinite")
    return {
        "selected_support_derivative": derivative,
        "predicted_loss_change": change,
        "diagnostic_affects_selection": False,
    }
