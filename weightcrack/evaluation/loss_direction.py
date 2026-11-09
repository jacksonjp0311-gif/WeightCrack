"""Practice-loss direction for one projection comparison.

``G`` is the gradient of mean cross-entropy with respect to ``W``.
``M = (G @ W.T + W @ G.T) / 2``. The unit leading eigenvector ``u`` satisfies

    dL/ds at s=0 = -u.T @ M @ u

for the full-matrix step ``ΔW = -s * u u.T @ W``. The study still edits only
the two columns chosen by absolute projection. A missing unique leading
eigenvalue is ``NO_UNIQUE_POSITIVE_PROJECTION_DIRECTION``. That status is not
a fallback and it is not proof that no useful sparse direction exists.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

NO_UNIQUE = "NO_UNIQUE_POSITIVE_PROJECTION_DIRECTION"


def _as_f64(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _failure(reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"status": NO_UNIQUE, "reason": reason}
    if extra:
        report.update(extra)
    return report


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def mean_cross_entropy(
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
) -> float:
    _hidden, logits = forward_hidden_logits(A, b, W, c, x)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_prob = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    rows = np.arange(labels.shape[0])
    return -float(np.mean(log_prob[rows, labels]))


def forward_hidden_logits(
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(_as_f64(x) @ _as_f64(A).T + _as_f64(b))
    logits = hidden @ _as_f64(W).T + _as_f64(c)
    return hidden, logits


def practice_loss_gradient(
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Return ``G = (P - Y).T @ H / N`` or the no-unique status."""
    hidden, logits = forward_hidden_logits(A, b, W, c, x)
    count = int(hidden.shape[0])
    classes = int(logits.shape[1])
    label_array = np.asarray(labels)
    if label_array.shape != (count,) or not np.issubdtype(label_array.dtype, np.integer):
        return _failure("practice labels do not index the logits")
    if label_array.size and (int(label_array.min()) < 0 or int(label_array.max()) >= classes):
        return _failure("practice labels do not index the logits")
    probabilities = stable_softmax(logits)
    one_hot = np.zeros((count, classes), dtype=np.float64)
    if count:
        one_hot[np.arange(count), label_array.astype(np.int64, copy=False)] = 1.0
    gradient = (probabilities - one_hot).T @ hidden / float(count)
    intermediates = (hidden, logits, probabilities, one_hot, gradient)
    if any(not np.isfinite(array).all() for array in intermediates):
        return _failure("a practice-loss intermediate is nonfinite")
    return {
        "status": "ok",
        "gradient": gradient,
        "count": count,
        "classes": classes,
        "hidden": int(hidden.shape[1]),
    }


def orient_unit(vector: np.ndarray) -> np.ndarray:
    """Unit vector whose largest-magnitude coordinate is positive.

    Equal magnitudes keep the lowest index.
    """
    unit = np.array(vector, dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(unit))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("direction norm is zero or nonfinite")
    unit /= norm
    index = int(np.argmax(np.abs(unit)))
    if unit[index] < 0.0:
        unit = -unit
    return unit


def direction_from_gradient(gradient: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    """Leading eigenvector of the symmetrized gradient, or no unique direction."""
    gradient = _as_f64(gradient)
    weight = _as_f64(weight)
    if gradient.shape != weight.shape or gradient.ndim != 2 or gradient.shape[0] < 2:
        return _failure("gradient shape does not define a projection direction")
    matrix = (gradient @ weight.T + weight @ gradient.T) / 2.0
    frobenius = float(np.linalg.norm(matrix, ord="fro"))
    tolerance = 1e-12 * max(1.0, frobenius)
    base = {"tau": tolerance, "frobenius_norm_M": frobenius}
    if not np.isfinite(matrix).all() or not math.isfinite(frobenius) or not math.isfinite(tolerance):
        return _failure("projection matrix is nonfinite", base)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        return _failure("eigendecomposition is nonfinite", base)
    largest = float(eigenvalues[-1])
    second = float(eigenvalues[-2])
    gap = largest - second
    observed = {
        **base,
        "largest_eigenvalue": largest,
        "second_eigenvalue": second,
        "eigen_gap": gap,
        "eigenvalues": np.array(eigenvalues, dtype=np.float64, copy=True),
    }
    if not (largest > tolerance and gap > tolerance):
        return _failure("no unique eigenvalue above the tolerance", observed)
    try:
        direction = orient_unit(eigenvectors[:, -1])
    except ValueError as exc:
        return _failure(str(exc), observed)
    full_derivative = -float(direction @ matrix @ direction)
    if not math.isfinite(full_derivative):
        return _failure("full-matrix derivative is nonfinite", observed)
    return {
        "status": "ok",
        "direction": direction,
        "full_matrix_derivative": full_derivative,
        **observed,
    }


def loss_sensitive_projection_direction(
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Direction from the practice targets, their labels, and the damaged model."""
    parameters = (_as_f64(A), _as_f64(b), _as_f64(W), _as_f64(c), _as_f64(x))
    if any(not np.isfinite(array).all() for array in parameters):
        return _failure("a stored parameter or practice input is nonfinite")
    gradient_report = practice_loss_gradient(A, b, W, c, x, labels)
    if gradient_report["status"] != "ok":
        return gradient_report
    report = direction_from_gradient(gradient_report["gradient"], W)
    report["count"] = gradient_report["count"]
    report["gradient"] = gradient_report["gradient"]
    return report


def selected_support_derivative(
    direction: np.ndarray,
    gradient: np.ndarray,
    weight: np.ndarray,
    columns: list[int],
) -> float:
    """``-sum_c (u.T G[:,c]) (u.T W[:,c])`` for the columns the edit will touch.

    This is a diagnostic. Callers must not use it to pick a direction, a
    column, or another candidate.
    """
    unit = orient_unit(direction)
    gradient = _as_f64(gradient)
    weight = _as_f64(weight)
    total = 0.0
    for column in columns:
        total += float(np.dot(unit, gradient[:, column]) * np.dot(unit, weight[:, column]))
    if not math.isfinite(total):
        raise ValueError("selected-support derivative is nonfinite")
    return -total
