"""Deterministic classifier used by the proxy fault study.

The stored parameters stay in their storage dtype. Logits are computed in
float64. Predictions are ``argmax`` of::

    f(x) = W tanh(Ax + b) + c

The localization site is those logits, named ``post_linear_logits``. It is the
output space of ``W`` (one coordinate per row), not the hidden tanh state.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..agents.localization import activation_difference

ACTIVATION_SITE = "post_linear_logits"


def forward_logits(
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    hidden = np.tanh(np.asarray(x, dtype=np.float64) @ np.asarray(A, dtype=np.float64).T + np.asarray(b, dtype=np.float64))
    return hidden @ np.asarray(W, dtype=np.float64).T + np.asarray(c, dtype=np.float64)


def predict(A: np.ndarray, b: np.ndarray, W: np.ndarray, c: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.argmax(forward_logits(A, b, W, c, x), axis=1).astype(np.int64, copy=False)


def prediction_sha256(predictions: np.ndarray) -> str:
    array = np.ascontiguousarray(predictions, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def split_counts(baseline: np.ndarray, candidate: np.ndarray, teacher: np.ndarray) -> dict[str, int]:
    """Corrections and losses against the intact teacher's labels."""
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    teacher = np.asarray(teacher)
    if baseline.shape != teacher.shape or candidate.shape != teacher.shape:
        raise ValueError("prediction populations do not match")
    base_ok = baseline == teacher
    new_ok = candidate == teacher
    return {
        "corrections": int(np.count_nonzero(~base_ok & new_ok)),
        "losses": int(np.count_nonzero(base_ok & ~new_ok)),
        "baseline_correct": int(np.count_nonzero(base_ok)),
        "candidate_correct": int(np.count_nonzero(new_ok)),
        "count": int(teacher.shape[0]),
    }


def preservation_flips(baseline: np.ndarray, candidate: np.ndarray) -> int:
    """Prediction changes relative to the damaged model, not the teacher."""
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    if baseline.shape != candidate.shape:
        raise ValueError("preservation populations do not match")
    return int(np.count_nonzero(baseline != candidate))


def derive_direction(
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    c: np.ndarray,
    x_practice_target: np.ndarray,
    y_practice_target: np.ndarray,
) -> dict[str, Any]:
    """Normalized mean logit difference on practice targets.

    ``top_k`` does not enter the vector. An empty failure set, an empty
    success set, or a zero direction is ``NO_LOCALIZATION_SIGNAL``.
    """
    logits = forward_logits(A, b, W, c, x_practice_target)
    if logits.ndim != 2 or int(logits.shape[1]) != int(W.shape[0]):
        raise ValueError("post_linear_logits width does not match W rows")
    labels = np.asarray(y_practice_target, dtype=np.int64)
    predicted = np.argmax(logits, axis=1).astype(np.int64, copy=False)
    incorrect = predicted != labels
    correct = ~incorrect
    report: dict[str, Any] = {
        "activation_site": ACTIVATION_SITE,
        "definition": (
            "normalized mean damaged post_linear_logits on incorrect practice "
            "targets minus the mean on correct practice targets"
        ),
        "dimension": int(logits.shape[1]),
        "matrix_rows": int(W.shape[0]),
        "matrix_columns": int(W.shape[1]),
        "n_incorrect": int(np.count_nonzero(incorrect)),
        "n_correct": int(np.count_nonzero(correct)),
        "top_k_affects_direction": False,
        "components": 1,
    }
    if report["n_incorrect"] == 0 or report["n_correct"] == 0:
        report["status"] = "NO_LOCALIZATION_SIGNAL"
        report["reason"] = (
            "no incorrect practice targets"
            if report["n_incorrect"] == 0
            else "no correct practice targets"
        )
        return report
    diff = activation_difference(logits[incorrect], logits[correct], top_k=1)
    alt = activation_difference(logits[incorrect], logits[correct], top_k=3)
    if not np.array_equal(np.asarray(diff["direction"]), np.asarray(alt["direction"])):
        raise ValueError("top_k changed the activation direction")
    report["norm"] = diff["norm"]
    report["status"] = diff["status"]
    if diff["status"] != "ok":
        report["reason"] = "direction norm is zero or nonfinite"
        return report
    report["direction"] = [float(value) for value in diff["direction"]]
    return report
