"""Agent-facing direction trial.

Delegates to the shared construction path used by application.
"""

from typing import Any
import numpy as np

from .localization import direction_trial


def try_direction(
    matrix: np.ndarray,
    direction: list[float],
    columns: list[int],
    strength: float,
    practice_score_fn,
    policy=None,
) -> dict[str, Any]:
    return direction_trial(matrix, direction, columns, strength, practice_score_fn, policy)
