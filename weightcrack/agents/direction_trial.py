"""Agent-facing direction trial.

This is the same construction and policy gate as ``localization.direction_trial``.
"""

from typing import Any

import numpy as np

from ..schema import WeightCrackPolicy
from .localization import direction_trial


def try_direction(
    matrix: np.ndarray,
    direction: list[float],
    columns: list[int],
    strength: float,
    practice_score_fn,
    policy: WeightCrackPolicy,
    tensor_name: str,
) -> dict[str, Any]:
    return direction_trial(
        matrix,
        direction,
        columns,
        strength,
        practice_score_fn,
        policy,
        tensor_name,
    )
