"""Agent-facing direction trial (scaffold).

Host executes a proposed direction on practice data only and returns the delta.
"""

from typing import Any


def try_direction(
    tensor: str,
    direction: list[float],
    strength: float,
    practice_suite: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply a temporary direction trial under the current policy budget
    and return practice score change. Does not write a permanent candidate.
    """
    return {
        "tensor": tensor,
        "strength": strength,
        "practice_delta": None,
        "status": "stub",
        "note": "host-executed trial pending full activation path",
    }
