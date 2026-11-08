"""Phase 2 checks localization instruments and the shared trial path."""

import numpy as np
import pytest

from weightcrack.agents.direction_trial import try_direction
from weightcrack.agents.localization import activation_difference, direction_trial, rank_columns
from weightcrack.errors import EditRejected
from weightcrack.schema import WeightCrackPolicy


def _policy() -> WeightCrackPolicy:
    return WeightCrackPolicy(
        allow_tensors=["W"],
        max_entries=64,
        max_fraction=0.125,
        max_abs_delta=10.0,
        allowed_ops=["project_remove"],
    )


def test_localization_instruments():
    rng = np.random.default_rng(1)
    dim = 32
    fail = rng.normal(size=(20, dim)) + np.array([1.0] + [0.0] * (dim - 1))
    success = rng.normal(size=(20, dim))
    matrix = rng.normal(size=(dim, 16)).astype(np.float64)

    diff = activation_difference(fail, success, top_k=1)
    again = activation_difference(fail, success, top_k=4)
    assert diff["status"] == "ok"
    assert diff["top_k_affects_direction"] is False
    assert diff["components"] == 1
    assert diff["direction"] == again["direction"]
    assert len(diff["direction"]) == dim

    ranked = rank_columns(matrix, diff["direction"], top_k=4)
    assert len(ranked["columns"]) == 4

    def dummy_score(candidate):
        return -float(np.linalg.norm(candidate[:, 0]))

    columns = ranked["columns"][:2]
    trial = direction_trial(matrix, diff["direction"], columns, 0.5, dummy_score, _policy(), "W")
    same = try_direction(matrix, diff["direction"], columns, 0.5, dummy_score, _policy(), "W")
    assert np.array_equal(trial["candidate"], same["candidate"])
    assert trial["delta"] == same["delta"]
    assert trial["accounting"]["touched_coordinates"] == dim * 2
    with pytest.raises(TypeError):
        direction_trial(matrix, diff["direction"], columns, 0.5, dummy_score)
    with pytest.raises(EditRejected):
        direction_trial(matrix, diff["direction"], columns, 0.5, lambda _matrix: float("nan"), _policy(), "W")
