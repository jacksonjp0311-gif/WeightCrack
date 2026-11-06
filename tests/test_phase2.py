"""Phase 2 verification for localization instruments."""

import numpy as np
from weightcrack.agents.localization import (
    activation_difference,
    rank_columns,
    direction_trial,
)


def test_localization_instruments():
    rng = np.random.default_rng(1)
    dim = 32
    fail = rng.normal(size=(20, dim)) + np.array([1.0] + [0.0] * (dim - 1))
    success = rng.normal(size=(20, dim))
    matrix = rng.normal(size=(dim, 16))

    diff = activation_difference(fail, success)
    assert len(diff["direction"]) == dim
    assert diff["norm"] > 0

    ranked = rank_columns(matrix, diff["direction"], top_k=4)
    assert len(ranked["columns"]) == 4

    def dummy_score(m):
        # Higher score if first column is smaller in L2 (toy)
        return -float(np.linalg.norm(m[:, 0]))

    trial = direction_trial(
        matrix, diff["direction"], ranked["columns"][:2], 0.5, dummy_score
    )
    assert "delta" in trial
    print("Phase 2 localization instruments: ok")
    return True


if __name__ == "__main__":
    test_localization_instruments()
