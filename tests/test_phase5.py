"""Phase 5: second sector and robustness check.

Confirms the same governed session works on a second sector
without relaxing policy, separation, or receipt invariants.
"""

import numpy as np
from pathlib import Path
import tempfile

from weightcrack.schema import WeightCrackPolicy
from weightcrack.agents.session import Session
from weightcrack.evaluation.suites import CODING_PRECISION_SUITE, all_suite_hashes


def test_second_sector_robustness():
    rng = np.random.default_rng(3)
    dim, cols = 32, 16
    matrix = rng.normal(size=(dim, cols)).astype(np.float32)
    fail = rng.normal(size=(10, dim)) + np.ones(dim) * 0.3
    success = rng.normal(size=(10, dim))

    policy = WeightCrackPolicy(allow_tensors=["mlp.down_proj"], max_entries=32, max_fraction=0.2)

    def score(m: np.ndarray) -> float:
        return -float(np.linalg.norm(m[:, 1]))

    session = Session(matrix, fail, success, policy, score)

    # Same tools, same gate path
    tools = session.tools()
    columns = tools["rank_columns"]["columns"][:2]
    trial = session.trial(columns, 0.4)
    assert "delta" in trial

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "coding_candidate.npz"
        result = session.propose_and_gate(columns, 0.4, out)
        assert result["status"] == "applied"
        assert len(session.ledger) == 1
        assert out.exists()
        # Receipt still present
        receipt_path = out.with_suffix(".receipt.json")
        assert receipt_path.exists()

    # Suite pins remain stable and distinct
    hashes = all_suite_hashes()
    assert hashes["math_loop"] != hashes["coding_precision"]
    assert len(hashes["coding_precision"]["holdout"]) == 64

    print("Phase 5 second sector and robustness: ok")
    return True


if __name__ == "__main__":
    test_second_sector_robustness()
