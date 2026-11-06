"""Phase 4 verification: agent loop under governance."""

import numpy as np
from pathlib import Path
import tempfile

from weightcrack.schema import WeightCrackPolicy
from weightcrack.agents.session import Session


def test_agent_loop():
    rng = np.random.default_rng(2)
    dim, cols = 32, 16
    matrix = rng.normal(size=(dim, cols)).astype(np.float32)
    fail = rng.normal(size=(10, dim)) + np.ones(dim) * 0.5
    success = rng.normal(size=(10, dim))

    policy = WeightCrackPolicy(allow_tensors=["mlp.down_proj"], max_entries=32, max_fraction=0.2)

    def score(m: np.ndarray) -> float:
        return -float(np.linalg.norm(m[:, 0]))

    session = Session(matrix, fail, success, policy, score)

    tools = session.tools()
    assert "activation_difference" in tools
    assert "rank_columns" in tools

    columns = tools["rank_columns"]["columns"][:2]
    trial = session.trial(columns, 0.5)
    assert "delta" in trial

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidate.npz"
        result = session.propose_and_gate(columns, 0.5, out)
        assert result["status"] == "applied"
        assert len(session.ledger) == 1
        assert out.exists()
    print("Phase 4 agent loop under governance: ok")
    return True


if __name__ == "__main__":
    test_agent_loop()
