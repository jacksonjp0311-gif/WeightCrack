"""Phase 5 checks that a second host sector uses the same gate.

The suite-hash assertions pin frozen example text. They do not measure coding
improvement, and a successful file write is not a behavioral pass.
"""

import numpy as np

from weightcrack.agents.session import Session, SessionConfig
from weightcrack.evaluation.suites import all_suite_hashes
from weightcrack.schema import WeightCrackPolicy


def test_second_sector_uses_the_same_gate(tmp_path):
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(32, 16)).astype(np.float32)
    fail = rng.normal(size=(10, 32)) + 0.3
    success = rng.normal(size=(10, 32))
    policy = WeightCrackPolicy(
        allow_tensors=["mlp.down_proj"],
        max_entries=64,
        max_fraction=0.125,
        max_abs_delta=100.0,
        allowed_ops=["project_remove"],
    )
    config = SessionConfig(
        task_id="phase5",
        tensor_name="mlp.down_proj",
        sector="coding_precision",
        policy=policy,
        output_root=tmp_path,
    )
    session = Session(matrix, fail, success, config, lambda candidate: -float(np.linalg.norm(candidate[:, 1])))
    columns = session.tools()["rank_columns"]["columns"][:2]
    trial = session.trial(columns, 0.4)
    result = session.propose(columns, 0.4)
    assert result["disposition"] == "applied"
    assert result["receipt"]["behavioral_qualification"] is None
    assert result["receipt"]["sector"] == "coding_precision"
    loaded = np.load(tmp_path / "phase5-0001.wcbundle" / "candidate.npz")["mlp.down_proj"]
    assert np.array_equal(loaded, trial["candidate"])

    hashes = all_suite_hashes()
    assert hashes["math_loop"] != hashes["coding_precision"]
    assert len(hashes["coding_precision"]["holdout"]) == 64
