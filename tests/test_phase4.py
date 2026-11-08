"""Phase 4 checks the host-owned session gate.

This is a governed apply/trial path. It does not measure a behavioral repair.
"""

import numpy as np

from weightcrack.agents.session import Session, SessionConfig
from weightcrack.schema import WeightCrackPolicy


def _session(tmp_path, max_entries, max_fraction, max_abs_delta, sector):
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(32, 16)).astype(np.float32)
    fail = rng.normal(size=(10, 32)) + 0.5
    success = rng.normal(size=(10, 32))
    policy = WeightCrackPolicy(
        allow_tensors=["mlp.down_proj"],
        max_entries=max_entries,
        max_fraction=max_fraction,
        max_abs_delta=max_abs_delta,
        allowed_ops=["project_remove"],
    )
    config = SessionConfig(
        task_id="phase4",
        tensor_name="mlp.down_proj",
        sector=sector,
        policy=policy,
        output_root=tmp_path,
    )

    def score(candidate: np.ndarray) -> float:
        return -float(np.linalg.norm(candidate[:, 0]))

    return Session(matrix, fail, success, config, score)


def test_agent_loop_uses_host_config(tmp_path):
    session = _session(tmp_path, max_entries=64, max_fraction=0.125, max_abs_delta=100.0, sector="practice_probe")
    tools = session.tools()
    columns = tools["rank_columns"]["columns"][:2]
    trial = session.trial(columns, 0.5)
    result = session.propose(columns, 0.5)
    assert result["disposition"] == "applied"
    assert result["receipt"]["behavioral_qualification"] is None
    assert result["receipt"]["task_id"] == "phase4"
    assert result["receipt"]["sector"] == "practice_probe"
    loaded = np.load(tmp_path / "phase4-0001.wcbundle" / "candidate.npz")["mlp.down_proj"]
    assert np.array_equal(loaded, trial["candidate"])
    assert session.practice_score_fn(loaded) == trial["practice_after"]
    assert "propose_and_gate" not in dir(session)


def test_underbudgeted_multcolumn_edit_is_rejected(tmp_path):
    # Two columns of a 32-row matrix touch 64 coordinates, so max_entries=32 rejects.
    session = _session(tmp_path, max_entries=32, max_fraction=0.2, max_abs_delta=100.0, sector="practice_probe")
    columns = session.tools()["rank_columns"]["columns"][:2]
    result = session.propose(columns, 0.5)
    assert result["disposition"] == "rejected"
    assert list(tmp_path.glob("*.wcbundle")) == []
