"""The controller must use construct_candidate, including tensor_name.

A missing tensor_name used to be swallowed as an ordinary rejection.
"""

import numpy as np
import pytest

from weightcrack.agents.controller_proposal import propose_and_gate
from weightcrack.canonical import array_content_sha256
from weightcrack.errors import EditRejected
from weightcrack.patches.huggingface import construct_candidate
from weightcrack.schema import PatchChange, WeightCrackPatch, WeightCrackPolicy


def _matrix() -> np.ndarray:
    generator = np.random.default_rng(11)
    return generator.normal(size=(4, 8)).astype(np.float32)


def _policy(**overrides) -> WeightCrackPolicy:
    values = {
        "allow_tensors": ["W"],
        "max_entries": 16,
        "max_fraction": 0.5,
        "max_abs_delta": 2.0,
        "allowed_ops": ["project_remove"],
    }
    values.update(overrides)
    return WeightCrackPolicy(**values)


def _patch(matrix: np.ndarray, **overrides) -> WeightCrackPatch:
    change = {
        "tensor": "W",
        "op": "project_remove",
        "index": [1],
        "direction": [1.0, 0.0, 0.0, 0.0],
        "strength": 0.25,
    }
    change.update(overrides.pop("change", {}))
    values = {
        "base_sha256": array_content_sha256(matrix),
        "reason": "controller regression",
        "sector": "synthetic_classifier",
        "changes": [PatchChange(**change)],
    }
    values.update(overrides)
    return WeightCrackPatch(**values)


def test_valid_controller_proposal_is_admissible(tmp_path):
    matrix = _matrix()
    snapshot = np.array(matrix, copy=True)
    patch = _patch(matrix)
    policy = _policy()
    result = propose_and_gate(patch, policy, {"W": matrix})
    assert result["status"] == "admissible"
    candidate, accounting, _normalized = construct_candidate(snapshot, patch, policy, "W")
    assert result["candidate_content_sha256"] == array_content_sha256(candidate)
    assert result["accounting"] == accounting
    assert np.array_equal(matrix, snapshot)
    assert list(tmp_path.iterdir()) == []


def test_invalid_controller_proposals_are_rejected_without_mutation():
    matrix = _matrix()
    snapshot = np.array(matrix, copy=True)
    policy = _policy()

    empty = _patch(matrix, changes=[])
    assert propose_and_gate(empty, policy, {"W": matrix})["status"] == "rejected"

    missing = propose_and_gate(_patch(matrix), policy, {})
    assert missing["status"] == "rejected"
    assert "not supplied" in missing["reason"]

    stale = _patch(matrix, base_sha256="ab" * 32)
    assert "base hash" in propose_and_gate(stale, policy, {"W": matrix})["reason"]

    malformed = _patch(matrix, base_sha256="GG" * 32)
    assert propose_and_gate(malformed, policy, {"W": matrix})["status"] == "rejected"

    blank = _patch(matrix, base_sha256="")
    assert propose_and_gate(blank, policy, {"W": matrix})["status"] == "rejected"

    upper = _patch(matrix, base_sha256=("AB" * 32).lower().upper())
    assert propose_and_gate(upper, policy, {"W": matrix})["status"] == "rejected"

    incomplete = _patch(matrix, change={"direction": None, "index": None, "strength": None})
    assert propose_and_gate(incomplete, policy, {"W": matrix})["status"] == "rejected"

    wrong_tensor = _patch(matrix, change={"tensor": "B"})
    assert propose_and_gate(wrong_tensor, policy, {"B": matrix})["status"] == "rejected"

    wrong_op = _patch(matrix, change={"op": "set", "value": 1.0})
    assert propose_and_gate(wrong_op, policy, {"W": matrix})["status"] == "rejected"

    tiny_budget = _policy(max_entries=1)
    assert "max_entries" in propose_and_gate(_patch(matrix), tiny_budget, {"W": matrix})["reason"]

    tiny_delta = _policy(max_abs_delta=1e-8)
    assert "max delta" in propose_and_gate(_patch(matrix), tiny_delta, {"W": matrix})["reason"]

    assert np.array_equal(matrix, snapshot)


def test_programming_errors_are_not_reported_as_policy_failures(monkeypatch):
    matrix = _matrix()

    def explode(*_args, **_kwargs):
        raise RuntimeError("programming fault")

    monkeypatch.setattr(
        "weightcrack.agents.controller_proposal.construct_candidate",
        explode,
    )
    with pytest.raises(RuntimeError, match="programming fault"):
        propose_and_gate(_patch(matrix), _policy(), {"W": matrix})


def test_edit_rejected_stays_a_rejection():
    matrix = _matrix()
    result = propose_and_gate(
        _patch(matrix, change={"index": [50]}),
        _policy(),
        {"W": matrix},
    )
    assert result["status"] == "rejected"
    assert result["reason"]
    with pytest.raises(EditRejected):
        construct_candidate(matrix, _patch(matrix, change={"index": [50]}), _policy(), "W")
