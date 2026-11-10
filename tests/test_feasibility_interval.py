"""Analytic final-layer intervals against dense argmax sampling."""

from pathlib import Path

import numpy as np
import pytest

import weightcrack.evaluation.feasibility_replay as replay

from weightcrack.evaluation.feasibility_replay import (
    behavior_holds,
    column_gamma,
    contains_strength,
    feasible_strength_intervals,
    grid_disagreements,
    predict_shifted,
    reconstruct_stored_candidate,
    select_interior_strength,
    separate_grid_findings,
    apply_policy_constraints,
)
from weightcrack.schema import WeightCrackPolicy


def _wide_case():
    direction = np.array([1.0, 0.0])
    logits = np.array(
        [
            [2.0, 0.0],
            [0.0, 0.5],
        ],
        dtype=np.float64,
    )
    gamma = np.array([1.0, -1.0])
    teacher = np.array([0, 0])
    baseline = np.array([0, 1])
    return logits, gamma, direction, teacher, baseline


def test_wide_interval_matches_argmax_away_from_the_boundary():
    logits, gamma, direction, teacher, baseline = _wide_case()
    solved = feasible_strength_intervals(logits, gamma, direction, teacher, baseline)
    assert solved["status"] == "ok"
    assert solved["exact_for"] == "final_layer_affine_logits"
    assert solved["intervals"] == [
        {"lo": 0.5, "hi": 2.0, "lo_closed": True, "hi_closed": True}
    ]
    assert contains_strength(0.5, solved["intervals"])
    assert contains_strength(2.0, solved["intervals"])
    assert not contains_strength(0.5 - 1e-9, solved["intervals"])
    assert grid_disagreements(logits, gamma, direction, teacher, baseline, solved["intervals"]) == []
    inside = predict_shifted(logits, gamma, direction, 1.25)
    outside = predict_shifted(logits, gamma, direction, 0.25)
    assert behavior_holds(inside, teacher, baseline)
    assert not behavior_holds(outside, teacher, baseline)
    assert select_interior_strength(solved["intervals"]) == 1.25


def test_point_interval_and_protected_examples():
    direction = np.array([1.0, 0.0])
    logits = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    gamma = np.array([1.0, -1.0])
    teacher = np.array([0, 0])
    baseline = np.array([0, 1])
    solved = feasible_strength_intervals(logits, gamma, direction, teacher, baseline)
    assert solved["intervals"] == [
        {"lo": 1.0, "hi": 1.0, "lo_closed": True, "hi_closed": True}
    ]
    assert select_interior_strength(solved["intervals"]) == 1.0
    at_point = predict_shifted(logits, gamma, direction, 1.0)
    assert behavior_holds(at_point, teacher, baseline)
    protected = feasible_strength_intervals(
        logits,
        gamma,
        direction,
        teacher,
        baseline,
        protection="baseline_predictions",
        protected_mask=np.array([False, True]),
    )
    assert protected["status"] == "empty"
    assert protected["intervals"] == []


def test_policy_and_storage_reconstruction_use_the_shared_gate():
    generator = np.random.default_rng(20261110)
    weight = generator.normal(size=(4, 6)).astype(np.float32)
    hidden = np.tanh(generator.normal(size=(5, 6))).astype(np.float64)
    direction = generator.normal(size=4)
    direction = direction / np.linalg.norm(direction)
    columns = [1, 4]
    logits = hidden @ weight.astype(np.float64).T
    unit, gamma = column_gamma(direction, columns, weight, hidden)
    teacher = np.argmax(logits, axis=1)
    baseline = np.array(teacher, copy=True)
    baseline[0] = (int(baseline[0]) + 1) % 4
    solved = feasible_strength_intervals(logits, gamma, unit, teacher, baseline)
    policy = WeightCrackPolicy(
        allow_tensors=["W"],
        max_entries=4,
        max_fraction=0.01,
        max_abs_delta=2.0,
        allowed_ops=["project_remove"],
    )
    blocked = apply_policy_constraints(
        solved["intervals"] or [{"lo": 0.0, "hi": 2.0, "lo_closed": False, "hi_closed": True}],
        weight,
        unit,
        columns,
        policy,
    )
    assert blocked["status"] == "empty"
    loose = WeightCrackPolicy(
        allow_tensors=["W"],
        max_entries=32,
        max_fraction=1.0,
        max_abs_delta=2.0,
        allowed_ops=["project_remove"],
    )
    admitted = apply_policy_constraints(
        [{"lo": 0.0, "hi": 2.0, "lo_closed": False, "hi_closed": True}],
        weight,
        unit,
        columns,
        loose,
    )
    assert admitted["status"] == "ok"
    strength = select_interior_strength(admitted["intervals"])
    assert strength is not None
    snapshot = np.array(weight, copy=True)
    candidate, accounting = reconstruct_stored_candidate(weight, unit, columns, strength)
    assert np.array_equal(weight, snapshot)
    assert candidate.dtype == weight.dtype
    assert accounting["touched_coordinates"] == 8
    assert candidate.shape == weight.shape


def test_replay_cli_takes_one_fresh_output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        replay,
        "run",
        lambda: {
            "feasible": 0,
            "matched": 0,
            "reconstructed": 0,
            "grid_disagreement_count": 0,
            "policy_exclusion_count": 0,
        },
    )
    assert replay.main(["weightcrack.evaluation.feasibility_replay"]) == 1
    output = tmp_path / "feasibility_interval_check"
    assert replay.main(["weightcrack.evaluation.feasibility_replay", str(output)]) == 0
    assert (output / "report.json").is_file()
    entry = (Path(replay.__file__).read_text(encoding="utf-8"))
    assert "raise SystemExit(main(sys.argv))" in entry


def test_policy_cap_is_not_an_analytic_disagreement():
    logits, gamma, direction, teacher, baseline = _wide_case()
    solved = feasible_strength_intervals(logits, gamma, direction, teacher, baseline)
    clipped = [{"lo": 0.5, "hi": 1.0, "lo_closed": True, "hi_closed": True}]
    split = separate_grid_findings(
        logits,
        gamma,
        direction,
        teacher,
        baseline,
        solved["intervals"],
        clipped,
    )
    assert split["behavior_disagreements"] == []
    assert split["policy_exclusions"]
    assert all(strength > 1.0 for strength in split["policy_exclusions"])


def test_source_mutation_during_reconstruction_is_not_silently_accepted(monkeypatch):
    """A self-comparison of the caller's array cannot detect an in-place edit."""
    weight = np.zeros((4, 6), dtype=np.float32)
    weight[0, 1] = np.float32(1.0)
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    original = replay.construct_candidate

    def mutate_after_gate(matrix, patch, policy, tensor_name):
        result = original(matrix, patch, policy, tensor_name)
        matrix[0, 0] = np.float32(9.0)
        return result

    monkeypatch.setattr(replay, "construct_candidate", mutate_after_gate)
    with pytest.raises(RuntimeError, match="source reconstruction changed"):
        reconstruct_stored_candidate(weight, direction, [1], 0.25)
