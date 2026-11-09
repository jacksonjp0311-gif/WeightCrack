"""Independent fixtures for the direction comparison.

These tests do not run the official 16-slot study.
"""

import json

import numpy as np

from weightcrack.evaluation.direction_experiment import (
    CONFIRMATION_SEED,
    QUESTION,
    confirmation_comparison,
    draw_confirmation_inputs,
    official_config,
    run,
)
from weightcrack.evaluation.direction_proposer import select_direction_finalists
from weightcrack.evaluation.loss_direction import (
    NO_UNIQUE,
    direction_from_gradient,
    loss_sensitive_projection_direction,
    mean_cross_entropy,
    orient_unit,
    practice_loss_gradient,
    selected_support_derivative,
)
from weightcrack.evaluation.proxy_experiment import (
    CONTROL_SEED,
    DOSES,
    STUDY_ID,
    official_config as v1_config,
)
from weightcrack.evaluation.proxy_model import predict
from weightcrack.evaluation.proxy_proposer import _select_method
from weightcrack.patches.huggingface import apply_project_remove
from weightcrack.schema import WeightCrackPolicy


def test_v2_protocol_does_not_redefine_v1():
    v1 = v1_config()
    v2 = official_config()
    assert STUDY_ID == "proxy_fault_v1"
    assert v1["study_id"] == "proxy_fault_v1"
    assert v1["control_seed"] == CONTROL_SEED == 1111
    assert "random_control" in v1
    assert v2["study_id"] == "proxy_direction_v2"
    assert v2["confirmation_seed"] == CONFIRMATION_SEED == 1112
    assert "control_seed" not in v2
    assert "random" not in v2["methods"]
    assert v2["methods"] == ["mean_logit_difference", "loss_sensitive_projection"]
    assert v2["question"] == QUESTION
    assert v2["doses"] == list(DOSES)
    assert v2["maximum_candidate_slots"] == 16
    assert v2["shapes"] == {"A": [16, 8], "b": [16], "W": [4, 16], "c": [4], "x": [8]}


def test_gradient_and_directional_derivative_match_finite_differences():
    generator = np.random.default_rng(20261109)
    hidden = 5
    classes = 4
    features = 3
    count = 7
    A = generator.standard_normal((hidden, features))
    b = generator.standard_normal(hidden)
    weight = generator.standard_normal((classes, hidden))
    bias = generator.standard_normal(classes)
    inputs = generator.standard_normal((count, features))
    labels = generator.integers(0, classes, size=count)
    gradient = practice_loss_gradient(A, b, weight, bias, inputs, labels)
    assert gradient["status"] == "ok"
    matrix = gradient["gradient"]
    step = 1e-5
    for row, column in ((0, 0), (3, 4), (1, 2)):
        plus = np.array(weight, copy=True)
        minus = np.array(weight, copy=True)
        plus[row, column] += step
        minus[row, column] -= step
        numeric = (
            mean_cross_entropy(A, b, plus, bias, inputs, labels)
            - mean_cross_entropy(A, b, minus, bias, inputs, labels)
        ) / (2 * step)
        assert abs(numeric - matrix[row, column]) < 1e-6
    direction = orient_unit(generator.standard_normal(classes))
    symmetric = (matrix @ weight.T + weight @ matrix.T) / 2.0
    analytic = -float(direction @ symmetric @ direction)
    delta = np.outer(direction, direction) @ weight
    forward = mean_cross_entropy(A, b, weight - step * delta, bias, inputs, labels)
    backward = mean_cross_entropy(A, b, weight + step * delta, bias, inputs, labels)
    assert abs((forward - backward) / (2 * step) - analytic) < 1e-6
    columns = list(range(weight.shape[1]))
    assert abs(selected_support_derivative(direction, matrix, weight, columns) - analytic) < 1e-10


def test_eigen_gate_and_sign():
    eye = np.eye(3)
    assert direction_from_gradient(eye, eye)["status"] == NO_UNIQUE
    assert direction_from_gradient(np.zeros((3, 3)), eye)["status"] == NO_UNIQUE
    leading = direction_from_gradient(np.diag([1.0, 2.0, 6.0]), eye)
    assert leading["status"] == "ok"
    assert np.allclose(leading["direction"], [0.0, 0.0, 1.0])
    assert abs(leading["full_matrix_derivative"] + leading["largest_eigenvalue"]) < 1e-10
    flipped = orient_unit(np.array([-0.8, 0.3]))
    assert flipped[0] > 0.0
    assert abs(flipped[0]) > abs(flipped[1])
    tied = orient_unit(np.array([-0.5, 0.5]))
    assert tied[0] > 0.0
    assert abs(abs(tied[0]) - abs(tied[1])) < 1e-12
    weight = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    weight[0, 0] = np.nan
    failed = loss_sensitive_projection_direction(
        np.eye(2), np.zeros(2), weight, np.zeros(3), np.zeros((2, 2)), np.array([0, 1])
    )
    assert failed["status"] == NO_UNIQUE


def test_confirmation_draw_and_comparison_rule():
    first_target, first_preserve = draw_confirmation_inputs()
    second_target, second_preserve = draw_confirmation_inputs()
    assert first_target.shape == (256, 8) and first_preserve.shape == (256, 8)
    assert first_target.dtype == np.float32 and first_preserve.dtype == np.float32
    assert np.array_equal(first_target, second_target)
    assert np.array_equal(first_preserve, second_preserve)
    assert not np.array_equal(first_target, first_preserve)
    loss = {"status": "qualified", "behavioral_repair": True, "corrections": 2}
    mean = {"status": "qualified", "behavioral_repair": True, "corrections": 2}
    methods = {"loss_sensitive_projection": loss, "mean_logit_difference": mean}
    assert confirmation_comparison(methods) == "NO_CONFIRMATION_WINNER"
    mean["corrections"] = 3
    assert confirmation_comparison(methods) == "MEAN_LOGIT_MORE_CORRECTIONS"
    mean["behavioral_repair"] = False
    assert confirmation_comparison(methods) == "LOSS_SENSITIVE_CONFIRMED"
    loss["behavioral_repair"] = False
    assert confirmation_comparison(methods) == "NEITHER_METHOD_CONFIRMED"
    methods = {
        "loss_sensitive_projection": {"status": "NO_QUALIFIED_CANDIDATE"},
        "mean_logit_difference": {"status": NO_UNIQUE},
    }
    assert confirmation_comparison(methods) == "NEITHER_METHOD_CONFIRMED"


def test_direction_run_refuses_protected_names(tmp_path):
    assert run(tmp_path / "proxy_fault_v1") == 1
    assert not (tmp_path / "proxy_fault_v1").exists()


def test_tiny_proposer_keeps_the_diagnostic_out_of_selection(tmp_path):
    generator = np.random.default_rng(0)
    intact_w = generator.standard_normal((3, 6)).astype(np.float32)
    damaged_w = apply_project_remove(intact_w, [1.0, 0.0, 0.0], [0, 1], 1.0)[0]
    model = {
        "A": generator.standard_normal((6, 4)).astype(np.float32),
        "b": generator.standard_normal(6).astype(np.float32),
        "W": damaged_w,
        "c": generator.standard_normal(3).astype(np.float32),
    }
    target = generator.standard_normal((12, 4)).astype(np.float32)
    preserve = generator.standard_normal((12, 4)).astype(np.float32)
    labels = {
        "y_practice_target": predict(model["A"], model["b"], intact_w, model["c"], target),
        "y_practice_preserve": predict(model["A"], model["b"], intact_w, model["c"], preserve),
    }
    practice = tmp_path / "practice_export.npz"
    source = tmp_path / "damaged_source.npz"
    np.savez(
        practice,
        A=model["A"],
        b=model["b"],
        W=model["W"],
        c=model["c"],
        x_practice_target=target,
        y_practice_target=labels["y_practice_target"],
        x_practice_preserve=preserve,
        y_practice_preserve=labels["y_practice_preserve"],
    )
    np.savez(source, A=model["A"], b=model["b"], W=model["W"], c=model["c"])
    policy = WeightCrackPolicy(
        allow_tensors=["W"],
        max_entries=6,
        max_fraction=1.0,
        max_abs_delta=2.0,
        allowed_ops=["project_remove"],
    )
    config = tmp_path / "public_config.json"
    config.write_text(
        json.dumps(
            {
                "study_id": "proxy_direction_v2",
                "methods": ["mean_logit_difference", "loss_sensitive_projection"],
                "tensor_name": "W",
                "sector": "synthetic_classifier",
                "shapes": {"W": [3, 6]},
                "population": {"practice_target": 12, "practice_preserve": 12},
                "doses": [0.05],
                "match_tolerance": 1e-3,
                "support_size": 2,
                "policy": policy.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    selection = select_direction_finalists(practice, config, source, candidates, deadline=10**12)
    assert set(selection["finalists"]) == {"mean_logit_difference", "loss_sensitive_projection"}
    assert "random" not in selection["methods"]
    for method, payload in selection["methods"].items():
        finalist = selection["finalists"][method]
        if payload["status"] != "ok":
            assert payload["slots"] == []
            assert finalist == {"status": payload["status"]}
            continue
        assert payload["diagnostic_affects_selection"] is False
        assert len(payload["columns"]) == 2
        expected = _select_method(payload["slots"])
        assert finalist["status"] == expected["status"]
        if expected["status"] == "qualified":
            assert finalist["bundle_name"] == expected["bundle_name"]
        assert len(payload["slots"]) == 1
