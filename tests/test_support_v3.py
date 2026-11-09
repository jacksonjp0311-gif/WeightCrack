"""Support ranking and the v3 protocol boundary.

These tests use independent fixtures. They do not run the official study.
"""

import json
import shutil

import numpy as np
import pytest

from weightcrack.agents.localization import rank_columns
from weightcrack.evaluation.loss_direction import (
    mean_cross_entropy,
    practice_loss_gradient,
    selected_support_derivative,
)
from weightcrack.evaluation.proxy_experiment import (
    CONTROL_SEED,
    DATA_SEED,
    FAULT_SEED,
    MODEL_SEED,
    STUDY_ID,
)
from weightcrack.evaluation.direction_experiment import CONFIRMATION_SEED
from weightcrack.bundle import json_bytes
from weightcrack.canonical import canonical_sha256, file_sha256, model_sha256
from weightcrack.evaluation.proxy_proposer import _select_method
from weightcrack.evaluation.support_experiment import (
    PROTECTED_DIRECTORIES,
    confirmation_comparison,
    draw_confirmation_inputs,
    official_config,
    prepare,
    run,
)
from weightcrack.evaluation.support_proposer import _refuse_host_material
from weightcrack.evaluation.support_rank import (
    INSUFFICIENT_DESCENT_SUPPORT,
    column_products,
    rank_magnitude,
    rank_signed_contributions,
)
from weightcrack.evaluation.verify_support import verify_study
from weightcrack.errors import EditRejected, VerificationFailure
from weightcrack.schema import WeightCrackPatch, WeightCrackPolicy
from weightcrack.patches.huggingface import apply_project_remove


def test_v3_protocol_does_not_redefine_v1_or_publish_seeds():
    config = official_config()
    assert STUDY_ID == "proxy_fault_v1"
    assert MODEL_SEED == 1108 and DATA_SEED == 1109 and FAULT_SEED == 1110
    assert CONTROL_SEED == 1111 and CONFIRMATION_SEED == 1112
    assert config["study_id"] == "proxy_support_v3"
    assert config["methods"] == ["magnitude_projection", "signed_loss_contribution"]
    assert "model_seed" not in config
    assert "confirmation_seed" not in config
    assert config["support_size"] == 2
    assert config["maximum_candidate_slots"] == 16
    assert config["population"] == {
        "practice_target": 128,
        "practice_preserve": 128,
        "confirmation_target": 256,
        "confirmation_preserve": 256,
    }
    assert PROTECTED_DIRECTORIES >= {
        "proxy_fault_v1",
        "proxy_direction_v2",
        "proxy_direction_v2_harness_failure",
    }


def test_magnitude_tie_uses_ascending_index_and_not_argsort():
    weight = np.zeros((1, 4), dtype=np.float64)
    weight[0, :] = [3.0, 3.0, 5.0, 1.0]
    direction = np.array([1.0])
    ranked = rank_magnitude(weight, direction, support_size=2)
    assert ranked["columns"] == [2, 0]
    argsort_columns = rank_columns(weight, direction, top_k=2)["columns"]
    assert argsort_columns != ranked["columns"]


def test_signed_ranking_keeps_two_positive_columns():
    weight = np.zeros((4, 6), dtype=np.float64)
    gradient = np.zeros((4, 6), dtype=np.float64)
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    weight[0, :] = [1.0, 1.0, 1.0, 1.0, -1.0, 1.0]
    gradient[0, :] = [0.2, 0.5, 0.5, -0.4, 0.3, 1e-16]
    ranked = rank_signed_contributions(direction, gradient, weight, support_size=2)
    assert ranked["status"] == "ok"
    assert ranked["columns"] == [1, 2]
    assert len(ranked["columns"]) == 2
    products = column_products(direction, gradient, weight)
    assert products["contributions"][5] <= products["tau"]
    sparse = rank_signed_contributions(
        np.array([1.0]),
        np.array([[0.2, -1.0, -1.0, -1.0]]),
        np.array([[1.0, 1.0, 1.0, 1.0]]),
        support_size=2,
    )
    assert sparse["status"] == INSUFFICIENT_DESCENT_SUPPORT
    assert sparse["columns"] == []


def test_positive_contribution_is_descent_and_matches_finite_difference():
    generator = np.random.default_rng(20261110)
    hidden = 5
    classes = 3
    features = 4
    count = 9
    hidden_matrix = generator.standard_normal((hidden, features))
    bias = generator.standard_normal(hidden)
    weight = generator.standard_normal((classes, hidden))
    out_bias = generator.standard_normal(classes)
    inputs = generator.standard_normal((count, features))
    labels = generator.integers(0, classes, size=count)
    gradient = practice_loss_gradient(hidden_matrix, bias, weight, out_bias, inputs, labels)
    assert gradient["status"] == "ok"
    matrix = gradient["gradient"]
    direction = generator.standard_normal(classes)
    direction = direction / np.linalg.norm(direction)
    columns = [1, 4]
    products = column_products(direction, matrix, weight)
    derivative = selected_support_derivative(direction, matrix, weight, columns)
    summed = -float(sum(products["contributions"][column] for column in columns))
    assert abs(derivative - summed) < 1e-12
    step = 1e-6

    def shifted(sign: float) -> np.ndarray:
        edited = np.array(weight, dtype=np.float64, copy=True)
        for column in columns:
            projection = float(np.dot(direction, edited[:, column]))
            edited[:, column] = edited[:, column] - sign * step * direction * projection
        return edited

    numeric = (
        mean_cross_entropy(hidden_matrix, bias, shifted(1.0), out_bias, inputs, labels)
        - mean_cross_entropy(hidden_matrix, bias, shifted(-1.0), out_bias, inputs, labels)
    ) / (2.0 * step)
    assert abs(numeric - derivative) < 1e-7
    # A positive a_j contributes a negative term to dL/ds.
    one = [int(np.argmax(products["contributions"]))]
    if float(products["contributions"][one[0]]) > 0.0:
        assert selected_support_derivative(direction, matrix, weight, one) < 0.0


def test_prepare_refuses_archives_and_omits_candidates(tmp_path):
    blocked = tmp_path / "proxy_direction_v2"
    assert prepare(blocked, tmp_path / "missing-seeds.json") == 1
    assert not blocked.exists()
    seeds = tmp_path / "seeds.json"
    seeds.write_text(
        json.dumps(
            {
                "model_seed": 31,
                "data_seed": 32,
                "fault_seed": 33,
                "confirmation_seed": 34,
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "case"
    assert prepare(destination, seeds) == 0
    assert not (destination / "candidates" / "selection.json").exists()
    assert (destination / "public_commitment.json").is_file()
    assert (destination / "host_only" / "confirmation_bank.npz").is_file()
    config = json.loads((destination / "proposal_inputs" / "public_config.json").read_text(encoding="utf-8"))
    assert "model_seed" not in config
    first, _second = draw_confirmation_inputs(34)
    again, _again_preserve = draw_confirmation_inputs(34)
    assert np.array_equal(first, again)
    assert first.shape == (256, 8) and first.dtype == np.float32
    try:
        _refuse_host_material(destination / "host_only" / "confirmation_bank.npz")
    except EditRejected:
        refused = True
    else:
        refused = False
    assert refused
    assert confirmation_comparison(
        {
            "magnitude_projection": {"status": "NO_QUALIFIED_CANDIDATE"},
            "signed_loss_contribution": {"status": "NO_QUALIFIED_CANDIDATE"},
        }
    ) == "NEITHER_METHOD_CONFIRMED"


def test_two_columns_stay_inside_the_coordinate_budget():
    weight = np.ones((4, 16), dtype=np.float32)
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    edited, accounting = apply_project_remove(weight, direction.tolist(), [0, 3], 0.5)
    assert accounting["touched_coordinates"] == 8
    assert accounting["touched_fraction"] == 0.125
    assert edited.shape == weight.shape
    assert edited.dtype == weight.dtype


def test_diagnostics_do_not_change_the_practice_winner():
    slots = []
    for index, corrections, norm in ((0, 1, 0.2), (1, 2, 0.5), (2, 2, 0.1)):
        slots.append(
            {
                "status": "applied",
                "behaviorally_qualified_on_practice": True,
                "dose_index": index,
                "bundle_name": f"m-{index:02d}.wcbundle",
                "candidate_content_sha256": "a" * 64,
                "candidate_file_sha256": "b" * 64,
                "stored_update_frobenius": norm,
                "strength": 0.1,
                "practice": {"corrections": corrections, "losses": 0, "preservation_flips": 0},
                "accounting": {"stored_update_frobenius": norm},
                "sign_status": "agree",
                "predicted_loss_change": -0.01,
            }
        )
    winner = _select_method(slots)
    assert winner["bundle_name"] == "m-02.wcbundle"
    mutated = json.loads(json.dumps(slots))
    mutated[0]["sign_status"] = "disagree"
    mutated[0]["predicted_loss_change"] = -100.0
    mutated[2]["diagnostic_affects_selection"] = True
    assert _select_method(mutated)["bundle_name"] == winner["bundle_name"]


def _retie(bundle) -> None:
    receipt = json.loads((bundle / "receipt.json").read_text(encoding="utf-8"))
    patch = WeightCrackPatch.model_validate(receipt["patch"])
    policy = WeightCrackPolicy.model_validate(receipt["policy"])
    receipt["patch_sha256"] = model_sha256(patch)
    receipt["policy_sha256"] = model_sha256(policy)
    receipt["normalized_patch_sha256"] = canonical_sha256(receipt["normalized_patch"])
    event = json.loads((bundle / "events.jsonl").read_text(encoding="utf-8"))
    event["patch_sha256"] = receipt["patch_sha256"]
    event["policy_sha256"] = receipt["policy_sha256"]
    event["candidate_file_sha256"] = receipt["candidate_file_sha256"]
    (bundle / "receipt.json").write_bytes(json_bytes(receipt))
    (bundle / "events.jsonl").write_bytes(json_bytes(event))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for name in ("candidate.npz", "receipt.json", "events.jsonl"):
        path = bundle / name
        manifest["files"][name] = {"sha256": file_sha256(path), "nbytes": path.stat().st_size}
    (bundle / "manifest.json").write_bytes(json_bytes(manifest))


def _write_json(path, payload) -> None:
    path.write_bytes(json_bytes(payload))


def test_verifier_rejects_tampered_disposable_studies(tmp_path):
    """A non-official seed set exercises the study verifier. It is not the preregistered case."""
    seeds = tmp_path / "seeds.json"
    seeds.write_text(
        json.dumps(
            {
                "model_seed": 41,
                "data_seed": 42,
                "fault_seed": 43,
                "confirmation_seed": 44,
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "disposable"
    assert prepare(destination, seeds) == 0
    assert run(destination) == 0
    assert verify_study(destination)["status"] == "verified"

    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(VerificationFailure, match="experiment report is missing"):
        verify_study(missing)

    premature = tmp_path / "premature"
    shutil.copytree(destination, premature)
    rows = [
        json.loads(line)
        for line in (premature / "study_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_stage = {row["stage"]: row for row in rows}
    reordered = [
        by_stage["host_committed"],
        by_stage["proposer_returned"],
        by_stage["confirmation_opened"],
        by_stage["lock_written"],
        by_stage["report_written"],
    ]
    (premature / "study_events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reordered),
        encoding="utf-8",
    )
    with pytest.raises(VerificationFailure, match="confirmation was opened before the practice lock"):
        verify_study(premature)

    contradicted = tmp_path / "contradicted"
    shutil.copytree(destination, contradicted)
    report = json.loads((contradicted / "experiment_report.json").read_text(encoding="utf-8"))
    lock = json.loads((contradicted / "proposal_lock.json").read_text(encoding="utf-8"))
    report["confirmation"]["no_edit"]["corrections"] = 9
    report["selection"]["finalists"]["magnitude_projection"] = {"status": "NO_QUALIFIED_CANDIDATE"}
    lock["selection"] = report["selection"]
    _write_json(contradicted / "experiment_report.json", report)
    _write_json(contradicted / "proposal_lock.json", lock)
    with pytest.raises(VerificationFailure):
        verify_study(contradicted)

    changed = tmp_path / "changed-bytes"
    shutil.copytree(destination, changed)
    bundles = list((changed / "candidates").glob("*.wcbundle"))
    assert bundles
    raw = bytearray((bundles[0] / "candidate.npz").read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (bundles[0] / "candidate.npz").write_bytes(raw)
    with pytest.raises(VerificationFailure):
        verify_study(changed)

    policy = tmp_path / "altered-policy"
    shutil.copytree(destination, policy)
    policy_bundle = next((policy / "candidates").glob("*.wcbundle"))
    receipt = json.loads((policy_bundle / "receipt.json").read_text(encoding="utf-8"))
    receipt["policy"]["allow_tensors"] = ["NOT_W"]
    (policy_bundle / "receipt.json").write_bytes(json_bytes(receipt))
    _retie(policy_bundle)
    with pytest.raises(VerificationFailure):
        verify_study(policy)

    absent = tmp_path / "missing-bundle"
    shutil.copytree(destination, absent)
    removed = next((absent / "candidates").glob("*.wcbundle"))
    shutil.rmtree(removed)
    with pytest.raises(VerificationFailure, match="missing candidate bundle"):
        verify_study(absent)
