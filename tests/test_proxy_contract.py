"""Contract checks for the preregistered proxy study.

The full run must finish inside the wall-time ceiling. It does not have to
produce a behavioral repair.
"""

import inspect
import json

import numpy as np
import pytest

from weightcrack.canonical import array_content_sha256
from weightcrack.errors import EditRejected, VerificationFailure
from weightcrack.evaluation.proxy_experiment import (
    CONTROL_SEED,
    DATA_SEED,
    DOSES,
    FAULT_SEED,
    FAULT_STRENGTH,
    MATCH_TOLERANCE,
    MAX_ABS_DELTA,
    MAX_ENTRIES,
    MAX_FRACTION,
    MODEL_SEED,
    WALL_LIMIT_S,
    evaluate,
    run,
)
from weightcrack.evaluation.proxy_model import predict
from weightcrack.evaluation.proxy_proposer import PRACTICE_KEYS, select_finalists
from weightcrack.evaluation.verify_proxy import verify_study
from weightcrack.schema import WeightCrackPolicy


def test_frozen_constants():
    assert (MODEL_SEED, DATA_SEED, FAULT_SEED, CONTROL_SEED) == (1108, 1109, 1110, 1111)
    assert FAULT_STRENGTH == 1.0
    assert DOSES == (0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20)
    assert MATCH_TOLERANCE == 1e-3
    assert (MAX_ENTRIES, MAX_FRACTION, MAX_ABS_DELTA, WALL_LIMIT_S) == (8, 0.125, 2.0, 60.0)
    assert "secret" not in inspect.signature(select_finalists).parameters


def test_evaluate_does_not_open_the_secret_without_a_lock(tmp_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("secret opened")

    monkeypatch.setattr("weightcrack.evaluation.proxy_experiment.np.load", explode)
    with pytest.raises(VerificationFailure, match="proposal lock"):
        evaluate(tmp_path)


def test_proposer_rejects_extra_practice_keys_and_secret_paths(tmp_path):
    rng = np.random.default_rng(0)
    model = {
        "A": rng.standard_normal((16, 8)).astype(np.float32),
        "b": rng.standard_normal(16).astype(np.float32),
        "W": rng.standard_normal((4, 16)).astype(np.float32),
        "c": rng.standard_normal(4).astype(np.float32),
    }
    x = rng.standard_normal((4, 8)).astype(np.float32)
    y = predict(model["A"], model["b"], model["W"], model["c"], x)
    practice = tmp_path / "practice_export.npz"
    np.savez(practice, heldout=x, **model, x_practice_target=x, y_practice_target=y, x_practice_preserve=x, y_practice_preserve=y)
    config = tmp_path / "public_config.json"
    policy = WeightCrackPolicy(
        allow_tensors=["W"],
        max_entries=8,
        max_fraction=1.0,
        max_abs_delta=2.0,
        allowed_ops=["project_remove"],
    )
    config.write_text(
        json.dumps(
            {
                "study_id": "tiny",
                "tensor_name": "W",
                "sector": "synthetic_classifier",
                "shapes": {"W": [4, 16]},
                "population": {"practice_target": 4},
                "policy": policy.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "damaged_source.npz"
    np.savez(source, **model)
    with pytest.raises(EditRejected, match="allowlist"):
        select_finalists(practice, config, source, tmp_path / "candidates", deadline=10**12)
    secret = tmp_path / "host_secret.npz"
    np.savez(secret, **model)
    with pytest.raises(EditRejected, match="disallowed"):
        select_finalists(secret, config, source, tmp_path / "candidates", deadline=10**12)


def test_no_practice_failures_do_not_regenerate(tmp_path):
    rng = np.random.default_rng(1)
    model = {
        "A": rng.standard_normal((16, 8)).astype(np.float32),
        "b": rng.standard_normal(16).astype(np.float32),
        "W": rng.standard_normal((4, 16)).astype(np.float32),
        "c": rng.standard_normal(4).astype(np.float32),
    }
    x_target = rng.standard_normal((4, 8)).astype(np.float32)
    x_preserve = rng.standard_normal((4, 8)).astype(np.float32)
    y_target = predict(model["A"], model["b"], model["W"], model["c"], x_target)
    y_preserve = predict(model["A"], model["b"], model["W"], model["c"], x_preserve)
    practice = tmp_path / "practice_export.npz"
    payload = {
        "A": model["A"],
        "b": model["b"],
        "W": model["W"],
        "c": model["c"],
        "x_practice_target": x_target,
        "y_practice_target": y_target,
        "x_practice_preserve": x_preserve,
        "y_practice_preserve": y_preserve,
    }
    assert tuple(payload) == PRACTICE_KEYS
    np.savez(practice, **payload)
    source = tmp_path / "damaged_source.npz"
    np.savez(source, A=model["A"], b=model["b"], W=model["W"], c=model["c"])
    policy = WeightCrackPolicy(
        allow_tensors=["W"],
        max_entries=8,
        max_fraction=1.0,
        max_abs_delta=2.0,
        allowed_ops=["project_remove"],
    )
    config_path = tmp_path / "public_config.json"
    config_path.write_text(
        json.dumps(
            {
                "study_id": "tiny",
                "tensor_name": "W",
                "sector": "synthetic_classifier",
                "shapes": {"W": [4, 16]},
                "population": {"practice_target": 4, "practice_preserve": 4},
                "doses": [0.05],
                "match_tolerance": 1e-3,
                "control_seed": 1111,
                "support_size": 2,
                "policy": policy.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    selection = select_finalists(practice, config_path, source, candidates, deadline=10**12)
    assert selection["status"] == "NO_LOCALIZATION_SIGNAL"
    assert list(candidates.glob("*.wcbundle")) == []
    assert array_content_sha256(model["W"]) == selection["no_edit"]["content_sha256"]


def test_historical_directory_is_refused(tmp_path):
    assert run(tmp_path / "synthetic_v1") == 1
    assert not (tmp_path / "synthetic_v1").exists()


def test_official_proxy_run_replays(tmp_path):
    out = tmp_path / "proxy_fault_v1"
    assert run(out) == 0
    report = json.loads((out / "experiment_report.json").read_text(encoding="utf-8"))
    assert report["status"] in {"complete", "NO_LOCALIZATION_SIGNAL"}
    assert report["timing"]["exceeded"] is False
    assert report["heldout"]["used_for_selection"] is False
    verdict = verify_study(out)
    assert verdict["status"] == "verified"
    assert verdict["study_status"] == report["status"]
