"""Regression tests for the three verifier gaps.

These checks use disposable bundles and disposable copies. They do not rewrite
``experiments/proxy_fault_v1``.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from weightcrack.bundle import json_bytes
from weightcrack.canonical import (
    array_content_sha256,
    canonical_sha256,
    file_sha256,
    model_sha256,
    tensor_manifest,
)
from weightcrack.errors import VerificationFailure
from weightcrack.evaluation.verify_proxy import verify_study
from weightcrack.patches.huggingface import apply_patch
from weightcrack.schema import WeightCrackPatch, WeightCrackPolicy
from weightcrack.verify import verify_bundle

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments" / "proxy_fault_v1"


def _policy(**overrides) -> WeightCrackPolicy:
    values = {
        "allow_tensors": ["W"],
        "max_entries": 8,
        "max_fraction": 1.0,
        "max_abs_delta": 2.0,
        "allowed_ops": ["project_remove"],
    }
    values.update(overrides)
    return WeightCrackPolicy(**values)


def _make_bundle(tmp_path: Path):
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    other = np.arange(3, dtype=np.int32)
    source = tmp_path / "source.npz"
    np.savez(source, W=matrix, other=other)
    bundle = tmp_path / "edit.wcbundle"
    patch = WeightCrackPatch(
        base_sha256=array_content_sha256(matrix),
        reason="practice column projection edit",
        sector="synthetic_classifier",
        evidence_ids=["fixture"],
        changes=[
            {
                "tensor": "W",
                "op": "project_remove",
                "index": [1],
                "direction": [1.0, 0.0, 0.0, 0.0],
                "strength": 0.5,
            }
        ],
    )
    apply_patch(source, patch, _policy(), bundle, tensor_name="W", task_id="fixture-00")
    return source, bundle, matrix


def _flip_creator_system(payload: bytes) -> bytes:
    data = bytearray(payload)
    signature = b"PK\x01\x02"
    start = 0
    flipped = 0
    while True:
        index = data.find(signature, start)
        if index < 0:
            break
        data[index + 5] ^= 3
        flipped += 1
        start = index + 4
    if flipped < 1:
        raise AssertionError("central directory headers were not found")
    return bytes(data)


def _retie(bundle: Path) -> None:
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


def test_creator_system_metadata_is_not_a_value_mismatch(tmp_path):
    source, bundle, matrix = _make_bundle(tmp_path)
    before = verify_bundle(bundle, source, "W", source_path=source)
    assert before["value_exact_reconstruction"] is True
    original = (bundle / "candidate.npz").read_bytes()
    flipped = _flip_creator_system(original)
    assert flipped != original
    assert len(flipped) == len(original)
    differences = [index for index, (left, right) in enumerate(zip(original, flipped)) if left != right]
    assert differences
    assert all(original[index] in (0, 3) and flipped[index] in (0, 3) for index in differences)
    (bundle / "candidate.npz").write_bytes(flipped)
    assert np.array_equal(_load_w(flipped), _load_w(original))
    receipt = json.loads((bundle / "receipt.json").read_text(encoding="utf-8"))
    receipt["candidate_file_sha256"] = file_sha256(bundle / "candidate.npz")
    receipt["candidate_nbytes"] = (bundle / "candidate.npz").stat().st_size
    (bundle / "receipt.json").write_bytes(json_bytes(receipt))
    _retie(bundle)
    verdict = verify_bundle(bundle, source, "W", source_path=source)
    assert verdict["value_exact_reconstruction"] is True
    assert verdict["resave_byte_identical"] is False
    assert verdict["accounting"] == before["accounting"]
    assert _load_w(flipped).shape == matrix.shape


def _load_w(payload: bytes) -> np.ndarray:
    import io

    with np.load(io.BytesIO(payload)) as handle:
        return np.array(handle["W"], copy=True)


def test_altered_archive_or_array_still_fails(tmp_path):
    source, bundle, _matrix = _make_bundle(tmp_path)
    intact = verify_bundle(bundle, source, "W", source_path=source)
    assert intact["value_exact_reconstruction"] is True

    raw = bytearray((bundle / "candidate.npz").read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (bundle / "candidate.npz").write_bytes(raw)
    with pytest.raises(VerificationFailure):
        verify_bundle(bundle, source, "W", source_path=source)

    content = tmp_path / "content"
    content.mkdir()
    source, bundle, _matrix = _make_bundle(content)
    with np.load(bundle / "candidate.npz") as handle:
        arrays = {key: np.array(handle[key], copy=True) for key in handle.files}
    arrays["W"] = np.array(arrays["W"], copy=True)
    arrays["W"][0, 0] = np.float32(arrays["W"][0, 0] + np.float32(1.0))
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    (bundle / "candidate.npz").write_bytes(buffer.getvalue())
    receipt = json.loads((bundle / "receipt.json").read_text(encoding="utf-8"))
    receipt["candidate_file_sha256"] = file_sha256(bundle / "candidate.npz")
    receipt["candidate_nbytes"] = (bundle / "candidate.npz").stat().st_size
    receipt["candidate_content_sha256"] = array_content_sha256(arrays["W"])
    receipt["candidate_manifest"] = tensor_manifest("W", arrays["W"])
    (bundle / "receipt.json").write_bytes(json_bytes(receipt))
    _retie(bundle)
    with pytest.raises(VerificationFailure):
        verify_bundle(bundle, source, "W", source_path=source)


def test_refreshed_checksums_do_not_admit_inconsistent_bundles(tmp_path):
    source, bundle, _matrix = _make_bundle(tmp_path)
    assert verify_bundle(bundle, source, "W", source_path=source)["value_exact_reconstruction"] is True

    def reject_after(mutate) -> None:
        trial = tmp_path / mutate.__name__
        cloned = trial / "edit.wcbundle"
        shutil.copytree(bundle, cloned)
        receipt = json.loads((cloned / "receipt.json").read_text(encoding="utf-8"))
        mutate(receipt)
        (cloned / "receipt.json").write_bytes(json_bytes(receipt))
        _retie(cloned)
        with pytest.raises(VerificationFailure):
            verify_bundle(cloned, source, "W", source_path=source)

    def allow_not_w(receipt):
        receipt["policy"]["allow_tensors"] = ["NOT_W"]

    def operation_set(receipt):
        receipt["patch"]["changes"][0]["op"] = "set"
        receipt["normalized_patch"]["op"] = "set"

    def zero_base(receipt):
        receipt["patch"]["base_sha256"] = "0" * 64

    def fabricated_manifest(receipt):
        receipt["source_manifest"]["shape"] = [1, 1]
        receipt["source_manifest"]["dtype"] = "float64"
        receipt["source_manifest"]["name"] = "NOT_W"
        receipt["candidate_manifest"]["content_sha256"] = "ab" * 32
        receipt["candidate_manifest"]["dtype"] = "float64"
        receipt["candidate_content_sha256"] = "ab" * 32

    reject_after(allow_not_w)
    reject_after(operation_set)
    reject_after(zero_base)
    reject_after(fabricated_manifest)


def _copy_archive(tmp_path: Path) -> Path:
    copied = tmp_path / "proxy_fault_v1"
    shutil.copytree(ARCHIVE, copied)
    return copied


def _write_report(path: Path, report: dict) -> None:
    path.write_bytes(json_bytes(report))


def test_archived_study_binds_scores_and_ignores_resave_platform(tmp_path):
    before = file_sha256(ARCHIVE / "experiment_report.json")
    copied = _copy_archive(tmp_path)
    verdict = verify_study(copied)
    assert verdict["status"] == "verified"
    assert verdict["study_status"] == "complete"
    assert verdict["applied_bundles"] == 16
    assert verdict["value_exact_reconstruction"] is True
    report = json.loads((copied / "experiment_report.json").read_text(encoding="utf-8"))
    assert report["heldout"]["methods"]["heuristic"]["behavioral_repair"] is False
    assert report["heldout"]["methods"]["random"] == {"status": "NO_QUALIFIED_CANDIDATE"}
    current = verdict["current_resave_byte_identical"]
    assert all(type(item) is bool for item in current)
    report["resave_byte_identical"] = [not item for item in current]
    _write_report(copied / "experiment_report.json", report)
    again = verify_study(copied)
    assert again["status"] == "verified"
    assert again["current_resave_byte_identical"] == current
    assert again["historical_resave_byte_identical"] != current
    assert file_sha256(ARCHIVE / "experiment_report.json") == before


def test_archived_report_rejects_demonstrated_inconsistencies(tmp_path):
    before = {
        path.relative_to(ARCHIVE).as_posix(): file_sha256(path)
        for path in ARCHIVE.rglob("*")
        if path.is_file()
    }
    copied = _copy_archive(tmp_path)
    original = json.loads((copied / "experiment_report.json").read_text(encoding="utf-8"))

    def expect_reject(mutate) -> None:
        report = json.loads(json.dumps(original))
        mutate(report)
        _write_report(copied / "experiment_report.json", report)
        with pytest.raises(VerificationFailure):
            verify_study(copied)

    def falsify_counts(report):
        scored = report["heldout"]["methods"]["heuristic"]
        scored["candidate_correct"] = 256
        scored["baseline_correct"] = 0
        scored["count"] = 999
        scored["candidate_content_sha256"] = "0" * 64
        scored["target_prediction_sha256"] = "0" * 64
        scored["preserve_prediction_sha256"] = "0" * 64

    def drop_finalist(report):
        report["heldout"]["methods"]["heuristic"] = {"status": "NO_QUALIFIED_CANDIDATE"}

    def zero_bundles(report):
        report["verified_applied_bundles"] = 0

    def exchange_arms(report):
        methods = report["heldout"]["methods"]
        methods["random"] = methods["heuristic"]
        methods["heuristic"] = {"status": "NO_QUALIFIED_CANDIDATE"}

    def truthy_repair(report):
        report["heldout"]["methods"]["heuristic"]["behavioral_repair"] = 0

    def zero_touched(report):
        report["heldout"]["methods"]["heuristic"]["accounting"]["touched_coordinates"] = 0

    def disagree_with_lock(report):
        report["selection"]["finalists"]["heuristic"]["bundle_name"] = "heuristic-00.wcbundle"

    expect_reject(falsify_counts)
    expect_reject(drop_finalist)
    expect_reject(zero_bundles)
    expect_reject(exchange_arms)
    expect_reject(truthy_repair)
    expect_reject(zero_touched)
    expect_reject(disagree_with_lock)
    after = {
        path.relative_to(ARCHIVE).as_posix(): file_sha256(path)
        for path in ARCHIVE.rglob("*")
        if path.is_file()
    }
    assert after == before
