"""Regression tests for projection application, budgets, and bundles."""

import numpy as np
import pytest
from pydantic import ValidationError

from weightcrack.bundle import read_completed_bundle, write_completion_manifest
from weightcrack.canonical import array_content_sha256, file_sha256
from weightcrack.errors import EditRejected, PolicyViolation, VerificationFailure
from weightcrack.patches.huggingface import apply_patch, apply_project_remove
from weightcrack.schema import PatchChange, WeightCrackPatch, WeightCrackPolicy
from weightcrack.verify import verify_bundle


def _policy(**overrides) -> WeightCrackPolicy:
    values = {
        "allow_tensors": ["W"],
        "max_entries": 64,
        "max_fraction": 1.0,
        "max_abs_delta": 10.0,
    }
    values.update(overrides)
    return WeightCrackPolicy(**values)


def _patch(matrix, columns, strength=1.0, tensor="W", op="project_remove", direction=None):
    if direction is None:
        direction = [1.0] * matrix.shape[0]
    return WeightCrackPatch(
        base_sha256=array_content_sha256(matrix),
        reason="test",
        sector="test",
        changes=[
            PatchChange(
                tensor=tensor,
                op=op,
                index=columns,
                direction=direction,
                strength=strength,
            )
        ],
    )


def _expected(matrix, columns, strength, direction):
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    working = np.array(matrix, dtype=np.float64, copy=True)
    for column in columns:
        projection = float(np.dot(unit, working[:, column]))
        working[:, column] = working[:, column] - strength * unit * projection
    return working.astype(matrix.dtype)


def test_extensionless_output_does_not_overwrite_source(tmp_path):
    matrix = np.ones((4, 4), dtype=np.float32)
    source = tmp_path / "base.npz"
    np.savez(source, W=matrix)
    before = source.read_bytes()
    with pytest.raises(EditRejected):
        apply_patch(source, _patch(matrix, [0, 1]), _policy(), tmp_path / "base", tensor_name="W")
    assert source.read_bytes() == before
    assert not (tmp_path / "base").exists()
    assert list(tmp_path.iterdir()) == [source]


def test_source_output_alias_is_rejected(tmp_path):
    matrix = np.ones((4, 4), dtype=np.float32)
    source = tmp_path / "base.npz"
    np.savez(source, W=matrix)
    before = file_sha256(source)
    with pytest.raises(EditRejected):
        apply_patch(source, _patch(matrix, [0]), _policy(), source, tensor_name="W")
    assert file_sha256(source) == before


def test_existing_bundle_is_rejected(tmp_path):
    matrix = np.eye(4, dtype=np.float32)
    bundle = tmp_path / "candidate.wcbundle"
    bundle.mkdir()
    with pytest.raises(EditRejected):
        apply_patch(matrix, _patch(matrix, [0]), _policy(), bundle, tensor_name="W")
    assert list(bundle.iterdir()) == []


def test_partial_write_leaves_no_completion_manifest(tmp_path, monkeypatch):
    matrix = np.eye(4, dtype=np.float32)
    kept = tmp_path / "keep.txt"
    kept.write_text("keep", encoding="utf-8")
    bundle = tmp_path / "candidate.wcbundle"

    def fail_manifest(path, manifest):
        raise RuntimeError("manifest failed")

    monkeypatch.setattr("weightcrack.bundle.write_completion_manifest", fail_manifest)
    with pytest.raises(RuntimeError, match="manifest failed"):
        apply_patch(matrix, _patch(matrix, [0]), _policy(), bundle, tensor_name="W")
    assert not bundle.exists()
    assert kept.read_text(encoding="utf-8") == "keep"
    assert write_completion_manifest.__name__ == "write_completion_manifest"


def test_reader_rejects_a_bundle_without_manifest(tmp_path):
    bundle = tmp_path / "torn.wcbundle"
    bundle.mkdir()
    (bundle / "candidate.npz").write_bytes(b"partial")
    with pytest.raises(VerificationFailure) as caught:
        read_completed_bundle(bundle)
    assert caught.value.status == "incomplete"


def test_coordinate_and_fraction_limits_count_distinct_coordinates(tmp_path):
    matrix = np.ones((4, 4), dtype=np.float64)
    direction = [1.0, 1.0, 1.0, 1.0]
    candidate, accounting = apply_project_remove(matrix, direction, [0, 1], 1.0)
    expected = _expected(matrix, [0, 1], 1.0, direction)
    assert np.array_equal(candidate, expected)
    assert accounting["operation_count"] == 1
    assert accounting["touched_coordinates"] == 8
    assert accounting["changed_coordinates"] == 8
    assert accounting["fraction_denominator"] == 16
    assert accounting["fraction_denominator_definition"] == "edited_tensor_scalar_count"
    assert accounting["max_abs_stored_delta"] == 1.0
    assert np.array_equal(candidate[:, 2:], matrix[:, 2:])

    def reject(policy, name):
        with pytest.raises(PolicyViolation):
            apply_patch(
                matrix,
                _patch(matrix, [0, 1], direction=direction),
                policy,
                tmp_path / name,
                tensor_name="W",
            )
        assert not (tmp_path / name).exists()

    with pytest.raises(PolicyViolation, match="max_entries"):
        apply_patch(
            matrix,
            _patch(matrix, [0, 1], direction=direction),
            _policy(max_entries=1),
            tmp_path / "entries.wcbundle",
            tensor_name="W",
        )
    reject(_policy(max_entries=8, max_fraction=0.0625), "fraction.wcbundle")
    reject(_policy(max_entries=8, max_fraction=1.0, max_abs_delta=0.001), "delta.wcbundle")
    assert not (tmp_path / "entries.wcbundle").exists()


def test_policy_rejection_publishes_nothing(tmp_path):
    matrix = np.ones((4, 4), dtype=np.float64)
    direction = [1.0, 1.0, 1.0, 1.0]
    bundle = tmp_path / "rejected.wcbundle"
    with pytest.raises(PolicyViolation):
        apply_patch(
            matrix,
            _patch(matrix, [0, 1], direction=direction),
            _policy(max_entries=1, max_fraction=0.0625, max_abs_delta=0.001),
            bundle,
            tensor_name="W",
        )
    assert not bundle.exists()


def test_nan_inf_columns_shape_and_unsupported_operations(tmp_path):
    matrix = np.eye(4, dtype=np.float32)
    bundle = tmp_path / "bad.wcbundle"
    with pytest.raises(EditRejected):
        apply_project_remove(matrix, [1.0, 0.0, 0.0, 0.0], [0], float("nan"))
    with pytest.raises(EditRejected):
        apply_project_remove(matrix, [1.0, 0.0, 0.0, 0.0], [0], float("inf"))
    with pytest.raises(EditRejected):
        apply_project_remove(np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32), [1.0, 0.0], [0], 1.0)
    with pytest.raises(EditRejected):
        apply_project_remove(matrix, [1.0, 0.0, 0.0, 0.0], [0, 0], 1.0)
    with pytest.raises(EditRejected):
        apply_project_remove(matrix, [1.0, 0.0, 0.0, 0.0], [4], 1.0)
    with pytest.raises(EditRejected):
        apply_project_remove(matrix, [1.0, 0.0], [0], 1.0)
    with pytest.raises(EditRejected):
        apply_project_remove(np.ones((2, 2), dtype=np.int32), [1.0, 0.0], [0], 1.0)
    with pytest.raises(EditRejected, match="unsupported operation"):
        apply_patch(matrix, _patch(matrix, [0], op="set"), _policy(), bundle, tensor_name="W")
    with pytest.raises(ValidationError):
        PatchChange(tensor="W", op="project_remove", index=[True], direction=[1.0], strength=1.0)
    with pytest.raises(ValidationError):
        PatchChange(tensor="W", op="project_remove", index=["0"], direction=[1.0], strength=1.0)
    with pytest.raises(ValidationError):
        PatchChange(tensor="W", op="project_remove", index=[1.0], direction=[1.0], strength=1.0)
    with pytest.raises(PolicyViolation):
        apply_patch(matrix, _patch(matrix, [0]), _policy(max_fraction=float("nan")), bundle, tensor_name="W")
    huge = np.full((2, 2), np.float32(3.0e38))
    with pytest.raises(EditRejected, match="nonfinite"):
        apply_project_remove(huge, [-0.6, 0.8], [0], 2.0)


def test_dtype_preservation_and_rounding_sensitive_limit(tmp_path):
    matrix = np.eye(4, dtype=np.float32)
    candidate, _ = apply_project_remove(matrix, [1.0, 0.0, 0.0, 0.0], [0], 0.5)
    assert candidate.dtype == np.float32
    wide = np.eye(4, dtype=np.float64)
    assert apply_project_remove(wide, [1.0, 0.0, 0.0, 0.0], [0], 0.5)[0].dtype == np.float64

    source = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gap = float(np.float32(1.0) - np.nextafter(np.float32(1.0), np.float32(0.0)))
    strength = gap * 0.51
    stored, accounting = apply_project_remove(source, [1.0, 0.0], [0], strength)
    working_delta = strength
    assert accounting["max_abs_stored_delta"] > working_delta
    midpoint = (accounting["max_abs_stored_delta"] + working_delta) / 2
    bundle = tmp_path / "round.wcbundle"
    with pytest.raises(PolicyViolation, match="max delta"):
        apply_patch(
            source,
            _patch(source, [0], strength=strength, direction=[1.0, 0.0]),
            _policy(max_entries=2, max_fraction=1.0, max_abs_delta=midpoint),
            bundle,
            tensor_name="W",
        )
    assert stored.dtype == np.float32
    assert not bundle.exists()


def test_tensor_identity_and_receipt_hash(tmp_path):
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    other = np.arange(3, dtype=np.int32)
    source = tmp_path / "source.npz"
    np.savez(source, weight=matrix)
    with pytest.raises(EditRejected, match="not in the source"):
        apply_patch(source, _patch(matrix, [0]), _policy(), tmp_path / "missing.wcbundle")
    with pytest.raises(EditRejected, match="explicit tensor_name"):
        apply_patch(matrix, _patch(matrix, [0]), _policy(), tmp_path / "anon.wcbundle")
    with pytest.raises(EditRejected, match="does not match"):
        apply_patch(matrix, _patch(matrix, [0], tensor="weight"), _policy(allow_tensors=["weight"]), tmp_path / "label.wcbundle", tensor_name="W")

    labeled = tmp_path / "labeled.npz"
    np.savez(labeled, W=matrix, other=other)
    before = file_sha256(labeled)
    bundle = tmp_path / "ok.wcbundle"
    receipt = apply_patch(labeled, _patch(matrix, [1], strength=0.5), _policy(), bundle, task_id="task-a")
    assert file_sha256(labeled) == before
    assert receipt["task_id"] == "task-a"
    assert receipt["disposition"] == "applied"
    assert receipt["behavioral_qualification"] is None
    assert receipt["candidate_file_sha256"] == file_sha256(bundle / "candidate.npz")
    loaded = np.load(bundle / "candidate.npz")
    assert loaded["W"].dtype == np.float32
    assert np.array_equal(loaded["other"], other)
    assert np.array_equal(loaded["W"][:, 0], matrix[:, 0])
    loaded.close()
    verdict = verify_bundle(bundle, labeled, "W", source_path=labeled)
    assert verdict["value_exact_reconstruction"] is True
    assert isinstance(verdict["resave_byte_identical"], bool)

    first = _patch(matrix, [1], strength=0.5)
    second = _patch(matrix, [1], strength=0.5)
    second = WeightCrackPatch(
        base_sha256="0" * 64,
        reason="test",
        sector="test",
        changes=first.changes,
    )
    assert first.compute_hash() != second.compute_hash()


def test_saved_artifact_tamper_and_reconstruction(tmp_path):
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    source = tmp_path / "source.npz"
    np.savez(source, W=matrix)
    bundle = tmp_path / "edit.wcbundle"
    apply_patch(source, _patch(matrix, [0, 1], strength=0.5), _policy(), bundle, tensor_name="W")
    verdict = verify_bundle(bundle, source, "W", source_path=source)
    assert verdict["value_exact_reconstruction"] is True
    payload = bytearray((bundle / "candidate.npz").read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    (bundle / "candidate.npz").write_bytes(payload)
    with pytest.raises(VerificationFailure):
        verify_bundle(bundle, source, "W", source_path=source)
