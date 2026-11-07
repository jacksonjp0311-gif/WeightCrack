"""Regression tests for projection application and accounting."""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from weightcrack.schema import WeightCrackPatch, WeightCrackPolicy, PatchChange
from weightcrack.patches.huggingface import apply_patch, apply_project_remove, EditRejected
from weightcrack.policy import enforce_realized, PolicyViolation


def _patch(matrix, columns, strength=0.5, tensor="W"):
    h = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
    return WeightCrackPatch(
        base_sha256=h,
        reason="test",
        sector="test",
        changes=[
            PatchChange(
                tensor=tensor,
                op="project_remove",
                index=columns,
                direction=np.ones(matrix.shape[0]).tolist(),
                strength=strength,
            )
        ],
    )


def test_extensionless_output_does_not_overwrite_source():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(4, 4)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "base.npz"
        np.savez(src, weight=matrix)
        out = Path(tmp) / "base"  # extensionless
        policy = WeightCrackPolicy(allow_tensors=["W"], max_entries=16, max_fraction=1.0, max_abs_delta=10.0)
        with pytest.raises(EditRejected):
            apply_patch(src, _patch(matrix, [0]), policy, out)
        # source unchanged
        loaded = np.load(src)["weight"]
        assert np.allclose(loaded, matrix)


def test_coordinate_count_not_operation_count():
    matrix = np.eye(4, dtype=np.float32)
    candidate, acc = apply_project_remove(matrix, np.ones(4), [0, 1], 0.5)
    assert acc["operation_count"] == 1
    assert acc["touched_coordinates"] == 8  # 4 rows x 2 columns
    assert acc["changed_coordinates"] > 0


def test_nan_strength_rejected():
    matrix = np.eye(4, dtype=np.float32)
    with pytest.raises(EditRejected):
        apply_project_remove(matrix, np.ones(4), [0], float("nan"))


def test_dtype_preserved():
    matrix = np.eye(4, dtype=np.float32)
    candidate, _ = apply_project_remove(matrix, np.ones(4), [0], 0.5)
    assert candidate.dtype == np.float32


def test_realized_delta_enforced():
    matrix = np.eye(4, dtype=np.float32)
    _, acc = apply_project_remove(matrix, np.ones(4), [0], 1.0)
    policy = WeightCrackPolicy(allow_tensors=["W"], max_entries=16, max_fraction=1.0, max_abs_delta=0.001)
    with pytest.raises(PolicyViolation):
        enforce_realized(acc, policy, matrix.size)


def test_receipt_contains_candidate_hash():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(4, 4)).astype(np.float32)
    policy = WeightCrackPolicy(allow_tensors=["W"], max_entries=16, max_fraction=1.0, max_abs_delta=10.0)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cand.npz"
        receipt = apply_patch(matrix, _patch(matrix, [1]), policy, out)
        assert "candidate_sha256" in receipt
        assert "candidate_content_hash" in receipt
        assert receipt["accounting"]["touched_coordinates"] == 4
