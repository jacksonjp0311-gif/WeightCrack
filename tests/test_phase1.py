"""Phase 1 checks the projection formula and the stored candidate."""

import numpy as np

from weightcrack.canonical import array_content_sha256
from weightcrack.patches.huggingface import apply_patch
from weightcrack.schema import PatchChange, WeightCrackPatch, WeightCrackPolicy


def test_direction_removal_matches_the_formula(tmp_path):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(32, 16)).astype(np.float32)
    direction = rng.normal(size=32)
    columns = [2, 5, 9]
    strength = 0.8
    patch = WeightCrackPatch(
        base_sha256=array_content_sha256(matrix),
        reason="phase1 verification",
        sector="math",
        changes=[
            PatchChange(
                tensor="mlp.down_proj",
                op="project_remove",
                index=columns,
                direction=direction.tolist(),
                strength=strength,
            )
        ],
    )
    # Three columns touch 96 coordinates. The old max_entries=64 budget was too small.
    policy = WeightCrackPolicy(
        allow_tensors=["mlp.down_proj"],
        max_entries=96,
        max_fraction=0.2,
        max_abs_delta=10.0,
    )
    bundle = tmp_path / "candidate.wcbundle"
    receipt = apply_patch(matrix, patch, policy, bundle, tensor_name="mlp.down_proj", task_id="phase1")
    assert receipt["disposition"] == "applied"
    assert receipt["behavioral_qualification"] is None
    assert receipt["accounting"]["operation_count"] == 1
    assert receipt["accounting"]["touched_coordinates"] == 96
    assert receipt["accounting"]["fraction_denominator"] == matrix.size

    loaded = np.load(bundle / "candidate.npz")["mlp.down_proj"]
    unit = direction / np.linalg.norm(direction)
    expected = np.array(matrix, dtype=np.float64, copy=True)
    for column in columns:
        projection = float(np.dot(unit, expected[:, column]))
        expected[:, column] = expected[:, column] - strength * unit * projection
    expected = expected.astype(np.float32)
    assert np.array_equal(loaded, expected)
    assert np.count_nonzero(loaded != matrix) == receipt["accounting"]["changed_coordinates"]
    untouched = [column for column in range(matrix.shape[1]) if column not in columns]
    assert np.array_equal(loaded[:, untouched], matrix[:, untouched])
    assert loaded.dtype == np.float32
