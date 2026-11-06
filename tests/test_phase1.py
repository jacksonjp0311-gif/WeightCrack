"""Minimal Phase 1 verification for direction removal."""

import hashlib
import numpy as np
from pathlib import Path
import tempfile

from weightcrack.schema import WeightCrackPatch, WeightCrackPolicy, PatchChange
from weightcrack.patches.huggingface import apply_patch


def test_direction_removal_roundtrip():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(32, 16)).astype(np.float32)
    base_hash = hashlib.sha256(matrix.tobytes()).hexdigest()

    direction = rng.normal(size=32).tolist()
    columns = [2, 5, 9]
    strength = 0.8

    patch = WeightCrackPatch(
        base_sha256=base_hash,
        reason="phase1 verification",
        sector="math",
        changes=[
            PatchChange(
                tensor="mlp.down_proj",
                op="project_remove",
                index=columns,
                direction=direction,
                strength=strength,
            )
        ],
    )
    policy = WeightCrackPolicy(allow_tensors=["mlp.down_proj"], max_entries=64, max_fraction=0.1)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidate.npz"
        receipt = apply_patch(matrix, patch, policy, out)
        assert receipt["status"] == "applied"
        assert receipt["changed_entries"] == 1
        loaded = np.load(out)["weight"]
        assert loaded.shape == matrix.shape
        # Confirm the math: at least one column changed
        assert not np.allclose(loaded[:, columns], matrix[:, columns])
        print("Phase 1 direction-removal round-trip: ok")
        return True


if __name__ == "__main__":
    test_direction_removal_roundtrip()
