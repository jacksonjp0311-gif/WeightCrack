"""Real-matrix probe runner.

Accepts a local .npz containing a single 'weight' array (an extracted MLP matrix).
Runs the same practice-only path used in the synthetic experiment against a
frozen loop-or-not style probe. Does not download models.

Usage:
  python -m weightcrack.evaluation.real_matrix_probe experiments/real_matrix/mlp_down_proj.npz experiments/real_probe
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from weightcrack.agents.localization import activation_difference, rank_columns, practice_direction_search
from weightcrack.patches.huggingface import apply_project_remove


def run_probe(matrix_path: Path, out_dir: Path, seed: int = 0) -> dict:
    data = np.load(matrix_path)
    W = data["weight"].astype(np.float32)
    if W.ndim != 2:
        return {"status": "INVALID_MATRIX", "shape": list(W.shape)}

    rng = np.random.default_rng(seed)
    # Synthetic probe inputs that match the matrix row dimension
    dim = W.shape[0]
    x_prac = rng.normal(size=(64, dim)).astype(np.float32)
    # Treat the matrix itself as a linear map; "failure" is defined by a fixed threshold on a random projection
    # This is a structural probe, not a language-model generation probe.
    proj = rng.normal(size=dim).astype(np.float32)
    scores = x_prac @ proj
    fail_mask = scores > np.median(scores)
    if fail_mask.sum() == 0 or (~fail_mask).sum() == 0:
        return {"status": "NO_LOCALIZATION_SIGNAL"}

    # Static ranking on the top PCA direction of the failure vectors
    fail_vecs = x_prac[fail_mask]
    if fail_vecs.shape[0] >= 2:
        centered = fail_vecs - fail_vecs.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
    else:
        direction = fail_vecs.mean(axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
    write = np.abs(W.T @ direction)
    cols = np.argsort(write)[::-1][:2].tolist()

    # One fixed dose
    try:
        cand, acc = apply_project_remove(W, direction, cols, 0.5)
    except Exception as exc:
        return {"status": "APPLY_FAILED", "reason": str(exc)}

    report = {
        "status": "probe_complete",
        "matrix_shape": list(W.shape),
        "matrix_hash": hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest(),
        "selected_columns": cols,
        "touched_coordinates": acc["touched_coordinates"],
        "max_abs_stored_delta": acc["max_abs_stored_delta"],
        "note": "Structural probe on a loaded matrix. Not a language-model generation measurement.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: real_matrix_probe.py <matrix.npz> <out_dir>")
        sys.exit(1)
    print(json.dumps(run_probe(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
