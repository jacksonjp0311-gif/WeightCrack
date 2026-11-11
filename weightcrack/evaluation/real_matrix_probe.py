"""Real-matrix probe runner.

Accepts a local .npz containing a single 'weight' array.
Requires a second .npz with 'fail' activation traces when a behavioral claim is intended.
Routes the candidate through construct_candidate so the host gate is applied.
Does not download models.

Usage:
  python -m weightcrack.evaluation.real_matrix_probe matrix.npz out_dir traces.npz
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from weightcrack.patches.huggingface import construct_candidate
from weightcrack.policy import gate_patch, PolicyViolation
from weightcrack.schema import WeightCrackPatch, WeightCrackPolicy, PatchChange


def run_probe(
    matrix_path: Path,
    out_dir: Path,
    traces_path: Path | None = None,
    seed: int = 0,
    top_k: int = 2,
    strength: float = 0.5,
) -> dict:
    data = np.load(matrix_path)
    W = data["weight"].astype(np.float32)
    if W.ndim != 2:
        return {"status": "INVALID_MATRIX", "shape": list(W.shape)}

    if traces_path is None or not traces_path.exists():
        return {"status": "TRACES_REQUIRED", "reason": "Supply traces.npz with a 'fail' array for a behavioral probe."}

    traces = np.load(traces_path)
    if "fail" not in traces:
        return {"status": "TRACES_MISSING_FAIL"}
    fail_vecs = traces["fail"].astype(np.float32)
    if fail_vecs.ndim == 1:
        fail_vecs = fail_vecs[None, :]
    if fail_vecs.shape[0] < 2:
        return {"status": "INSUFFICIENT_FAILURES", "n_fail": int(fail_vecs.shape[0])}

    # PCA direction of failure activations
    centered = fail_vecs - fail_vecs.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    # Static ranking
    write = np.abs(W.T @ direction)
    cols = np.argsort(write)[::-1][:top_k].tolist()

    # Build a patch and route through the host path
    source_sha = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()
    patch = WeightCrackPatch(
        base_sha256=source_sha,
        changes=[
            PatchChange(
                op="project_remove",
                tensor="weight",
                direction=direction.tolist(),
                index=cols,
                strength=strength,
            )
        ],
    )
    policy = WeightCrackPolicy(max_entries=10000, max_fraction=0.05, max_abs_delta=1.0)

    try:
        gate_patch(patch, policy, base_param_count=W.size)
        candidate, accounting = construct_candidate(W, patch, policy)
    except (PolicyViolation, Exception) as exc:
        return {"status": "REJECTED", "reason": str(exc)}

    report = {
        "status": "probe_complete",
        "matrix_shape": list(W.shape),
        "matrix_hash": source_sha,
        "selected_columns": cols,
        "n_failure_traces": int(fail_vecs.shape[0]),
        "strength": strength,
        "touched_coordinates": accounting.get("touched_coordinates"),
        "max_abs_stored_delta": accounting.get("max_abs_stored_delta"),
        "note": "Host-gated probe using supplied failure traces. Not a generation measurement until holdout is opened.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: real_matrix_probe.py <matrix.npz> <out_dir> <traces.npz>")
        sys.exit(1)
    print(json.dumps(run_probe(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
