"""Ceiling comparison: does expanding the edit class raise success?

Compares, on the same locked seeds:
- project_remove on 2 columns (current)
- project_remove on 4 and 8 columns
- project_remove with higher strength grid
- rank-1 write (W[:, c] += s * u * scale) on the same columns

Oracle columns and direction are known. This measures the edit class, not localization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weightcrack.evaluation.synthetic_experiment import _predict, _model_forward


def _inject(W, cols, direction, strength):
    out = W.copy()
    for c in cols:
        out[:, c] = out[:, c] - strength * direction * np.dot(direction, W[:, c])
    return out


def _project_remove(W, cols, direction, strength):
    out = W.copy()
    u = direction / (np.linalg.norm(direction) + 1e-12)
    for c in cols:
        proj = np.dot(u, out[:, c])
        out[:, c] = out[:, c] - strength * u * proj
    return out


def _rank1_write(W, cols, direction, strength):
    out = W.copy()
    u = direction / (np.linalg.norm(direction) + 1e-12)
    for c in cols:
        out[:, c] = out[:, c] + strength * u
    return out


def run_ceiling(n_seeds: int = 200, out_dir: Path | None = None) -> dict:
    doses = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0]
    configs = [
        ("proj_2", _project_remove, 2),
        ("proj_4", _project_remove, 4),
        ("proj_8", _project_remove, 8),
        ("rank1_2", _rank1_write, 2),
        ("rank1_4", _rank1_write, 4),
    ]
    results = {name: {"success": 0, "trials": 0} for name, _, _ in configs}

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(16, 8)).astype(np.float32)
        b = rng.normal(size=16).astype(np.float32)
        W = rng.normal(size=(4, 16)).astype(np.float32)
        c = rng.normal(size=4).astype(np.float32)
        fault_cols = rng.choice(16, size=2, replace=False).tolist()
        fault_dir = rng.normal(size=4).astype(np.float32)
        fault_dir /= np.linalg.norm(fault_dir)
        W_dam = _inject(W, fault_cols, fault_dir, 2.0)

        x = rng.normal(size=(128, 8)).astype(np.float32)
        y = _predict(A, b, W, c, x)
        y_dam = _predict(A, b, W_dam, c, x)
        base_correct = int((y_dam == y).sum())
        if base_correct == 128:
            continue

        for name, edit_fn, n_cols in configs:
            # oracle columns: the true fault columns, padded if needed
            cols = fault_cols[:n_cols] if n_cols <= 2 else fault_cols + rng.choice(
                [i for i in range(16) if i not in fault_cols], size=n_cols - 2, replace=False
            ).tolist()
            best = base_correct
            for dose in doses:
                cand = edit_fn(W_dam, cols, fault_dir, dose)
                score = int((_predict(A, b, cand, c, x) == y).sum())
                if score > best:
                    best = score
            results[name]["trials"] += 1
            if best > base_correct:
                results[name]["success"] += 1

    summary = {
        "n_seeds": n_seeds,
        "doses": doses,
        "results": {
            name: {
                "success_rate": r["success"] / max(r["trials"], 1),
                "successes": r["success"],
                "trials": r["trials"],
            }
            for name, r in results.items()
        },
    }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ceiling.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/ceiling")
    print(json.dumps(run_ceiling(200, out), indent=2))
