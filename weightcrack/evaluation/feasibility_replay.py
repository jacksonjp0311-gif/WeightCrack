"""Replayable feasibility check for the final-layer proxy.

Usage:
  python -m weightcrack.evaluation.feasibility_replay
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weightcrack.evaluation.synthetic_experiment import _predict, _model_forward
from weightcrack.patches.huggingface import apply_project_remove


def find_feasible_strength(u, cols, W, h, logits, y, y_base, max_s=2.0):
    gamma = np.zeros(len(y))
    for j in cols:
        gamma += (u @ W[:, j]) * h[:, j]
    for s in np.linspace(0.001, max_s, 200):
        z = logits - s * np.outer(gamma, u)
        pred = z.argmax(1)
        delta = int((pred == y).sum()) - int((y_base == y).sum())
        lost = int(np.sum((y_base == y) & (pred != y)))
        if delta > 0 and lost == 0:
            return float(s)
    return None


def run(n_seeds: int = 20, offset: int = 300) -> dict:
    results = []
    matched = 0
    feasible = 0
    for seed in range(offset, offset + n_seeds):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(16, 8)).astype(np.float32)
        b = rng.normal(size=16).astype(np.float32)
        W = rng.normal(size=(4, 16)).astype(np.float32)
        c = rng.normal(size=4).astype(np.float32)
        fault_cols = rng.choice(16, size=6, replace=False).tolist()
        fault_dir = rng.normal(size=4).astype(np.float32)
        fault_dir /= np.linalg.norm(fault_dir)
        W_dam = W.copy()
        for col in fault_cols:
            W_dam[:, col] = W_dam[:, col] - 6.0 * fault_dir * np.dot(fault_dir, W[:, col])
        x = rng.normal(size=(128, 8)).astype(np.float32)
        y = _predict(A, b, W, c, x)
        y_dam = _predict(A, b, W_dam, c, x)
        fail = y_dam != y
        if fail.sum() < 10:
            results.append({"seed": seed, "status": "skip"})
            continue
        logits, h = _model_forward(A, b, W_dam, c, x)
        centered = logits[fail] - logits[fail].mean(0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        u = vh[0]
        write = np.abs(W_dam.T @ u)
        cols = np.argsort(write)[::-1][:2].tolist()
        s = find_feasible_strength(u, cols, W_dam, h, logits, y, y_dam)
        if s is None:
            results.append({"seed": seed, "status": "no_feasible"})
            continue
        feasible += 1
        cand, _ = apply_project_remove(W_dam, u, cols, s)
        after = int((_predict(A, b, cand, c, x) == y).sum())
        base = int((y_dam == y).sum())
        prev_ok = y_dam == y
        cand_pred = _predict(A, b, cand, c, x)
        lost = int(np.sum(prev_ok & (cand_pred != y)))
        ok = after > base and lost == 0
        if ok:
            matched += 1
        results.append({"seed": seed, "status": "matched" if ok else "storage_fail", "strength": s})
    return {
        "feasible": feasible,
        "matched": matched,
        "seeds": results,
    }


if __name__ == "__main__":
    out = Path("experiments/feasibility_replay")
    out.mkdir(exist_ok=True)
    report = run()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"feasible": report["feasible"], "matched": report["matched"]}, indent=2))
