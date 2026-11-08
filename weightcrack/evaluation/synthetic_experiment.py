"""Bounded synthetic fault-recovery experiment.

One intact teacher, one fixed rank-one fault on two columns of W,
practice-only localization, heuristic vs random column selection,
held-out evaluation after locking.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from weightcrack.agents.localization import activation_difference, rank_columns, practice_direction_search
from weightcrack.patches.huggingface import apply_project_remove, apply_patch
from weightcrack.schema import WeightCrackPatch, WeightCrackPolicy, PatchChange


def _model_forward(A, b, W, c, x):
    h = np.tanh(x @ A.T + b)
    logits = h @ W.T + c
    return logits, h


def _predict(A, b, W, c, x):
    logits, _ = _model_forward(A, b, W, c, x)
    return np.argmax(logits, axis=1)


# These directory names hold historical reports. Do not overwrite them.
HISTORICAL_REPORTS = {"synthetic_v1", "synthetic_strong", "synthetic_search"}


def run_experiment(out_dir: Path, seed: int = 0, fault_strength: float = 0.8) -> dict:
    """Earlier synthetic loop. Not the preregistered proxy_fault_v1 study."""
    start = time.time()
    out_dir = Path(out_dir)
    if out_dir.name in HISTORICAL_REPORTS:
        raise FileExistsError(
            f"{out_dir.name} is historical evidence. Choose a new output directory."
        )
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intact teacher
    A = rng.normal(size=(16, 8)).astype(np.float32)
    b = rng.normal(size=16).astype(np.float32)
    W = rng.normal(size=(4, 16)).astype(np.float32)
    c = rng.normal(size=4).astype(np.float32)

    # Fixed fault: rank-one on two columns of W
    fault_cols = [3, 7]
    fault_dir = rng.normal(size=4).astype(np.float32)
    fault_dir /= np.linalg.norm(fault_dir)
    W_damaged = W.copy()
    for col in fault_cols:
        W_damaged[:, col] = W_damaged[:, col] - fault_strength * fault_dir * np.dot(fault_dir, W[:, col])

    # Populations
    x_prac_t = rng.normal(size=(128, 8)).astype(np.float32)
    x_prac_p = rng.normal(size=(128, 8)).astype(np.float32)
    x_hold_t = rng.normal(size=(256, 8)).astype(np.float32)
    x_hold_p = rng.normal(size=(256, 8)).astype(np.float32)

    y_prac_t = _predict(A, b, W, c, x_prac_t)
    y_prac_p = _predict(A, b, W, c, x_prac_p)
    y_hold_t = _predict(A, b, W, c, x_hold_t)
    y_hold_p = _predict(A, b, W, c, x_hold_p)

    # Damaged predictions on practice
    y_prac_t_dam = _predict(A, b, W_damaged, c, x_prac_t)
    fail_mask = y_prac_t_dam != y_prac_t
    success_mask = ~fail_mask

    if fail_mask.sum() == 0:
        return {"status": "NO_LOCALIZATION_SIGNAL", "wall_time_s": time.time() - start}

    # Activation difference in the output space of W (logits)
    logits_fail, _ = _model_forward(A, b, W_damaged, c, x_prac_t[fail_mask])
    logits_success, _ = _model_forward(A, b, W_damaged, c, x_prac_t[success_mask])
    diff = activation_difference(logits_fail, logits_success)
    direction = np.array(diff["direction"])
    assert direction.shape[0] == W_damaged.shape[0], "direction must match W rows"

    # Mean-difference direction and columns
    ranked = rank_columns(W_damaged, direction, top_k=2)
    heur_cols = ranked["columns"]

    # Practice-only direction search (mean difference + top principal directions of failures)
    def score_target(W_cand, x, y):
        pred = _predict(A, b, W_cand, c, x)
        return int(np.sum(pred == y))

    def _prac_score(W_cand):
        return float(score_target(W_cand, x_prac_t, y_prac_t))

    policy = WeightCrackPolicy(
        allow_tensors=["W"],
        max_entries=32,
        max_fraction=0.5,
        max_abs_delta=2.0,
        allowed_ops=["project_remove"],
    )
    search = practice_direction_search(
        W_damaged,
        logits_fail,
        logits_success,
        _prac_score,
        policy,
        "W",
        strength=0.5,
    )
    search_direction = np.array(search["direction"]) if search and "direction" in search else direction
    search_cols = search.get("columns", heur_cols) if search else heur_cols

    # Random control columns
    ctrl_rng = np.random.default_rng(seed + 1000)
    rand_cols = ctrl_rng.choice(16, size=2, replace=False).tolist()

    base_hash = hashlib.sha256(np.ascontiguousarray(W_damaged).tobytes()).hexdigest()

    doses = [0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20]
    W_norm = float(np.linalg.norm(W_damaged))

    def score_target(W_cand, x, y):
        pred = _predict(A, b, W_cand, c, x)
        return int(np.sum(pred == y))

    def score_preserve(W_cand, x, y):
        pred = _predict(A, b, W_cand, c, x)
        return int(np.sum(pred != y))  # flips

    results = {"heuristic": [], "random": [], "search": []}

    for name, cols, dir_vec in [
        ("heuristic", heur_cols, direction),
        ("random", rand_cols, direction),
        ("search", search_cols, search_direction),
    ]:
        best = None
        for dose in doses:
            target_norm = dose * W_norm
            # Calibrate strength from geometry (no behavioral feedback)
            # Approximate: strength ~ target_norm / (||u|| * ||col||) ; use fixed mapping
            strength = min(2.0, max(0.01, dose * 2.0))
            try:
                cand, acc = apply_project_remove(W_damaged, dir_vec, cols, strength)
            except Exception as exc:
                results[name].append({"dose": dose, "status": "inadmissible", "error": str(exc)})
                continue

            t_before = score_target(W_damaged, x_prac_t, y_prac_t)
            t_after = score_target(cand, x_prac_t, y_prac_t)
            p_flips = score_preserve(cand, x_prac_p, y_prac_p)
            correction = t_after - t_before
            # Previously correct must not be lost
            prev_correct = y_prac_t_dam == y_prac_t
            cand_pred = _predict(A, b, cand, c, x_prac_t)
            lost = int(np.sum(prev_correct & (cand_pred != y_prac_t)))

            eligible = correction > 0 and lost == 0 and p_flips == 0
            record = {
                "dose": dose,
                "strength": strength,
                "correction": correction,
                "lost": lost,
                "preserve_flips": p_flips,
                "eligible": eligible,
                "accounting": acc,
                "content_hash": hashlib.sha256(np.ascontiguousarray(cand).tobytes()).hexdigest(),
            }
            results[name].append(record)
            if eligible and (best is None or correction > best["correction"] or
                            (correction == best["correction"] and acc["max_abs_stored_delta"] < best["accounting"]["max_abs_stored_delta"])):
                best = record

        results[name + "_finalist"] = best or {"status": "NO_QUALIFIED_CANDIDATE"}

    # Lock hashes before held-out
    locked = {
        "heuristic": results["heuristic_finalist"],
        "random": results["random_finalist"],
        "search": results["search_finalist"],
    }

    # Held-out evaluation
    heldout = {}
    for name in ["heuristic", "random", "search"]:
        fin = locked[name]
        if fin.get("status") == "NO_QUALIFIED_CANDIDATE":
            heldout[name] = {"status": "NO_QUALIFIED_CANDIDATE"}
            continue
        strength = fin["strength"]
        cols = {"heuristic": heur_cols, "random": rand_cols, "search": search_cols}[name]
        dir_vec = search_direction if name == "search" else direction
        cand, _ = apply_project_remove(W_damaged, dir_vec, cols, strength)
        t_gain = score_target(cand, x_hold_t, y_hold_t) - score_target(W_damaged, x_hold_t, y_hold_t)
        prev_ok = _predict(A, b, W_damaged, c, x_hold_t) == y_hold_t
        cand_pred = _predict(A, b, cand, c, x_hold_t)
        losses = int(np.sum(prev_ok & (cand_pred != y_hold_t)))
        p_flips = score_preserve(cand, x_hold_p, y_hold_p)
        repair = t_gain > 0 and losses == 0 and p_flips == 0
        heldout[name] = {
            "target_gain": t_gain,
            "losses": losses,
            "preserve_flips": p_flips,
            "behavioral_repair": repair,
        }

    wall = time.time() - start
    report = {
        "status": "complete",
        "wall_time_s": wall,
        "fault_strength": fault_strength,
        "base_hash": base_hash,
        "direction_norm": float(np.linalg.norm(direction)),
        "heuristic_columns": heur_cols,
        "search_columns": search_cols,
        "random_columns": rand_cols,
        "practice_failures": int(fail_mask.sum()),
        "locked_finalists": locked,
        "heldout": heldout,
        "no_edit_target_practice": int(np.sum(y_prac_t_dam == y_prac_t)),
    }
    (out_dir / "experiment_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/synthetic_legacy_rerun")
    print(json.dumps(run_experiment(out), indent=2))
