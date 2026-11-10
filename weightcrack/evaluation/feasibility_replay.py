"""Exact strength intervals for the fixed-hidden-layer classifier.

The hidden state ``h = tanh(Ax + b)`` stays fixed when only ``W`` changes.
Logits then move as ``z(s) = z(0) - s * gamma * u`` with one scalar ``gamma``
per example. Class order is an intersection of affine inequalities, so the
admissible strengths are exact intervals. This is not an exact solver for a
transformer layer whose downstream activations depend on the edited weights.

The permitted strength window is ``(0, 2]``. A 200-point grid is only a
comparison sample. It does not choose the strength.

Usage:
  python -m weightcrack.evaluation.feasibility_replay experiments/feasibility_interval_check
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from weightcrack.canonical import array_content_sha256
from weightcrack.errors import EditRejected, PolicyViolation
from weightcrack.patches.huggingface import (
    STRENGTH_RANGE,
    apply_project_remove,
    construct_candidate,
)
from weightcrack.schema import PatchChange, WeightCrackPatch, WeightCrackPolicy

from .synthetic_experiment import HISTORICAL_REPORTS, _model_forward, _predict

EXACT_SCOPE = "final_layer_affine_logits"
COEFFICIENT_SCALE = 1e-12
GRID_COUNT = 200
BOUNDARY_MARGIN = 1e-6
PROTECTED_OUTPUTS = HISTORICAL_REPORTS | {
    "proxy_fault_v1",
    "proxy_direction_v2",
    "proxy_direction_v2_harness_failure",
    "proxy_support_v3",
    "ceiling",
    "static_rank",
}


def _unit(direction: np.ndarray) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("direction must be a vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("direction has zero or invalid norm")
    unit = vector / norm
    if not np.isfinite(unit).all():
        raise ValueError("direction normalization is nonfinite")
    return unit


def column_gamma(
    direction: np.ndarray,
    columns: list[int],
    weight: np.ndarray,
    hidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``gamma_i = sum_j (u · W[:,j]) h_ij`` for one unit direction."""
    unit = _unit(direction)
    weight = np.asarray(weight, dtype=np.float64)
    hidden = np.asarray(hidden, dtype=np.float64)
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden columns do not match the weight matrix")
    if unit.shape != (weight.shape[0],):
        raise ValueError("direction length must match matrix rows")
    gamma = np.zeros(hidden.shape[0], dtype=np.float64)
    for column in columns:
        if type(column) is not int or column < 0 or column >= weight.shape[1]:
            raise ValueError("column is outside the weight matrix")
        projection = float(np.dot(unit, weight[:, column]))
        gamma += projection * hidden[:, column]
    if not np.isfinite(gamma).all():
        raise ValueError("column gamma is nonfinite")
    return unit, gamma


def predict_shifted(
    logits: np.ndarray,
    gamma: np.ndarray,
    direction: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Argmax of the affine logits. Ties keep the smallest class index."""
    if not math.isfinite(strength):
        raise ValueError("strength is nonfinite")
    unit = _unit(direction)
    logits = np.asarray(logits, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    shifted = logits - float(strength) * gamma[:, None] * unit[None, :]
    if not np.isfinite(shifted).all():
        raise ValueError("shifted logits are nonfinite")
    return np.argmax(shifted, axis=1)


class _Span:
    def __init__(self, lo: float, hi: float, lo_closed: bool, hi_closed: bool):
        self.lo = float(lo)
        self.hi = float(hi)
        self.lo_closed = bool(lo_closed)
        self.hi_closed = bool(hi_closed)

    def empty(self) -> bool:
        if not math.isfinite(self.lo) and self.lo > 0:
            return True
        if self.lo < self.hi:
            return False
        return not (self.lo == self.hi and self.lo_closed and self.hi_closed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "lo_closed": self.lo_closed,
            "hi_closed": self.hi_closed,
        }


def _window() -> _Span:
    return _Span(STRENGTH_RANGE[0], STRENGTH_RANGE[1], False, True)


def _intersect(left: _Span, right: _Span) -> _Span | None:
    if left.lo > right.lo:
        lo, lo_closed = left.lo, left.lo_closed
    elif right.lo > left.lo:
        lo, lo_closed = right.lo, right.lo_closed
    else:
        lo, lo_closed = left.lo, left.lo_closed and right.lo_closed
    if left.hi < right.hi:
        hi, hi_closed = left.hi, left.hi_closed
    elif right.hi < left.hi:
        hi, hi_closed = right.hi, right.hi_closed
    else:
        hi, hi_closed = left.hi, left.hi_closed and right.hi_closed
    span = _Span(lo, hi, lo_closed, hi_closed)
    if span.empty():
        return None
    return span


def _constraint(a: float, b: float, closed: bool) -> _Span | None:
    """Strengths where ``a - s b`` is positive, or nonnegative when ``closed``."""
    scale = max(1.0, abs(a), abs(b))
    if abs(b) <= COEFFICIENT_SCALE * scale:
        if abs(a) <= COEFFICIENT_SCALE * scale:
            return _Span(-math.inf, math.inf, False, False) if closed else None
        return _Span(-math.inf, math.inf, False, False) if a > 0.0 else None
    root = a / b
    if not math.isfinite(root):
        return None
    if b > 0.0:
        return _Span(-math.inf, root, False, closed)
    return _Span(root, math.inf, closed, False)


def _label_interval(
    logit_row: np.ndarray,
    gamma_value: float,
    direction: np.ndarray,
    label: int,
) -> _Span | None:
    if type(label) is not int or label < 0 or label >= int(logit_row.shape[0]):
        raise ValueError("label is outside the logit row")
    span: _Span | None = _Span(-math.inf, math.inf, False, False)
    for other in range(int(logit_row.shape[0])):
        if other == label:
            continue
        difference = float(logit_row[label] - logit_row[other])
        slope = float(gamma_value) * float(direction[label] - direction[other])
        # A smaller index wins a tie, so an earlier class needs a strict gap.
        closed = other > label
        piece = _constraint(difference, slope, closed)
        if piece is None or span is None:
            return None
        span = _intersect(span, piece)
        if span is None:
            return None
    return span


def _merge(spans: list[_Span]) -> list[_Span]:
    ordered = [span for span in spans if span is not None and not span.empty()]
    ordered.sort(key=lambda span: (span.lo, not span.lo_closed))
    merged: list[_Span] = []
    for span in ordered:
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        touches = previous.hi > span.lo or (
            previous.hi == span.lo and (previous.hi_closed or span.lo_closed)
        )
        if not touches:
            merged.append(span)
            continue
        if span.hi > previous.hi:
            previous.hi = span.hi
            previous.hi_closed = span.hi_closed
        elif span.hi == previous.hi:
            previous.hi_closed = previous.hi_closed or span.hi_closed
    return merged


def _clip_window(spans: list[_Span], cap: float) -> list[_Span]:
    limit = _Span(STRENGTH_RANGE[0], min(STRENGTH_RANGE[1], cap), False, True)
    clipped = []
    for span in spans:
        piece = _intersect(span, limit)
        if piece is not None:
            clipped.append(piece)
    return _merge(clipped)


def contains_strength(strength: float, intervals: list[dict[str, Any]]) -> bool:
    if not math.isfinite(strength):
        return False
    for interval in intervals:
        lo = float(interval["lo"])
        hi = float(interval["hi"])
        if strength < lo or strength > hi:
            continue
        if strength == lo and not interval["lo_closed"]:
            continue
        if strength == hi and not interval["hi_closed"]:
            continue
        return True
    return False


def _retention_mask(
    teacher: np.ndarray,
    baseline: np.ndarray,
    protection: str,
    protected_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    teacher = np.asarray(teacher)
    baseline = np.asarray(baseline)
    if teacher.shape != baseline.shape:
        raise ValueError("teacher and baseline populations differ")
    if protection == "correct_targets":
        retain = baseline == teacher
        required = teacher
        return retain, required
    if protection != "baseline_predictions":
        raise ValueError("protection mode is not recognized")
    if protected_mask is None:
        raise ValueError("baseline protection requires a host mask")
    retain = np.asarray(protected_mask, dtype=bool)
    if retain.shape != teacher.shape:
        raise ValueError("protection mask does not match the population")
    required = np.array(baseline, copy=True)
    required[retain] = baseline[retain]
    return retain, required


def _correction_mask(teacher: np.ndarray, baseline: np.ndarray, retain: np.ndarray) -> np.ndarray:
    return (baseline != teacher) & ~retain


def feasible_strength_intervals(
    logits: np.ndarray,
    gamma: np.ndarray,
    direction: np.ndarray,
    teacher: np.ndarray,
    baseline: np.ndarray,
    protection: str = "correct_targets",
    protected_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Intervals where retention holds and at least one target is corrected.

    ``correct_targets`` keeps every currently correct teacher prediction.
    ``baseline_predictions`` keeps the host mask's current predictions.
    Either way, a strength is feasible only when some incorrect, unprotected
    target becomes the teacher label.
    """
    unit = _unit(direction)
    logits = np.asarray(logits, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    teacher = np.asarray(teacher)
    baseline = np.asarray(baseline)
    if logits.ndim != 2 or gamma.shape != (logits.shape[0],) or teacher.shape != (logits.shape[0],):
        raise ValueError("logit population does not match gamma and labels")
    if not np.isfinite(logits).all() or not np.isfinite(gamma).all():
        return {
            "status": "empty",
            "reason": "nonfinite logits",
            "intervals": [],
            "exact_for": EXACT_SCOPE,
            "protection": protection,
        }
    retain, required = _retention_mask(teacher, baseline, protection, protected_mask)
    corrections = _correction_mask(teacher, baseline, retain)
    report: dict[str, Any] = {
        "status": "empty",
        "reason": None,
        "intervals": [],
        "exact_for": EXACT_SCOPE,
        "protection": protection,
        "retained_examples": int(retain.sum()),
        "correction_candidates": int(corrections.sum()),
    }
    if int(corrections.sum()) == 0:
        report["reason"] = "no incorrect target can change without violating protection"
        return report
    retained: _Span | None = _Span(-math.inf, math.inf, False, False)
    for index in np.flatnonzero(retain):
        piece = _label_interval(logits[index], float(gamma[index]), unit, int(required[index]))
        if piece is None or retained is None:
            retained = None
            break
        retained = _intersect(retained, piece)
        if retained is None:
            break
    if retained is None:
        report["reason"] = "retention constraints have an empty intersection"
        return report
    pieces = []
    for index in np.flatnonzero(corrections):
        piece = _label_interval(logits[index], float(gamma[index]), unit, int(teacher[index]))
        if piece is None:
            continue
        both = _intersect(piece, retained)
        if both is not None:
            pieces.append(both)
    merged = _clip_window(pieces, math.inf)
    if not merged:
        report["reason"] = "no strength in (0, 2] corrects a target while retention holds"
        return report
    report["status"] = "ok"
    report["intervals"] = [span.as_dict() for span in merged]
    return report


def select_interior_strength(intervals: list[dict[str, Any]]) -> float | None:
    """A finite strength strictly inside the permitted window and the interval."""
    if not intervals:
        return None
    first = intervals[0]
    lo = float(first["lo"])
    hi = float(first["hi"])
    left = lo if first["lo_closed"] else math.nextafter(lo, hi)
    right = hi if first["hi_closed"] else math.nextafter(hi, lo)
    if not math.isfinite(left) or not math.isfinite(right) or left > right:
        return None
    if left == right:
        strength = left
    else:
        strength = (left + right) / 2.0
    if not math.isfinite(strength) or not STRENGTH_RANGE[0] < strength <= STRENGTH_RANGE[1]:
        return None
    if not contains_strength(strength, intervals):
        return None
    return float(strength)


def _stored_delta_cap(unit: np.ndarray, weight: np.ndarray, columns: list[int], limit: float) -> float:
    peak = 0.0
    for column in columns:
        projection = abs(float(np.dot(unit, np.asarray(weight[:, column], dtype=np.float64))))
        peak = max(peak, projection * float(np.max(np.abs(unit))))
    if peak == 0.0:
        return math.inf
    return float(limit) / peak


def apply_policy_constraints(
    intervals: list[dict[str, Any]],
    weight: np.ndarray,
    direction: np.ndarray,
    columns: list[int],
    policy: WeightCrackPolicy,
) -> dict[str, Any]:
    """Drop strengths the declared coordinate or stored-delta budget cannot admit."""
    touched = int(weight.shape[0]) * len(columns)
    denominator = int(weight.size)
    if touched > policy.max_entries:
        return {"status": "empty", "reason": "touched coordinates exceed max_entries", "intervals": []}
    if denominator < 1 or touched / denominator > float(policy.max_fraction):
        return {"status": "empty", "reason": "touched fraction exceeds max_fraction", "intervals": []}
    unit = _unit(direction)
    cap = _stored_delta_cap(unit, weight, columns, float(policy.max_abs_delta))
    spans = [
        _Span(float(item["lo"]), float(item["hi"]), bool(item["lo_closed"]), bool(item["hi_closed"]))
        for item in intervals
    ]
    clipped = _clip_window(spans, cap)
    if not clipped:
        return {"status": "empty", "reason": "stored-delta budget excludes the interval", "intervals": []}
    return {"status": "ok", "reason": None, "intervals": [span.as_dict() for span in clipped]}


def behavior_holds(
    prediction: np.ndarray,
    teacher: np.ndarray,
    baseline: np.ndarray,
    protection: str = "correct_targets",
    protected_mask: np.ndarray | None = None,
) -> bool:
    retain, required = _retention_mask(teacher, baseline, protection, protected_mask)
    if np.any(retain & (prediction != required)):
        return False
    corrections = _correction_mask(teacher, baseline, retain)
    return bool(np.any(corrections & (prediction == teacher)))


def grid_disagreements(
    logits: np.ndarray,
    gamma: np.ndarray,
    direction: np.ndarray,
    teacher: np.ndarray,
    baseline: np.ndarray,
    intervals: list[dict[str, Any]],
    protection: str = "correct_targets",
    protected_mask: np.ndarray | None = None,
    count: int = GRID_COUNT,
    margin: float = BOUNDARY_MARGIN,
) -> list[float]:
    """Sample the old grid away from interval endpoints and compare predicates."""
    unit = _unit(direction)
    found = []
    for strength in np.linspace(0.001, STRENGTH_RANGE[1], count):
        strength = float(strength)
        near = False
        for interval in intervals:
            if min(abs(strength - float(interval["lo"])), abs(strength - float(interval["hi"]))) <= margin:
                near = True
                break
        if near:
            continue
        prediction = predict_shifted(logits, gamma, unit, strength)
        numeric = behavior_holds(prediction, teacher, baseline, protection, protected_mask)
        if numeric != contains_strength(strength, intervals):
            found.append(strength)
    return found


def separate_grid_findings(
    logits: np.ndarray,
    gamma: np.ndarray,
    direction: np.ndarray,
    teacher: np.ndarray,
    baseline: np.ndarray,
    behavior_intervals: list[dict[str, Any]],
    policy_intervals: list[dict[str, Any]],
    protection: str = "correct_targets",
    protected_mask: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Keep solver mismatches separate from strengths removed only by policy.

    A stored-delta cap can exclude a strength whose predictions are still
    feasible. That exclusion is not an analytic disagreement.
    """
    behavior = grid_disagreements(
        logits,
        gamma,
        direction,
        teacher,
        baseline,
        behavior_intervals,
        protection,
        protected_mask,
    )
    policy = grid_disagreements(
        logits,
        gamma,
        direction,
        teacher,
        baseline,
        policy_intervals,
        protection,
        protected_mask,
    )
    seen = set(behavior)
    exclusions = [
        strength
        for strength in policy
        if strength not in seen and contains_strength(strength, behavior_intervals)
    ]
    return {"behavior_disagreements": behavior, "policy_exclusions": exclusions}


def _policy() -> WeightCrackPolicy:
    return WeightCrackPolicy(
        allow_tensors=["W"],
        max_entries=32,
        max_fraction=0.5,
        max_abs_delta=2.0,
        allowed_ops=["project_remove"],
    )


class ReconstructionMismatch(RuntimeError):
    """The shared gate and the direct application stored different values."""


def reconstruct_stored_candidate(weight: np.ndarray, direction: np.ndarray, columns: list[int], strength: float):
    """Build the stored matrix through the shared gate and through direct application."""
    source = np.array(weight, copy=True)
    unit = _unit(direction)
    policy = _policy()
    patch = WeightCrackPatch(
        base_sha256=array_content_sha256(weight),
        reason="final-layer feasibility replay",
        sector="synthetic_classifier",
        evidence_ids=["feasibility_interval"],
        changes=[
            PatchChange(
                tensor="W",
                op="project_remove",
                index=list(columns),
                direction=unit.tolist(),
                strength=float(strength),
            )
        ],
    )
    candidate, accounting, _normalized = construct_candidate(weight, patch, policy, "W")
    if (
        not isinstance(weight, np.ndarray)
        or weight.shape != source.shape
        or weight.dtype != source.dtype
        or not np.array_equal(weight, source)
    ):
        raise RuntimeError("source reconstruction changed the caller's matrix")
    direct, direct_accounting = apply_project_remove(source, unit, list(columns), float(strength))
    if not np.array_equal(candidate, direct) or accounting != direct_accounting:
        raise ReconstructionMismatch("stored candidate does not match independent application")
    return candidate, accounting


def run(n_seeds: int = 20, offset: int = 300) -> dict[str, Any]:
    """Same twenty seeds as the recorded grid check. Strengths come from the intervals."""
    results = []
    feasible = 0
    matched = 0
    reconstructed = 0
    disagreements = 0
    policy_exclusions = 0
    for seed in range(offset, offset + n_seeds):
        rng = np.random.default_rng(seed)
        hidden = rng.normal(size=(16, 8)).astype(np.float32)
        bias = rng.normal(size=16).astype(np.float32)
        weight = rng.normal(size=(4, 16)).astype(np.float32)
        out_bias = rng.normal(size=4).astype(np.float32)
        fault_columns = rng.choice(16, size=6, replace=False).tolist()
        fault_direction = rng.normal(size=4).astype(np.float32)
        fault_direction = fault_direction / np.linalg.norm(fault_direction)
        damaged = np.array(weight, copy=True)
        for column in fault_columns:
            damaged[:, column] = damaged[:, column] - 6.0 * fault_direction * np.dot(fault_direction, weight[:, column])
        inputs = rng.normal(size=(128, 8)).astype(np.float32)
        teacher = _predict(hidden, bias, weight, out_bias, inputs)
        baseline = _predict(hidden, bias, damaged, out_bias, inputs)
        failed = baseline != teacher
        if int(failed.sum()) < 10:
            results.append({"seed": seed, "status": "skip"})
            continue
        logits, activations = _model_forward(hidden, bias, damaged, out_bias, inputs)
        centered = np.asarray(logits[failed], dtype=np.float64)
        centered = centered - centered.mean(axis=0)
        _values, _singular, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        scores = np.abs(np.asarray(damaged, dtype=np.float64).T @ direction)
        columns = np.argsort(scores)[::-1][:2].tolist()
        unit, gamma = column_gamma(direction, columns, damaged, activations)
        solved = feasible_strength_intervals(logits, gamma, unit, teacher, baseline)
        if solved["status"] != "ok":
            empty_sample = grid_disagreements(
                logits, gamma, unit, teacher, baseline, solved["intervals"]
            )
            disagreements += len(empty_sample)
            results.append(
                {
                    "seed": seed,
                    "status": "no_feasible",
                    "reason": solved["reason"],
                    "behavior_grid_disagreements": empty_sample,
                }
            )
            continue
        constrained = apply_policy_constraints(solved["intervals"], damaged, unit, columns, _policy())
        if constrained["status"] != "ok":
            results.append({"seed": seed, "status": "policy_empty", "reason": constrained["reason"]})
            continue
        strength = select_interior_strength(constrained["intervals"])
        if strength is None:
            results.append({"seed": seed, "status": "no_finite_strength"})
            continue
        feasible += 1
        split = separate_grid_findings(
            logits,
            gamma,
            unit,
            teacher,
            baseline,
            solved["intervals"],
            constrained["intervals"],
        )
        sample = split["behavior_disagreements"]
        exclusions = split["policy_exclusions"]
        disagreements += len(sample)
        policy_exclusions += len(exclusions)
        source = np.array(damaged, copy=True)
        try:
            candidate, accounting = reconstruct_stored_candidate(damaged, unit, columns, strength)
        except (EditRejected, PolicyViolation, ReconstructionMismatch) as exc:
            results.append(
                {
                    "seed": seed,
                    "status": "reconstruction_rejected",
                    "strength": strength,
                    "reason": str(exc),
                    "behavior_intervals": solved["intervals"],
                    "intervals": constrained["intervals"],
                    "behavior_grid_disagreements": sample,
                    "policy_grid_exclusions": exclusions,
                }
            )
            continue
        if not np.array_equal(damaged, source):
            raise RuntimeError("feasibility replay mutated the damaged source")
        reconstructed += 1
        stored_prediction = _predict(hidden, bias, candidate, out_bias, inputs)
        holds = behavior_holds(stored_prediction, teacher, baseline)
        if holds:
            matched += 1
        results.append(
            {
                "seed": seed,
                "status": "matched" if holds else "storage_mismatch",
                "strength": strength,
                "behavior_intervals": solved["intervals"],
                "intervals": constrained["intervals"],
                "behavior_grid_disagreements": sample,
                "policy_grid_exclusions": exclusions,
                "value_exact_reconstruction": True,
                "candidate_content_sha256": array_content_sha256(candidate),
                "stored_update_frobenius": accounting["stored_update_frobenius"],
                "touched_coordinates": accounting["touched_coordinates"],
                "max_abs_stored_delta": accounting["max_abs_stored_delta"],
            }
        )
    return {
        "exact_for": EXACT_SCOPE,
        "solver": "affine_logit_intervals",
        "seeds_offset": offset,
        "seed_count": n_seeds,
        "feasible": feasible,
        "matched": matched,
        "reconstructed": reconstructed,
        "grid_disagreement_count": disagreements,
        "policy_exclusion_count": policy_exclusions,
        "grid_comparison": "behavior_intervals_against_200_point_grid_away_from_endpoints",
        "seeds": results,
    }


def write_report(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    if out_dir.name in PROTECTED_OUTPUTS:
        raise FileExistsError(f"{out_dir.name} is protected evidence")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not an empty output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m weightcrack.evaluation.feasibility_replay OUTPUT_DIR",
            file=sys.stderr,
        )
        return 1
    try:
        report = run()
        write_report(Path(argv[1]), report)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "feasible": report["feasible"],
                "matched": report["matched"],
                "reconstructed": report["reconstructed"],
                "grid_disagreement_count": report["grid_disagreement_count"],
                "policy_exclusion_count": report["policy_exclusion_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
