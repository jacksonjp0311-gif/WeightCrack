"""Practice-only support comparison for proxy_support_v3.

The proposer receives the practice export, the public config, the damaged
source, and a candidates directory. It does not receive the intact teacher
or the confirmation bank. A separate process does not make that boundary a
filesystem sandbox.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..canonical import array_content_sha256, f64_hex, file_sha256
from ..errors import EditRejected
from .loss_direction import NO_UNIQUE, loss_sensitive_projection_direction, mean_cross_entropy
from .proxy_proposer import (
    PRACTICE_KEYS,
    SOURCE_KEYS,
    _jsonable,
    _load_npz,
    _one_slot,
    _policy,
    _select_method,
    _write_json,
)
from .support_rank import (
    INSUFFICIENT_DESCENT_SUPPORT,
    column_products,
    predicted_support_change,
    rank_magnitude,
    rank_signed_contributions,
    sign_agreement,
)

METHODS = ("magnitude_projection", "signed_loss_contribution")
STUDY_ID = "proxy_support_v3"
_FORBIDDEN_PARTS = ("secret", "heldout", "holdout", "intact", "confirmation", "host_only", "teacher")


def _refuse_host_material(path: Path) -> None:
    parts = [part.lower() for part in Path(path).parts]
    if any(any(token in part for token in _FORBIDDEN_PARTS) for part in parts):
        raise EditRejected("proposer received a disallowed path")


def _scalar_hex(value: float) -> str:
    return f64_hex(np.asarray([value], dtype=np.float64))


def _vector_record(vector: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(vector, dtype=np.float64)
    return {
        "direction_f64_hex": f64_hex(array),
        "direction_sha256": array_content_sha256(array),
        "norm": float(np.linalg.norm(array)),
    }


def _direction_record(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "ok":
        observed = {
            key: report[key]
            for key in (
                "tau",
                "frobenius_norm_M",
                "largest_eigenvalue",
                "second_eigenvalue",
                "eigen_gap",
            )
            if key in report
        }
        if "eigenvalues" in report:
            observed["eigenvalues_f64_hex"] = f64_hex(np.asarray(report["eigenvalues"], dtype=np.float64))
        if "tau" in report:
            observed["tau_f64_hex"] = _scalar_hex(float(report["tau"]))
        return {
            "status": NO_UNIQUE,
            "reason": str(report.get("reason", NO_UNIQUE)),
            **observed,
        }
    vector = np.asarray(report["direction"], dtype=np.float64)
    return {
        "status": "ok",
        "definition": "leading eigenvector of (G @ W.T + W @ G.T) / 2",
        "vector": vector,
        "tau": float(report["tau"]),
        "tau_f64_hex": _scalar_hex(float(report["tau"])),
        "frobenius_norm_M": float(report["frobenius_norm_M"]),
        "largest_eigenvalue": float(report["largest_eigenvalue"]),
        "second_eigenvalue": float(report["second_eigenvalue"]),
        "eigen_gap": float(report["eigen_gap"]),
        "eigenvalues_f64_hex": f64_hex(np.asarray(report["eigenvalues"], dtype=np.float64)),
        "full_matrix_derivative": float(report["full_matrix_derivative"]),
        "full_matrix_derivative_f64_hex": _scalar_hex(float(report["full_matrix_derivative"])),
        "count": int(report["count"]),
        **_vector_record(vector),
    }


def _attach_diagnostics(
    slot: dict[str, Any],
    practice: dict[str, np.ndarray],
    direction: np.ndarray,
    gradient: np.ndarray,
    matrix: np.ndarray,
    columns: list[int],
    saved: np.ndarray | None,
) -> None:
    """Record the prediction and the measured practice loss. Do not select on it."""
    before = mean_cross_entropy(
        practice["A"], practice["b"], matrix, practice["c"],
        practice["x_practice_target"], practice["y_practice_target"],
    )
    after = None
    if saved is not None:
        after = mean_cross_entropy(
            practice["A"], practice["b"], saved, practice["c"],
            practice["x_practice_target"], practice["y_practice_target"],
        )
    products = column_products(direction, gradient, matrix)
    selected = np.asarray([products["contributions"][column] for column in columns], dtype=np.float64)
    strength = slot.get("strength")
    record: dict[str, Any] = {
        "signed_contributions": [float(value) for value in selected],
        "signed_contributions_f64_hex": f64_hex(selected),
        "practice_loss_before": before,
        "practice_loss_before_f64_hex": _scalar_hex(before),
        "practice_loss_after": after,
        "diagnostic_affects_selection": False,
    }
    if isinstance(strength, (int, float)) and not isinstance(strength, bool):
        predicted = predicted_support_change(direction, gradient, matrix, columns, float(strength))
        record["selected_support_derivative"] = predicted["selected_support_derivative"]
        record["selected_support_derivative_f64_hex"] = _scalar_hex(predicted["selected_support_derivative"])
        record["predicted_loss_change"] = predicted["predicted_loss_change"]
        record["predicted_loss_change_f64_hex"] = _scalar_hex(predicted["predicted_loss_change"])
        if after is not None:
            actual = after - before
            record["practice_loss_after_f64_hex"] = _scalar_hex(after)
            record["practice_loss_change"] = actual
            record["practice_loss_change_f64_hex"] = _scalar_hex(actual)
            record.update(sign_agreement(predicted["predicted_loss_change"], actual))
    slot.update(record)


def _empty_method(status: str, reason: str, shared: dict[str, Any]) -> dict[str, Any]:
    return {"status": status, "reason": reason, "slots": [], "columns": [], **shared}


def select_support_finalists(
    practice_path: Path,
    config_path: Path,
    source_matrix_path: Path,
    candidates_dir: Path,
    deadline: float,
) -> dict[str, Any]:
    """Choose at most one practice finalist for each support rule. Never reads confirmation data."""
    for path in (practice_path, config_path, source_matrix_path, candidates_dir):
        _refuse_host_material(Path(path))
    practice_path = Path(practice_path)
    config_path = Path(config_path)
    source_matrix_path = Path(source_matrix_path)
    candidates_dir = Path(candidates_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("study_id") != STUDY_ID:
        raise EditRejected("support proposer received the wrong study")
    if list(config.get("methods", [])) != list(METHODS):
        raise EditRejected("support proposer method set does not match the study")
    practice = _load_npz(practice_path, PRACTICE_KEYS)
    source = _load_npz(source_matrix_path, SOURCE_KEYS)
    for key in SOURCE_KEYS:
        if not np.array_equal(practice[key], source[key]):
            raise EditRejected("practice export and source matrix disagree")
    matrix = source["W"]
    if list(matrix.shape) != list(config["shapes"]["W"]):
        raise EditRejected("source shape does not match the frozen config")
    if int(practice["x_practice_target"].shape[0]) != int(config["population"]["practice_target"]):
        raise EditRejected("practice population does not match the frozen config")
    if int(config["support_size"]) != 2:
        raise EditRejected("this study edits exactly two columns")
    policy = _policy(config)
    base_norm = float(np.linalg.norm(np.asarray(matrix, dtype=np.float64)))
    from .proxy_model import predict, prediction_sha256

    baseline = {
        "target": predict(source["A"], source["b"], matrix, source["c"], practice["x_practice_target"]),
        "preserve": predict(source["A"], source["b"], matrix, source["c"], practice["x_practice_preserve"]),
    }
    no_edit = {
        "content_sha256": array_content_sha256(matrix),
        "source_file_sha256": file_sha256(source_matrix_path),
        "practice_target_correct": int(np.count_nonzero(baseline["target"] == practice["y_practice_target"])),
        "practice_target_count": int(practice["y_practice_target"].shape[0]),
        "practice_target_prediction_sha256": prediction_sha256(baseline["target"]),
        "practice_preserve_prediction_sha256": prediction_sha256(baseline["preserve"]),
        "practice_loss": mean_cross_entropy(
            source["A"], source["b"], matrix, source["c"],
            practice["x_practice_target"], practice["y_practice_target"],
        ),
    }
    identities = {
        "source_file_sha256": file_sha256(source_matrix_path),
        "practice_export_sha256": file_sha256(practice_path),
        "damaged_W_sha256": array_content_sha256(matrix),
        "practice_target_sha256": array_content_sha256(practice["x_practice_target"]),
        "practice_label_sha256": array_content_sha256(practice["y_practice_target"]),
    }
    loss = loss_sensitive_projection_direction(
        source["A"], source["b"], matrix, source["c"],
        practice["x_practice_target"], practice["y_practice_target"],
    )
    prepared = _direction_record(loss)
    selection: dict[str, Any] = {
        "schema": "weightcrack.support_selection.v1",
        "study_id": STUDY_ID,
        "status": "ok",
        "source": identities,
        "no_edit": no_edit,
        "direction": {key: value for key, value in prepared.items() if key != "vector"},
        "methods": {},
        "finalists": {},
    }
    shared_failure = {
        "direction_sha256": prepared.get("direction_sha256"),
        "diagnostic_affects_selection": False,
    }
    if prepared["status"] != "ok":
        selection["status"] = NO_UNIQUE
        for method in METHODS:
            selection["methods"][method] = _empty_method(
                NO_UNIQUE, str(prepared.get("reason", NO_UNIQUE)), shared_failure
            )
            selection["finalists"][method] = {"status": NO_UNIQUE}
        return _jsonable(selection)
    direction = prepared.pop("vector")
    gradient = np.asarray(loss["gradient"], dtype=np.float64)
    products = column_products(direction, gradient, matrix)
    selection["contribution_tau"] = float(products["tau"])
    selection["contribution_tau_f64_hex"] = _scalar_hex(float(products["tau"]))
    selection["contributions_f64_hex"] = f64_hex(np.asarray(products["contributions"], dtype=np.float64))
    selection["magnitudes_f64_hex"] = f64_hex(np.asarray(products["magnitudes"], dtype=np.float64))
    magnitude = rank_magnitude(matrix, direction, support_size=2)
    signed = rank_signed_contributions(direction, gradient, matrix, support_size=2)
    plans = {
        "magnitude_projection": magnitude,
        "signed_loss_contribution": signed,
    }
    direction_list = np.asarray(direction, dtype=np.float64).tolist()
    for method in METHODS:
        if time.time() > deadline:
            selection["status"] = "RESOURCE_LIMIT"
            selection["methods"][method] = _empty_method("RESOURCE_LIMIT", "deadline before this method", shared_failure)
            selection["finalists"][method] = {"status": "RESOURCE_LIMIT"}
            continue
        plan = plans[method]
        if plan["status"] != "ok":
            selection["methods"][method] = {
                "status": plan["status"],
                "reason": plan["status"],
                "columns": [],
                "slots": [],
                "positive_count": plan.get("positive_count"),
                "contribution_tau": float(products["tau"]),
                "direction_sha256": prepared["direction_sha256"],
                "diagnostic_affects_selection": False,
            }
            selection["finalists"][method] = {"status": plan["status"]}
            continue
        columns = [int(column) for column in plan["columns"]]
        if len(columns) != 2:
            raise EditRejected("support ranking did not return two columns")
        record: dict[str, Any] = {
            "status": "ok",
            "columns": columns,
            "direction_sha256": prepared["direction_sha256"],
            "diagnostic_affects_selection": False,
            "contribution_tau": float(products["tau"]),
        }
        if method == "magnitude_projection":
            record["column_scores"] = [float(score) for score in plan["scores"]]
        else:
            record["selected_contributions"] = [float(value) for value in plan["selected_contributions"]]
            record["positive_count"] = int(plan["positive_count"])
        slots = []
        limit_status = None
        for index, dose in enumerate(config["doses"]):
            if time.time() > deadline:
                limit_status = "RESOURCE_LIMIT"
                break
            slot = _one_slot(
                method,
                index,
                float(dose),
                matrix,
                practice,
                direction_list,
                columns,
                policy,
                config,
                source_matrix_path,
                candidates_dir,
                baseline,
                base_norm,
            )
            saved = None
            if slot.get("status") == "applied":
                with np.load(candidates_dir / slot["bundle_name"] / "candidate.npz") as handle:
                    saved = np.array(handle["W"], copy=True)
            _attach_diagnostics(slot, practice, direction, gradient, matrix, columns, saved)
            slots.append(slot)
        record["slots"] = slots
        selection["methods"][method] = record
        if limit_status == "RESOURCE_LIMIT":
            selection["status"] = "RESOURCE_LIMIT"
            selection["finalists"][method] = {"status": "RESOURCE_LIMIT"}
            continue
        selection["finalists"][method] = _select_method(slots)
    return _jsonable(selection)


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print(
            "usage: python -m weightcrack.evaluation.support_proposer "
            "PRACTICE CONFIG SOURCE CANDIDATES DEADLINE",
            file=sys.stderr,
        )
        return 1
    practice, config, source, candidates, deadline_text = argv[1:]
    try:
        deadline = float(deadline_text)
        if not math.isfinite(deadline):
            raise ValueError("deadline is not finite")
        Path(candidates).mkdir(parents=True, exist_ok=True)
        selection = select_support_finalists(
            Path(practice), Path(config), Path(source), Path(candidates), deadline
        )
        _write_json(Path(candidates) / "selection.json", selection)
    except Exception as exc:
        print(f"proposer_error: {exc}", file=sys.stderr)
        return 1
    print(f"status={selection['status']}")
    if selection["status"] == "RESOURCE_LIMIT":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
