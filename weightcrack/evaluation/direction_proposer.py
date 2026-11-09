"""Practice-only direction comparison for proxy_direction_v2.

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

from ..agents.localization import rank_columns
from ..canonical import array_content_sha256, f64_hex, file_sha256
from ..errors import EditRejected
from .loss_direction import (
    NO_UNIQUE,
    loss_sensitive_projection_direction,
    practice_loss_gradient,
    selected_support_derivative,
)
from .proxy_model import derive_direction, predict, prediction_sha256
from .proxy_proposer import (
    PRACTICE_KEYS,
    SOURCE_KEYS,
    _jsonable,
    _load_npz,
    _one_slot,
    _policy,
    _refuse_secret_path,
    _select_method,
    _write_json,
)

METHODS = ("mean_logit_difference", "loss_sensitive_projection")
STUDY_ID = "proxy_direction_v2"


def _vector_record(vector: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(vector, dtype=np.float64)
    return {
        "direction_f64_hex": f64_hex(array),
        "direction_sha256": array_content_sha256(array),
        "norm": float(np.linalg.norm(array)),
    }


def _scalar_hex(value: float) -> str:
    return f64_hex(np.asarray([value], dtype=np.float64))


def _support_diagnostic(
    direction: np.ndarray,
    gradient: np.ndarray,
    matrix: np.ndarray,
    columns: list[int],
) -> dict[str, Any]:
    value = selected_support_derivative(direction, gradient, matrix, columns)
    return {
        "selected_support_derivative": value,
        "selected_support_derivative_f64_hex": _scalar_hex(value),
        "diagnostic_affects_selection": False,
    }


def _empty_method(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "reason": reason, "slots": []}


def _method_slots(
    method: str,
    matrix: np.ndarray,
    practice: dict[str, np.ndarray],
    direction: np.ndarray,
    columns: list[int],
    policy,
    config: dict[str, Any],
    source_path: Path,
    candidates_dir: Path,
    baseline: dict[str, np.ndarray],
    base_norm: float,
    deadline: float,
) -> tuple[list[dict[str, Any]], str | None]:
    slots = []
    direction_list = np.asarray(direction, dtype=np.float64).tolist()
    for index, dose in enumerate(config["doses"]):
        if time.time() > deadline:
            return slots, "RESOURCE_LIMIT"
        slots.append(
            _one_slot(
                method,
                index,
                float(dose),
                matrix,
                practice,
                direction_list,
                columns,
                policy,
                config,
                source_path,
                candidates_dir,
                baseline,
                base_norm,
            )
        )
    return slots, None


def select_direction_finalists(
    practice_path: Path,
    config_path: Path,
    source_matrix_path: Path,
    candidates_dir: Path,
    deadline: float,
) -> dict[str, Any]:
    """Choose at most one practice finalist for each direction. Never reads confirmation data."""
    for path in (practice_path, config_path, source_matrix_path, candidates_dir):
        _refuse_secret_path(Path(path))
    practice_path = Path(practice_path)
    config_path = Path(config_path)
    source_matrix_path = Path(source_matrix_path)
    candidates_dir = Path(candidates_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("study_id") != STUDY_ID:
        raise EditRejected("direction proposer received the wrong study")
    if list(config.get("methods", [])) != list(METHODS):
        raise EditRejected("direction proposer method set does not match the study")
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
        raise EditRejected("this study ranks exactly two columns")
    policy = _policy(config)
    base_norm = float(np.linalg.norm(np.asarray(matrix, dtype=np.float64)))
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
    }
    identities = {
        "source_file_sha256": file_sha256(source_matrix_path),
        "practice_export_sha256": file_sha256(practice_path),
        "damaged_W_sha256": array_content_sha256(matrix),
        "practice_target_sha256": array_content_sha256(practice["x_practice_target"]),
        "practice_label_sha256": array_content_sha256(practice["y_practice_target"]),
    }
    gradient_report = practice_loss_gradient(
        source["A"],
        source["b"],
        matrix,
        source["c"],
        practice["x_practice_target"],
        practice["y_practice_target"],
    )
    selection: dict[str, Any] = {
        "schema": "weightcrack.direction_selection.v1",
        "study_id": STUDY_ID,
        "status": "ok",
        "source": identities,
        "no_edit": no_edit,
        "methods": {},
        "finalists": {},
    }
    if gradient_report["status"] != "ok":
        selection["status"] = NO_UNIQUE
        selection["gradient_status"] = gradient_report["status"]
        for method in METHODS:
            selection["methods"][method] = _empty_method(NO_UNIQUE, str(gradient_report.get("reason", NO_UNIQUE)))
            selection["finalists"][method] = {"status": NO_UNIQUE}
        return _jsonable(selection)
    gradient = gradient_report["gradient"]
    derived = derive_direction(
        source["A"],
        source["b"],
        matrix,
        source["c"],
        practice["x_practice_target"],
        practice["y_practice_target"],
    )
    loss = loss_sensitive_projection_direction(
        source["A"],
        source["b"],
        matrix,
        source["c"],
        practice["x_practice_target"],
        practice["y_practice_target"],
    )
    prepared = {
        "mean_logit_difference": _prepare_mean(derived),
        "loss_sensitive_projection": _prepare_loss(loss),
    }
    for method in METHODS:
        if time.time() > deadline:
            selection["status"] = "RESOURCE_LIMIT"
            selection["methods"][method] = _empty_method("RESOURCE_LIMIT", "deadline before this direction")
            selection["finalists"][method] = {"status": "RESOURCE_LIMIT"}
            continue
        prepared_method = prepared[method]
        if prepared_method["status"] != "ok":
            selection["methods"][method] = prepared_method
            selection["finalists"][method] = {"status": prepared_method["status"]}
            continue
        direction = prepared_method.pop("vector")
        ranked = rank_columns(matrix, direction, top_k=2)
        columns = [int(column) for column in ranked["columns"]]
        if len(columns) != 2:
            raise EditRejected("column ranking did not return two columns")
        diagnostic_gradient = gradient
        if method == "loss_sensitive_projection" and loss.get("status") == "ok":
            diagnostic_gradient = loss["gradient"]
        record = {
            **prepared_method,
            "columns": columns,
            "column_scores": [float(score) for score in ranked["scores"]],
            **_support_diagnostic(direction, diagnostic_gradient, matrix, columns),
        }
        slots, limit_status = _method_slots(
            method,
            matrix,
            practice,
            direction,
            columns,
            policy,
            config,
            source_matrix_path,
            candidates_dir,
            baseline,
            base_norm,
            deadline,
        )
        record["slots"] = slots
        selection["methods"][method] = record
        if limit_status == "RESOURCE_LIMIT":
            selection["status"] = "RESOURCE_LIMIT"
            selection["finalists"][method] = {"status": "RESOURCE_LIMIT"}
            continue
        selection["finalists"][method] = _select_method(slots)
    return _jsonable(selection)


def _prepare_mean(derived: dict[str, Any]) -> dict[str, Any]:
    if derived.get("status") != "ok":
        return _empty_method(
            "NO_LOCALIZATION_SIGNAL",
            str(derived.get("reason", "mean logit direction is unavailable")),
        )
    vector = np.asarray(derived["direction"], dtype=np.float64)
    return {
        "status": "ok",
        "definition": derived["definition"],
        "vector": vector,
        **_vector_record(vector),
    }


def _prepare_loss(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "ok":
        observed = {
            key: report[key]
            for key in ("tau", "frobenius_norm_M", "largest_eigenvalue", "second_eigenvalue", "eigen_gap")
            if key in report
        }
        if "eigenvalues" in report:
            observed["eigenvalues_f64_hex"] = f64_hex(np.asarray(report["eigenvalues"], dtype=np.float64))
        if "tau" in report:
            observed["tau_f64_hex"] = _scalar_hex(float(report["tau"]))
        return {
            "status": NO_UNIQUE,
            "reason": str(report.get("reason", NO_UNIQUE)),
            "slots": [],
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


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print(
            "usage: python -m weightcrack.evaluation.direction_proposer "
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
        selection = select_direction_finalists(
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
