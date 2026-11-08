"""Practice-only column proposer for the proxy fault study.

``select_finalists`` receives the practice export, the public config, the
damaged source matrix, and a candidates directory. It does not receive the
intact teacher, the injected fault, or held-out labels. Running it in another
process does not make that boundary a filesystem sandbox: the function simply
has no secret argument and refuses secret-looking filenames.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.localization import direction_trial, rank_columns
from ..canonical import array_content_sha256, file_sha256
from ..errors import EditRejected
from ..patches.huggingface import apply_patch, apply_project_remove
from ..policy import PolicyViolation
from ..schema import PatchChange, WeightCrackPatch, WeightCrackPolicy
from .proxy_model import (
    derive_direction,
    predict,
    prediction_sha256,
    preservation_flips,
    split_counts,
)

PRACTICE_KEYS = (
    "A",
    "b",
    "W",
    "c",
    "x_practice_target",
    "y_practice_target",
    "x_practice_preserve",
    "y_practice_preserve",
)
SOURCE_KEYS = ("A", "b", "W", "c")
METHODS = ("heuristic", "random")


def _refuse_secret_path(path: Path) -> None:
    name = path.name.lower()
    if any(token in name for token in ("secret", "heldout", "holdout", "intact")):
        raise EditRejected("proposer received a disallowed path")


def _load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise EditRejected(f"missing proposer input: {path.name}")
    with np.load(path) as handle:
        keys = tuple(handle.files)
        if keys != required and set(keys) != set(required):
            raise EditRejected(f"{path.name} keys are not the practice allowlist")
        if set(keys) != set(required) or len(keys) != len(required):
            raise EditRejected(f"{path.name} keys are not the practice allowlist")
        return {key: np.array(handle[key], copy=True) for key in required}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EditRejected("nonfinite value in proposer record")
        return value
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    from ..bundle import exclusive_create_bytes

    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    exclusive_create_bytes(path, text.encode("utf-8"))


def _policy(config: dict[str, Any]) -> WeightCrackPolicy:
    policy = WeightCrackPolicy.model_validate(config["policy"])
    if policy.allowed_ops != ["project_remove"]:
        raise EditRejected("public policy must allow only project_remove")
    if policy.allow_tensors != [config["tensor_name"]]:
        raise EditRejected("public policy tensor does not match the study tensor")
    return policy


def _calibrate(
    matrix: np.ndarray,
    direction: list[float],
    columns: list[int],
    target_norm: float,
    tolerance: float,
) -> tuple[float | None, dict[str, Any]]:
    """Match a stored update norm from source geometry. No labels are used."""
    record: dict[str, Any] = {"target_norm": target_norm}
    unit = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if not math.isfinite(norm) or norm < 1e-12 or not math.isfinite(target_norm) or target_norm <= 0:
        record["reason"] = "invalid calibration geometry"
        return None, record
    unit = unit / norm
    work = np.asarray(matrix, dtype=np.float64)
    geometry = math.sqrt(sum(float(np.dot(unit, work[:, column])) ** 2 for column in columns))
    record["geometry"] = geometry
    if not math.isfinite(geometry) or geometry < 1e-12:
        record["reason"] = "zero projection geometry"
        return None, record
    initial = target_norm / geometry
    record["strength_initial"] = initial
    if not (0.0 < initial <= 2.0):
        record["reason"] = "strength outside (0, 2]"
        return None, record
    _, first = apply_project_remove(matrix, direction, columns, float(initial))
    stored = float(first["stored_update_frobenius"])
    record["stored_norm_initial"] = stored
    if not math.isfinite(stored) or stored <= 0.0:
        record["reason"] = "stored update norm is zero"
        return None, record
    adjusted = float(initial) * (target_norm / stored)
    record["strength"] = adjusted
    if not (0.0 < adjusted <= 2.0):
        record["reason"] = "adjusted strength outside (0, 2]"
        return None, record
    _, second = apply_project_remove(matrix, direction, columns, float(adjusted))
    stored_final = float(second["stored_update_frobenius"])
    relative = abs(stored_final - target_norm) / target_norm
    record["stored_norm"] = stored_final
    record["relative_error"] = relative
    record["max_abs_stored_delta"] = float(second["max_abs_stored_delta"])
    if not math.isfinite(relative) or relative > tolerance:
        record["reason"] = "stored norm outside match tolerance"
        return None, record
    return float(adjusted), record


def _score_candidate(model: dict[str, np.ndarray], candidate_W: np.ndarray, baseline: dict[str, np.ndarray]) -> dict[str, Any]:
    target = predict(model["A"], model["b"], candidate_W, model["c"], model["x_practice_target"])
    preserve = predict(model["A"], model["b"], candidate_W, model["c"], model["x_practice_preserve"])
    counts = split_counts(baseline["target"], target, model["y_practice_target"])
    flips = preservation_flips(baseline["preserve"], preserve)
    eligible = counts["corrections"] > 0 and counts["losses"] == 0 and flips == 0
    return {
        "corrections": counts["corrections"],
        "losses": counts["losses"],
        "baseline_correct": counts["baseline_correct"],
        "candidate_correct": counts["candidate_correct"],
        "count": counts["count"],
        "preservation_flips": flips,
        "preservation_count": int(model["y_practice_preserve"].shape[0]),
        "eligible": eligible,
        "target_prediction_sha256": prediction_sha256(target),
        "preserve_prediction_sha256": prediction_sha256(preserve),
    }


def _one_slot(
    method: str,
    dose_index: int,
    dose: float,
    matrix: np.ndarray,
    arrays: dict[str, np.ndarray],
    direction: list[float],
    columns: list[int],
    policy: WeightCrackPolicy,
    config: dict[str, Any],
    source_path: Path,
    candidates_dir: Path,
    baseline: dict[str, np.ndarray],
    base_norm: float,
) -> dict[str, Any]:
    target_norm = float(dose) * base_norm
    slot: dict[str, Any] = {
        "method": method,
        "dose_index": dose_index,
        "dose": float(dose),
        "columns": columns,
        "target_norm": target_norm,
        "status": "rejected",
        "disposition": "rejected",
        "behavioral_qualification": None,
    }
    strength, calibration = _calibrate(
        matrix, direction, columns, target_norm, float(config["match_tolerance"])
    )
    slot["calibration"] = {key: value for key, value in calibration.items() if key != "reason"}
    if strength is None:
        slot["reason"] = calibration["reason"]
        return slot
    bundle = candidates_dir / f"{method}-{dose_index:02d}.wcbundle"
    patch = WeightCrackPatch(
        base_sha256=array_content_sha256(matrix),
        reason="practice column projection edit",
        sector=str(config["sector"]),
        evidence_ids=[str(config["study_id"])],
        changes=[
            PatchChange(
                tensor=str(config["tensor_name"]),
                op="project_remove",
                index=columns,
                direction=direction,
                strength=strength,
            )
        ],
    )
    try:
        receipt = apply_patch(
            source_path,
            patch,
            policy,
            bundle,
            tensor_name=str(config["tensor_name"]),
            task_id=f"{method}-{dose_index:02d}",
        )
    except (EditRejected, PolicyViolation) as exc:
        slot["reason"] = str(exc)
        return slot
    with np.load(bundle / "candidate.npz") as handle:
        saved = np.array(handle["W"], copy=True)
    trial = direction_trial(
        matrix,
        direction,
        columns,
        strength,
        lambda _matrix: 0.0,
        policy,
        str(config["tensor_name"]),
    )
    if not np.array_equal(saved, trial["candidate"]):
        raise RuntimeError("trial candidate does not match the saved candidate")
    if saved.dtype != matrix.dtype:
        raise RuntimeError("saved dtype does not match the source")
    practice = _score_candidate(arrays, saved, baseline)
    slot.update(
        {
            "status": "applied",
            "disposition": "applied",
            "reason": None,
            "bundle_name": bundle.name,
            "strength": strength,
            "candidate_content_sha256": receipt["candidate_content_sha256"],
            "candidate_file_sha256": receipt["candidate_file_sha256"],
            "source_file_sha256": receipt["source_file_sha256"],
            "accounting": receipt["accounting"],
            "stored_update_frobenius": receipt["accounting"]["stored_update_frobenius"],
            "practice": practice,
            "behaviorally_qualified_on_practice": practice["eligible"],
            "trial_application_equivalent": True,
        }
    )
    return slot


def _select_method(slots: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        slot
        for slot in slots
        if slot.get("status") == "applied" and slot.get("behaviorally_qualified_on_practice") is True
    ]
    if not eligible:
        return {"status": "NO_QUALIFIED_CANDIDATE"}
    eligible.sort(
        key=lambda slot: (
            -int(slot["practice"]["corrections"]),
            float(slot["stored_update_frobenius"]),
            int(slot["dose_index"]),
        )
    )
    winner = eligible[0]
    return {
        "status": "qualified",
        "dose_index": winner["dose_index"],
        "bundle_name": winner["bundle_name"],
        "candidate_content_sha256": winner["candidate_content_sha256"],
        "candidate_file_sha256": winner["candidate_file_sha256"],
        "stored_update_frobenius": winner["stored_update_frobenius"],
        "strength": winner["strength"],
        "practice": winner["practice"],
        "accounting": winner["accounting"],
    }


def select_finalists(
    practice_path: Path,
    config_path: Path,
    source_matrix_path: Path,
    candidates_dir: Path,
    deadline: float,
) -> dict[str, Any]:
    """Choose at most one practice finalist per method. Never reads held-out data."""
    for path in (practice_path, config_path, source_matrix_path, candidates_dir):
        _refuse_secret_path(Path(path))
    practice_path = Path(practice_path)
    config_path = Path(config_path)
    source_matrix_path = Path(source_matrix_path)
    candidates_dir = Path(candidates_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    practice = _load_npz(practice_path, PRACTICE_KEYS)
    source = _load_npz(source_matrix_path, SOURCE_KEYS)
    for key in SOURCE_KEYS:
        if not np.array_equal(practice[key], source[key]):
            raise EditRejected("practice export and source matrix disagree")
    matrix = source["W"]
    expected = config["shapes"]["W"]
    if list(matrix.shape) != list(expected):
        raise EditRejected("source shape does not match the frozen config")
    if int(practice["x_practice_target"].shape[0]) != int(config["population"]["practice_target"]):
        raise EditRejected("practice population does not match the frozen config")
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
    derived = derive_direction(
        source["A"],
        source["b"],
        matrix,
        source["c"],
        practice["x_practice_target"],
        practice["y_practice_target"],
    )
    selection: dict[str, Any] = {
        "schema": "weightcrack.proxy_selection.v1",
        "study_id": config["study_id"],
        "status": derived["status"] if derived["status"] != "ok" else "ok",
        "activation": {key: value for key, value in derived.items() if key != "direction"},
        "no_edit": no_edit,
        "methods": {},
        "finalists": {},
    }
    if derived["status"] != "ok":
        return selection
    if time.time() > deadline:
        selection["status"] = "RESOURCE_LIMIT"
        return selection
    direction = derived["direction"]
    selection["activation"]["direction_sha256"] = array_content_sha256(np.asarray(direction, dtype=np.float64))
    ranked = rank_columns(matrix, direction, top_k=int(config["support_size"]))
    heuristic_columns = [int(column) for column in ranked["columns"]]
    if len(heuristic_columns) != int(config["support_size"]):
        raise EditRejected("heuristic support size does not match the frozen config")
    control = np.random.default_rng(int(config["control_seed"]))
    random_columns = [int(column) for column in control.choice(matrix.shape[1], size=int(config["support_size"]), replace=False)]
    column_sets = {"heuristic": heuristic_columns, "random": random_columns}
    doses = [float(dose) for dose in config["doses"]]
    for method in METHODS:
        slots = []
        for index, dose in enumerate(doses):
            if time.time() > deadline:
                selection["status"] = "RESOURCE_LIMIT"
                selection["methods"][method] = {"columns": column_sets[method], "slots": slots}
                return selection
            slots.append(
                _one_slot(
                    method,
                    index,
                    dose,
                    matrix,
                    practice,
                    direction,
                    column_sets[method],
                    policy,
                    config,
                    source_matrix_path,
                    candidates_dir,
                    baseline,
                    base_norm,
                )
            )
        selection["methods"][method] = {"columns": column_sets[method], "slots": slots}
        selection["finalists"][method] = _select_method(slots)
    return selection


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print("usage: python -m weightcrack.evaluation.proxy_proposer PRACTICE CONFIG SOURCE CANDIDATES DEADLINE", file=sys.stderr)
        return 1
    practice, config, source, candidates, deadline_text = argv[1:]
    try:
        deadline = float(deadline_text)
        if not math.isfinite(deadline):
            raise ValueError("deadline is not finite")
        Path(candidates).mkdir(parents=True, exist_ok=True)
        selection = select_finalists(Path(practice), Path(config), Path(source), Path(candidates), deadline)
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
