"""Replay proxy_direction_v2.

    python -m weightcrack.evaluation.verify_direction experiments/proxy_direction_v2

The locked selection is the selection record. Fresh resave booleans are
observations and are not required to match the historical runtime. This
verifier does not accept the v1 random-control arm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.localization import rank_columns
from ..canonical import array_content_sha256, f64_hex, file_sha256
from ..errors import VerificationFailure
from ..verify import verify_bundle
from .direction_experiment import (
    CONFIRMATION_SEED,
    DAMAGED_W_SHA256,
    INTACT_W_SHA256,
    METHODS,
    PRACTICE_EXPORT_SHA256,
    QUESTION,
    SOURCE_FILE_SHA256,
    STUDY_ID,
    confirmation_comparison,
    draw_confirmation_inputs,
    official_config,
    preservation_change_kinds,
)
from .direction_proposer import STUDY_ID as PROPOSER_STUDY_ID
from .loss_direction import loss_sensitive_projection_direction, practice_loss_gradient, selected_support_derivative
from .proxy_experiment import (
    DOSES,
    MATCH_TOLERANCE,
    TENSOR_NAME,
    _json_bytes,
    platform_version,
)
from .proxy_model import derive_direction, predict, prediction_sha256, preservation_flips, split_counts
from .proxy_proposer import _select_method

BANNED_BANK_KEYS = {
    "x_heldout_target",
    "y_heldout_target",
    "x_heldout_preserve",
    "y_heldout_preserve",
    "x_practice_target",
    "y_practice_target",
}


def _fail(message: str) -> None:
    raise VerificationFailure("rejected", message)


def _real_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} is not a boolean")
    return value


def _hex_vector(values: np.ndarray) -> str:
    return f64_hex(np.ascontiguousarray(values, dtype=np.float64))


def _scalar_hex(value: float) -> str:
    return f64_hex(np.asarray([value], dtype=np.float64))


def _load_report(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "experiment_report.json"
    if not path.is_file():
        _fail("experiment report is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _expect_confirmation_fields(scored: dict[str, Any], replay: dict[str, Any], label: str) -> None:
    for key in ("corrections", "losses", "baseline_correct", "candidate_correct", "count"):
        if scored.get(key) != replay["counts"][key]:
            _fail(f"confirmation {key} disagrees for {label}")
    kinds = replay["kinds"]
    if scored.get("preservation_flips") != kinds["flips"]:
        _fail(f"confirmation preservation flips disagree for {label}")
    if scored.get("preservation_corrections") != kinds["corrections"]:
        _fail(f"confirmation preservation corrections disagree for {label}")
    if scored.get("preservation_regressions") != kinds["regressions"]:
        _fail(f"confirmation preservation regressions disagree for {label}")
    if scored.get("preservation_incorrect_to_incorrect") != kinds["incorrect_to_incorrect"]:
        _fail(f"confirmation incorrect-to-incorrect changes disagree for {label}")
    if scored.get("preservation_count") != replay["preservation_count"]:
        _fail(f"confirmation preservation count disagrees for {label}")
    if scored.get("target_prediction_sha256") != prediction_sha256(replay["target"]):
        _fail(f"confirmation target prediction hash disagrees for {label}")
    if scored.get("preserve_prediction_sha256") != prediction_sha256(replay["preserve"]):
        _fail(f"confirmation preservation prediction hash disagrees for {label}")
    repair = replay["counts"]["corrections"] > 0 and replay["counts"]["losses"] == 0 and kinds["flips"] == 0
    if _real_bool(scored.get("behavioral_repair"), f"{label} behavioral repair") is not repair:
        _fail(f"behavioral repair flag disagrees for {label}")


def _replay_confirmation(out_dir: Path, report: dict[str, Any], lock: dict[str, Any], verdicts: dict[str, dict]) -> None:
    bank_path = out_dir / "host_only" / "confirmation_bank.npz"
    with np.load(bank_path) as handle:
        if set(handle.files) & BANNED_BANK_KEYS:
            _fail("confirmation bank contains practice or v1 held-out arrays")
        bank = {key: np.array(handle[key], copy=True) for key in handle.files}
    disclosure = json.loads((out_dir / "host_only" / "confirmation_bank.json").read_text(encoding="utf-8"))
    if disclosure.get("blinded_fault_replication") is not False or disclosure.get("known_damaged_model") is not True:
        _fail("confirmation disclosure does not describe the known damaged model")
    if disclosure.get("confirmation_seed") != CONFIRMATION_SEED:
        _fail("confirmation seed does not match the precommitment")
    if array_content_sha256(bank["W_intact"]) != INTACT_W_SHA256:
        _fail("confirmation teacher is not the frozen intact matrix")
    if array_content_sha256(bank["W_damaged"]) != DAMAGED_W_SHA256:
        _fail("confirmation source is not the frozen damaged matrix")
    drawn_target, drawn_preserve = draw_confirmation_inputs(CONFIRMATION_SEED)
    if not np.array_equal(bank["x_confirmation_target"], drawn_target):
        _fail("confirmation target inputs do not match seed 1112")
    if not np.array_equal(bank["x_confirmation_preserve"], drawn_preserve):
        _fail("confirmation preservation inputs do not match seed 1112")
    y_target = predict(bank["A"], bank["b"], bank["W_intact"], bank["c"], drawn_target)
    y_preserve = predict(bank["A"], bank["b"], bank["W_intact"], bank["c"], drawn_preserve)
    if not np.array_equal(bank["y_confirmation_target"], y_target) or not np.array_equal(
        bank["y_confirmation_preserve"], y_preserve
    ):
        _fail("confirmation labels do not match the intact teacher")
    for key, value in {
        "confirmation_target_sha256": array_content_sha256(drawn_target),
        "confirmation_preserve_sha256": array_content_sha256(drawn_preserve),
        "confirmation_target_label_sha256": array_content_sha256(y_target),
        "confirmation_preserve_label_sha256": array_content_sha256(y_preserve),
    }.items():
        if disclosure.get(key) != value:
            _fail(f"confirmation disclosure {key} does not match the bank")
    baseline_target = predict(bank["A"], bank["b"], bank["W_damaged"], bank["c"], drawn_target)
    baseline_preserve = predict(bank["A"], bank["b"], bank["W_damaged"], bank["c"], drawn_preserve)
    recorded = report["confirmation"]
    if recorded.get("used_for_selection") is not False or recorded.get("opened_after_lock") is not True:
        _fail("confirmation selection flags are not real booleans")
    no_edit = _confirmation_replay(bank, bank["W_damaged"], baseline_target, baseline_preserve, y_target, y_preserve)
    _expect_confirmation_fields(recorded["no_edit"], no_edit, "no-edit")
    methods = recorded.get("methods")
    if set(methods) != set(METHODS):
        _fail("confirmation method set is not the frozen pair")
    for method in METHODS:
        finalist = lock["selection"]["finalists"][method]
        scored = methods[method]
        if type(scored.get("status")) is not str or scored["status"] != finalist.get("status"):
            _fail(f"confirmation status does not match the lock for {method}")
        if finalist.get("status") != "qualified":
            if scored != {"status": finalist["status"]}:
                _fail(f"unqualified method carries confirmation scores for {method}")
            continue
        for key in ("bundle_name", "dose_index", "candidate_content_sha256", "candidate_file_sha256"):
            if scored.get(key) != finalist.get(key):
                _fail(f"confirmation {key} does not match the locked finalist for {method}")
        verdict = verdicts.get(finalist["bundle_name"])
        if verdict is None:
            _fail(f"locked finalist was not reconstructed for {method}")
        with np.load(out_dir / "candidates" / finalist["bundle_name"] / "candidate.npz") as handle:
            candidate = np.array(handle["W"], copy=True)
        replay = _confirmation_replay(bank, candidate, baseline_target, baseline_preserve, y_target, y_preserve)
        _expect_confirmation_fields(scored, replay, method)
        if scored.get("accounting") != verdict["accounting"] or finalist.get("accounting") != verdict["accounting"]:
            _fail(f"confirmation accounting does not match reconstruction for {method}")
        if float(finalist["stored_update_frobenius"]) != float(verdict["accounting"]["stored_update_frobenius"]):
            _fail(f"finalist update norm does not match reconstruction for {method}")
        if scored.get("practice") != finalist.get("practice"):
            _fail(f"duplicated practice record disagrees for {method}")
    expected = confirmation_comparison(methods)
    if report.get("comparison") != expected or recorded.get("comparison") != expected:
        _fail("confirmation comparison does not match the precommitted rule")


def _confirmation_replay(bank, matrix, baseline_target, baseline_preserve, y_target, y_preserve) -> dict[str, Any]:
    target = predict(bank["A"], bank["b"], matrix, bank["c"], bank["x_confirmation_target"])
    preserve = predict(bank["A"], bank["b"], matrix, bank["c"], bank["x_confirmation_preserve"])
    return {
        "counts": split_counts(baseline_target, target, y_target),
        "kinds": preservation_change_kinds(baseline_preserve, preserve, y_preserve),
        "target": target,
        "preserve": preserve,
        "preservation_count": int(bank["x_confirmation_preserve"].shape[0]),
    }


def _check_direction(method: str, payload: dict[str, Any], practice: dict[str, np.ndarray], gradient: np.ndarray) -> None:
    matrix = practice["W"]
    if method == "mean_logit_difference":
        derived = derive_direction(
            practice["A"], practice["b"], matrix, practice["c"],
            practice["x_practice_target"], practice["y_practice_target"],
        )
        if derived.get("status") != "ok":
            if payload.get("status") != "NO_LOCALIZATION_SIGNAL" or payload.get("slots"):
                _fail("mean logit failure was not recorded")
            return
        vector = np.asarray(derived["direction"], dtype=np.float64)
    else:
        report = loss_sensitive_projection_direction(
            practice["A"], practice["b"], matrix, practice["c"],
            practice["x_practice_target"], practice["y_practice_target"],
        )
        if report.get("status") != "ok":
            if payload.get("status") != report["status"] or payload.get("slots"):
                _fail("loss-sensitive failure was not recorded")
            return
        vector = np.asarray(report["direction"], dtype=np.float64)
        if payload.get("eigenvalues_f64_hex") != _hex_vector(report["eigenvalues"]):
            _fail("loss-sensitive eigenvalues do not match recomputation")
        if payload.get("tau_f64_hex") != _scalar_hex(float(report["tau"])):
            _fail("loss-sensitive tolerance does not match recomputation")
        if payload.get("full_matrix_derivative_f64_hex") != _scalar_hex(float(report["full_matrix_derivative"])):
            _fail("full-matrix derivative does not match recomputation")
        gradient = report["gradient"]
    if payload.get("status") != "ok":
        _fail(f"{method} omitted a direction the formula produced")
    if payload.get("direction_f64_hex") != _hex_vector(vector):
        _fail(f"{method} direction does not match recomputation")
    ranked = rank_columns(matrix, vector, top_k=2)
    if payload.get("columns") != ranked["columns"]:
        _fail(f"{method} columns do not match absolute projection")
    if not np.allclose(np.asarray(payload.get("column_scores"), dtype=np.float64), np.asarray(ranked["scores"])):
        _fail(f"{method} column scores do not match absolute projection")
    diagnostic = selected_support_derivative(vector, gradient, matrix, ranked["columns"])
    if payload.get("selected_support_derivative_f64_hex") != _scalar_hex(diagnostic):
        _fail(f"{method} selected-support diagnostic does not match recomputation")
    if payload.get("diagnostic_affects_selection") is not False:
        _fail(f"{method} diagnostic was marked as a selection input")
    slots = payload.get("slots")
    if not isinstance(slots, list) or len(slots) != len(DOSES):
        _fail(f"{method} did not keep the eight dose slots")
    for index, (slot, dose) in enumerate(zip(slots, DOSES)):
        if int(slot.get("dose_index")) != index or float(slot.get("dose")) != float(dose):
            _fail(f"{method} dose slot identity changed")
        if slot.get("columns") != ranked["columns"]:
            _fail(f"{method} slot changed the ranked columns")


def _audit_slots(out_dir: Path, selection: dict[str, Any], practice: dict[str, np.ndarray]) -> dict[str, dict]:
    baseline_target = predict(
        practice["A"], practice["b"], practice["W"], practice["c"], practice["x_practice_target"]
    )
    baseline_preserve = predict(
        practice["A"], practice["b"], practice["W"], practice["c"], practice["x_practice_preserve"]
    )
    verdicts = {}
    source = out_dir / "proposal_inputs" / "damaged_source.npz"
    for method in METHODS:
        payload = selection["methods"][method]
        applied_slots = []
        for slot in payload.get("slots", []):
            if slot.get("status") != "applied":
                continue
            applied_slots.append(slot)
            bundle = out_dir / "candidates" / slot["bundle_name"]
            verdict = verify_bundle(bundle, source, TENSOR_NAME, source_path=source)
            if verdict["value_exact_reconstruction"] is not True:
                _fail("reconstruction is not value-exact")
            verdicts[slot["bundle_name"]] = verdict
            with np.load(bundle / "candidate.npz") as handle:
                candidate = np.array(handle["W"], copy=True)
            target = predict(practice["A"], practice["b"], candidate, practice["c"], practice["x_practice_target"])
            preserve = predict(
                practice["A"], practice["b"], candidate, practice["c"], practice["x_practice_preserve"]
            )
            counts = split_counts(baseline_target, target, practice["y_practice_target"])
            flips = preservation_flips(baseline_preserve, preserve)
            recorded = slot.get("practice")
            if recorded.get("corrections") != counts["corrections"] or recorded.get("losses") != counts["losses"]:
                _fail(f"practice corrections do not match the saved candidate for {method}")
            if recorded.get("preservation_flips") != flips or recorded.get("count") != counts["count"]:
                _fail(f"practice population does not match the saved candidate for {method}")
            if recorded.get("baseline_correct") != counts["baseline_correct"]:
                _fail(f"practice baseline does not match the saved candidate for {method}")
            if recorded.get("candidate_correct") != counts["candidate_correct"]:
                _fail(f"practice candidate count does not match the saved candidate for {method}")
            if recorded.get("target_prediction_sha256") != prediction_sha256(target):
                _fail(f"practice target hash does not match the saved candidate for {method}")
            if recorded.get("preserve_prediction_sha256") != prediction_sha256(preserve):
                _fail(f"practice preservation hash does not match the saved candidate for {method}")
            eligible = counts["corrections"] > 0 and counts["losses"] == 0 and flips == 0
            if _real_bool(recorded.get("eligible"), f"{method} eligibility") is not eligible:
                _fail(f"practice eligibility does not match the saved candidate for {method}")
            reconstructed = float(verdict["accounting"]["stored_update_frobenius"])
            if float(slot["stored_update_frobenius"]) != reconstructed:
                _fail(f"slot update norm does not match reconstruction for {method}")
            relative = abs(reconstructed - float(slot["target_norm"])) / float(slot["target_norm"])
            if relative > MATCH_TOLERANCE:
                _fail(f"stored update norm is outside the frozen tolerance for {method}")
            if slot.get("accounting") != verdict["accounting"]:
                _fail(f"slot accounting does not match reconstruction for {method}")
        finalist = selection["finalists"][method]
        if payload.get("status") != "ok":
            if payload.get("slots") or finalist != {"status": payload["status"]}:
                _fail(f"failed direction still has a finalist for {method}")
            continue
        expected = _select_method(payload["slots"])
        if finalist.get("status") != expected.get("status"):
            _fail(f"finalist does not match the practice rule for {method}")
        if expected.get("status") == "qualified" and finalist.get("bundle_name") != expected.get("bundle_name"):
            _fail(f"finalist bundle does not match the practice rule for {method}")
    return verdicts


def verify_study(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if PROPOSER_STUDY_ID != STUDY_ID:
        _fail("direction proposer study id drifted")
    report = _load_report(out_dir)
    if report.get("schema") != "weightcrack.direction_report.v1" or report.get("study_id") != STUDY_ID:
        _fail("unexpected direction report identity")
    if report.get("question") != QUESTION:
        _fail("study question changed")
    timing = report.get("timing")
    if not isinstance(timing, dict) or timing.get("limit_seconds") != 60.0 or "wall_seconds" not in timing:
        _fail("timing evidence is missing")
    if report["status"] == "RESOURCE_LIMIT":
        if report.get("confirmation_opened") is True and report.get("invalidated_by_time_limit") is not True:
            _fail("a timed-out study scored confirmation as a result")
        return {"status": "verified", "study_status": "RESOURCE_LIMIT", "wall_seconds": timing["wall_seconds"]}
    if report["status"] == "VERIFICATION_FAILED":
        return {"status": "verified", "study_status": "VERIFICATION_FAILED", "wall_seconds": timing["wall_seconds"]}
    if report["status"] != "complete":
        _fail("unknown study status")
    lock_path = out_dir / "proposal_lock.json"
    if file_sha256(lock_path) != report["lock_sha256"]:
        _fail("lock file hash does not match the report")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("confirmation_used_for_selection") is not False or lock.get("heldout_used_for_selection") is not False:
        _fail("lock says confirmation or held-out data was used for selection")
    if report.get("selection") != lock.get("selection"):
        _fail("duplicated report selection does not match the lock")
    config_path = out_dir / "proposal_inputs" / "public_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != json.loads(_json_bytes(official_config()).decode("utf-8")):
        _fail("public config does not match the frozen direction study")
    if file_sha256(config_path) != report["public_config_sha256"]:
        _fail("public config hash does not match the report")
    practice_path = out_dir / "proposal_inputs" / "practice_export.npz"
    source_path = out_dir / "proposal_inputs" / "damaged_source.npz"
    if file_sha256(practice_path) != PRACTICE_EXPORT_SHA256 or report["practice_export_sha256"] != PRACTICE_EXPORT_SHA256:
        _fail("practice export is not the frozen v1 file")
    if file_sha256(source_path) != SOURCE_FILE_SHA256 or report["source_file_sha256"] != SOURCE_FILE_SHA256:
        _fail("damaged source is not the frozen v1 file")
    if file_sha256(out_dir / "host_only" / "confirmation_bank.npz") != lock["confirmation_bank_sha256"]:
        _fail("confirmation bank hash does not match the lock")
    if report.get("confirmation_bank_sha256") != lock["confirmation_bank_sha256"]:
        _fail("report confirmation hash does not match the lock")
    events = [
        json.loads(line)
        for line in (out_dir / "study_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = [event["stage"] for event in events]
    if "lock_written" not in stages or "confirmation_opened" not in stages:
        _fail("event record is missing the lock or confirmation stage")
    if stages.index("lock_written") > stages.index("confirmation_opened"):
        _fail("confirmation stage was recorded before the lock")
    with np.load(practice_path) as handle:
        practice = {key: np.array(handle[key], copy=True) for key in handle.files}
    gradient_report = practice_loss_gradient(
        practice["A"], practice["b"], practice["W"], practice["c"],
        practice["x_practice_target"], practice["y_practice_target"],
    )
    if gradient_report["status"] != "ok":
        _fail("practice loss gradient could not be recomputed")
    selection = lock["selection"]
    if set(selection.get("methods", {})) != set(METHODS) or set(selection.get("finalists", {})) != set(METHODS):
        _fail("lock method set is not the frozen pair")
    for method in METHODS:
        _check_direction(method, selection["methods"][method], practice, gradient_report["gradient"])
    verdicts = _audit_slots(out_dir, selection, practice)
    applied = len(verdicts)
    historical = report.get("resave_byte_identical")
    if not isinstance(historical, list) or len(historical) != applied or any(type(item) is not bool for item in historical):
        _fail("historical resave observations are not a boolean list")
    if type(report.get("verified_applied_bundles")) is not int or report["verified_applied_bundles"] != applied:
        _fail("reported applied bundle count does not match the lock")
    budget = report.get("budget")
    if not isinstance(budget, dict):
        _fail("budget record is missing")
    recorded_slots = sum(len(selection["methods"][method].get("slots", [])) for method in METHODS)
    rejected = recorded_slots - applied
    if budget.get("slots_recorded") != recorded_slots or budget.get("applied_bundles") != applied:
        _fail("reported budget does not match the lock")
    if budget.get("rejected_slots") != rejected or budget.get("maximum_candidate_slots") != 16:
        _fail("reported budget does not match the protocol cap")
    if applied:
        if report.get("value_exact_reconstruction") is not True:
            _fail("report does not record value-exact reconstruction")
    elif report.get("value_exact_reconstruction") is not None:
        _fail("report records reconstruction without an applied bundle")
    _replay_confirmation(out_dir, report, lock, verdicts)
    current_flags = []
    for method in METHODS:
        for slot in selection["methods"][method].get("slots", []):
            if slot.get("status") == "applied":
                current_flags.append(verdicts[slot["bundle_name"]]["resave_byte_identical"])
    return {
        "status": "verified",
        "study_status": report["status"],
        "applied_bundles": applied,
        "wall_seconds": timing["wall_seconds"],
        "comparison": report["comparison"],
        "value_exact_reconstruction": True if applied else None,
        "historical_resave_byte_identical": historical,
        "current_resave_byte_identical": current_flags,
        "current_runtime": {"python": platform_version(), "numpy": np.__version__},
    }


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "experiments/proxy_direction_v2")
    try:
        verdict = verify_study(out)
    except VerificationFailure as exc:
        print(f"status=rejected detail={exc}")
        return 1
    except Exception as exc:
        print(f"status=rejected detail={type(exc).__name__}: {exc}")
        return 1
    print(
        f"status={verdict['status']} study_status={verdict['study_status']} "
        f"wall_seconds={verdict['wall_seconds']} comparison={verdict.get('comparison')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
