"""Independent replay of proxy_support_v3.

The lock is the selection record. Reported scores are recomputed from saved
candidates and from the frozen generators. This module does not verify v1 or v2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..canonical import array_content_sha256, f64_from_hex, f64_hex, file_sha256
from ..errors import VerificationFailure
from ..patches.huggingface import apply_project_remove
from ..verify import verify_bundle
from .proxy_experiment import DOSES, FAULT_STRENGTH, MATCH_TOLERANCE, TENSOR_NAME
from .proxy_model import predict, prediction_sha256, preservation_flips, split_counts
from .proxy_proposer import PRACTICE_KEYS, _load_npz, _select_method
from .loss_direction import loss_sensitive_projection_direction, mean_cross_entropy
from .support_experiment import (
    QUESTION,
    confirmation_comparison,
    draw_confirmation_inputs,
    draw_practice_inputs,
    official_config,
    source_identity,
)
from .support_proposer import METHODS, STUDY_ID
from .support_rank import (
    column_products,
    predicted_support_change,
    rank_magnitude,
    rank_signed_contributions,
    sign_agreement,
)


def _fail(message: str) -> None:
    raise VerificationFailure("rejected", message)


def _real_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} is not a real boolean")
    return value


def _scalar_hex(value: float) -> str:
    return f64_hex(np.asarray([value], dtype=np.float64))


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _events(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "study_events.jsonl"
    if not path.is_file():
        _fail("study events are missing")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _event_indexes(rows: list[dict[str, Any]]) -> dict[str, int]:
    found = {}
    for index, row in enumerate(rows):
        stage = row.get("stage")
        if stage in found:
            _fail(f"duplicate study event: {stage}")
        if isinstance(stage, str):
            found[stage] = index
    return found


def verify_study(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    report_path = out_dir / "experiment_report.json"
    if not report_path.is_file():
        _fail("experiment report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "weightcrack.support_report.v1" or report.get("study_id") != STUDY_ID:
        _fail("unexpected support report identity")
    if report.get("question") != QUESTION:
        _fail("study question changed")
    timing = report.get("timing")
    if not isinstance(timing, dict) or timing.get("limit_seconds") != 60.0 or "wall_seconds" not in timing:
        _fail("timing evidence is missing")
    if report.get("status") == "RESOURCE_LIMIT":
        if report.get("confirmation_opened") is True and report.get("invalidated_by_time_limit") is not True:
            _fail("a timed-out study scored confirmation as a result")
        return {"status": "verified", "study_status": "RESOURCE_LIMIT", "wall_seconds": timing["wall_seconds"]}
    if report.get("status") == "VERIFICATION_FAILED":
        if report.get("confirmation_opened") is True:
            _fail("a failed study scored confirmation as a result")
        return {"status": "verified", "study_status": "VERIFICATION_FAILED", "wall_seconds": timing["wall_seconds"]}
    if report.get("status") != "complete":
        _fail("unknown study status")
    if _real_bool(report.get("confirmation_opened"), "confirmation_opened") is not True:
        _fail("a complete study did not open confirmation after the lock")
    events = _event_indexes(_events(out_dir))
    required = ["host_committed", "proposer_returned", "lock_written", "confirmation_opened", "report_written"]
    if any(stage not in events for stage in required):
        _fail("study events are missing a required stage")
    if not (
        events["host_committed"]
        < events["proposer_returned"]
        < events["lock_written"]
        < events["confirmation_opened"]
        < events["report_written"]
    ):
        _fail("confirmation was opened before the practice lock")
    commitment_path = out_dir / "public_commitment.json"
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if file_sha256(commitment_path) != report.get("commitment_sha256"):
        _fail("commitment hash does not match the report")
    if commitment.get("source_identity") != source_identity() or report.get("source_identity") != source_identity():
        _fail("source identity does not match the running modules")
    config_path = out_dir / "proposal_inputs" / "public_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != json.loads(_json_bytes(official_config()).decode("utf-8")):
        _fail("public config does not match the frozen support study")
    if file_sha256(config_path) != report.get("public_config_sha256"):
        _fail("public config hash does not match the report")
    if file_sha256(config_path) != commitment["files"]["public_config.json"]:
        _fail("public config does not match the pre-trial commitment")
    host = json.loads((out_dir / "host_only" / "host_config.json").read_text(encoding="utf-8"))
    forbidden_seeds = {1108, 1109, 1110, 1111, 1112}
    host_seeds = [host.get("model_seed"), host.get("data_seed"), host.get("fault_seed"), host.get("confirmation_seed")]
    if any(type(seed) is not int for seed in host_seeds) or len(set(host_seeds)) != 4:
        _fail("host seeds are not four distinct integers")
    if forbidden_seeds.intersection(host_seeds):
        _fail("host seeds reuse a v1 or v2 seed")
    if host.get("fault_strength") != FAULT_STRENGTH:
        _fail("fault strength changed")
    if file_sha256(out_dir / "host_only" / "host_config.json") != commitment["files"]["host_config.json"]:
        _fail("host config does not match the pre-trial commitment")
    with np.load(out_dir / "host_only" / "intact_model.npz") as handle:
        intact = {key: np.array(handle[key], copy=True) for key in ("A", "b", "W", "c")}
    practice = _load_npz(out_dir / "proposal_inputs" / "practice_export.npz", PRACTICE_KEYS)
    source = _load_npz(out_dir / "proposal_inputs" / "damaged_source.npz", ("A", "b", "W", "c"))
    x_target, x_preserve = draw_practice_inputs(int(host["data_seed"]))
    if not np.array_equal(practice["x_practice_target"], x_target) or not np.array_equal(
        practice["x_practice_preserve"], x_preserve
    ):
        _fail("practice inputs do not match the preregistered generator")
    y_target = predict(intact["A"], intact["b"], intact["W"], intact["c"], x_target)
    y_preserve = predict(intact["A"], intact["b"], intact["W"], intact["c"], x_preserve)
    if not np.array_equal(practice["y_practice_target"], y_target) or not np.array_equal(
        practice["y_practice_preserve"], y_preserve
    ):
        _fail("practice labels do not match the intact teacher")
    fault_columns = [int(column) for column in host["fault_columns"]]
    fault_direction = f64_from_hex(host["fault_direction_f64_hex"]).tolist()
    damaged_w, _accounting = apply_project_remove(intact["W"], fault_direction, fault_columns, float(FAULT_STRENGTH))
    if not np.array_equal(source["W"], damaged_w) or not np.array_equal(practice["W"], damaged_w):
        _fail("damaged source does not match the preregistered fault")
    for name in ("damaged_source.npz", "practice_export.npz"):
        folder = "proposal_inputs"
        if file_sha256(out_dir / folder / name) != commitment["files"][name]:
            _fail(f"{name} does not match the pre-trial commitment")
    with np.load(out_dir / "host_only" / "confirmation_bank.npz") as handle:
        bank = {key: np.array(handle[key], copy=True) for key in handle.files}
    if file_sha256(out_dir / "host_only" / "confirmation_bank.npz") != commitment["files"]["confirmation_bank.npz"]:
        _fail("confirmation bank does not match the pre-trial commitment")
    if file_sha256(out_dir / "host_only" / "confirmation_bank.npz") != report.get("confirmation_bank_sha256"):
        _fail("confirmation bank hash does not match the report")
    drawn_target, drawn_preserve = draw_confirmation_inputs(int(host["confirmation_seed"]))
    if not np.array_equal(bank["x_confirmation_target"], drawn_target) or not np.array_equal(
        bank["x_confirmation_preserve"], drawn_preserve
    ):
        _fail("confirmation inputs do not match the committed confirmation seed")
    from .direction_experiment import draw_confirmation_inputs as draw_v2

    v2_target, _v2_preserve = draw_v2()
    if np.array_equal(bank["x_confirmation_target"], v2_target):
        _fail("confirmation bank reuses the v2 draw")
    expected_labels = predict(intact["A"], intact["b"], intact["W"], intact["c"], drawn_target)
    expected_preserve = predict(intact["A"], intact["b"], intact["W"], intact["c"], drawn_preserve)
    if not np.array_equal(bank["y_confirmation_target"], expected_labels) or not np.array_equal(
        bank["y_confirmation_preserve"], expected_preserve
    ):
        _fail("confirmation labels do not match the intact teacher")
    lock_path = out_dir / "proposal_lock.json"
    if file_sha256(lock_path) != report.get("lock_sha256"):
        _fail("lock file hash does not match the report")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("confirmation_used_for_selection") is not False or lock.get("heldout_used_for_selection") is not False:
        _fail("lock says confirmation or held-out data was used for selection")
    if report.get("selection") != lock.get("selection"):
        _fail("duplicated report selection does not match the lock")
    if report.get("confirmation", {}).get("used_for_selection") is not False:
        _fail("confirmation record says it was used for selection")
    selection = lock["selection"]
    if list(selection.get("methods", {})) != list(METHODS) or list(selection.get("finalists", {})) != list(METHODS):
        _fail("method set does not match the study")
    loss = loss_sensitive_projection_direction(
        practice["A"], practice["b"], practice["W"], practice["c"],
        practice["x_practice_target"], practice["y_practice_target"],
    )
    recorded_direction = selection.get("direction", {})
    if loss.get("status") != "ok":
        if selection.get("status") != loss["status"]:
            _fail("direction failure was not recorded")
        for method in METHODS:
            if selection["finalists"][method] != {"status": loss["status"]}:
                _fail("a failed direction still has a finalist")
            if selection["methods"][method].get("slots"):
                _fail("a failed direction still has slots")
    else:
        vector = np.asarray(loss["direction"], dtype=np.float64)
        if recorded_direction.get("direction_f64_hex") != f64_hex(vector):
            _fail("loss-sensitive direction does not match recomputation")
        if recorded_direction.get("full_matrix_derivative_f64_hex") != _scalar_hex(float(loss["full_matrix_derivative"])):
            _fail("full-matrix derivative does not match recomputation")
        gradient = np.asarray(loss["gradient"], dtype=np.float64)
        products = column_products(vector, gradient, practice["W"])
        if selection.get("contributions_f64_hex") != f64_hex(products["contributions"]):
            _fail("signed contributions do not match recomputation")
        if selection.get("magnitudes_f64_hex") != f64_hex(products["magnitudes"]):
            _fail("projection magnitudes do not match recomputation")
        if selection.get("contribution_tau_f64_hex") != _scalar_hex(float(products["tau"])):
            _fail("contribution tolerance does not match recomputation")
        magnitude = rank_magnitude(practice["W"], vector, support_size=2)
        signed = rank_signed_contributions(vector, gradient, practice["W"], support_size=2)
        plans = {"magnitude_projection": magnitude, "signed_loss_contribution": signed}
        _replay_methods(out_dir, selection, practice, vector, gradient, plans)
    _replay_confirmation(out_dir, report, lock, practice)
    return {
        "status": "verified",
        "study_status": report["status"],
        "wall_seconds": timing["wall_seconds"],
        "comparison": report.get("comparison"),
    }


def _replay_methods(out_dir, selection, practice, vector, gradient, plans) -> None:
    baseline_target = predict(
        practice["A"], practice["b"], practice["W"], practice["c"], practice["x_practice_target"]
    )
    baseline_preserve = predict(
        practice["A"], practice["b"], practice["W"], practice["c"], practice["x_practice_preserve"]
    )
    source = out_dir / "proposal_inputs" / "damaged_source.npz"
    for method in METHODS:
        payload = selection["methods"][method]
        plan = plans[method]
        finalist = selection["finalists"][method]
        if plan["status"] != "ok":
            if payload.get("status") != plan["status"] or payload.get("slots") or payload.get("columns"):
                _fail(f"{method} did not record the support failure")
            if finalist != {"status": plan["status"]}:
                _fail(f"{method} failure finalist does not match the ranking")
            continue
        if payload.get("columns") != plan["columns"]:
            _fail(f"{method} columns do not match the frozen ranking")
        if payload.get("diagnostic_affects_selection") is not False:
            _fail(f"{method} diagnostic was marked as a selection input")
        slots = payload.get("slots")
        if not isinstance(slots, list) or len(slots) != len(DOSES):
            _fail(f"{method} did not keep the eight dose slots")
        for index, (slot, dose) in enumerate(zip(slots, DOSES)):
            if int(slot.get("dose_index")) != index or float(slot.get("dose")) != float(dose):
                _fail(f"{method} dose slot identity changed")
            if slot.get("columns") != plan["columns"]:
                _fail(f"{method} slot changed the ranked columns")
            if slot.get("status") != "applied":
                continue
            bundle = out_dir / "candidates" / slot["bundle_name"]
            if not bundle.is_dir():
                _fail(f"missing candidate bundle for {method}")
            verdict = verify_bundle(bundle, source, TENSOR_NAME, source_path=source)
            if verdict["value_exact_reconstruction"] is not True:
                _fail("reconstruction is not value-exact")
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
            if recorded.get("preservation_flips") != flips:
                _fail(f"practice preservation does not match the saved candidate for {method}")
            if recorded.get("baseline_correct") != counts["baseline_correct"] or recorded.get("candidate_correct") != counts["candidate_correct"]:
                _fail(f"practice counts do not match the saved candidate for {method}")
            eligible = counts["corrections"] > 0 and counts["losses"] == 0 and flips == 0
            if _real_bool(recorded.get("eligible"), f"{method} eligibility") is not eligible:
                _fail(f"practice eligibility does not match the saved candidate for {method}")
            if slot.get("accounting") != verdict["accounting"]:
                _fail(f"slot accounting does not match reconstruction for {method}")
            reconstructed = float(verdict["accounting"]["stored_update_frobenius"])
            relative = abs(reconstructed - float(slot["target_norm"])) / float(slot["target_norm"])
            if relative > MATCH_TOLERANCE:
                _fail(f"stored update norm is outside the frozen tolerance for {method}")
            before = mean_cross_entropy(
                practice["A"], practice["b"], practice["W"], practice["c"],
                practice["x_practice_target"], practice["y_practice_target"],
            )
            after = mean_cross_entropy(
                practice["A"], practice["b"], candidate, practice["c"],
                practice["x_practice_target"], practice["y_practice_target"],
            )
            predicted = predicted_support_change(vector, gradient, practice["W"], plan["columns"], float(slot["strength"]))
            if slot.get("selected_support_derivative_f64_hex") != _scalar_hex(predicted["selected_support_derivative"]):
                _fail(f"selected-support derivative does not match recomputation for {method}")
            if slot.get("predicted_loss_change_f64_hex") != _scalar_hex(predicted["predicted_loss_change"]):
                _fail(f"predicted loss change does not match recomputation for {method}")
            if slot.get("practice_loss_change_f64_hex") != _scalar_hex(after - before):
                _fail(f"practice loss change does not match recomputation for {method}")
            agreement = sign_agreement(predicted["predicted_loss_change"], after - before)
            if slot.get("sign_status") != agreement["sign_status"]:
                _fail(f"loss-sign record does not match recomputation for {method}")
            if slot.get("diagnostic_affects_selection") is not False:
                _fail(f"{method} slot diagnostic was marked as a selection input")
        expected = _select_method(payload["slots"])
        if finalist.get("status") != expected.get("status"):
            _fail(f"finalist does not match the practice rule for {method}")
        if expected.get("status") == "qualified" and finalist.get("bundle_name") != expected.get("bundle_name"):
            _fail(f"finalist bundle does not match the practice rule for {method}")


def _replay_confirmation(out_dir, report, lock, practice) -> None:
    confirmation = report.get("confirmation")
    if not isinstance(confirmation, dict):
        _fail("confirmation record is missing")
    if confirmation.get("comparison") != report.get("comparison"):
        _fail("comparison fields disagree")
    with np.load(out_dir / "host_only" / "confirmation_bank.npz") as handle:
        bank = {key: np.array(handle[key], copy=True) for key in handle.files}
    teacher = {"A": bank["A"], "b": bank["b"], "c": bank["c"], "W": bank["W_damaged"]}
    baseline_target = predict(teacher["A"], teacher["b"], teacher["W"], teacher["c"], bank["x_confirmation_target"])
    baseline_preserve = predict(
        teacher["A"], teacher["b"], teacher["W"], teacher["c"], bank["x_confirmation_preserve"]
    )
    from .support_experiment import _score_matrix

    no_edit = _score_matrix(teacher, teacher["W"], baseline_target, baseline_preserve, bank)
    recorded_control = confirmation.get("no_edit")
    for key in ("corrections", "losses", "baseline_correct", "candidate_correct", "preservation_flips"):
        if recorded_control.get(key) != no_edit[key]:
            _fail("no-edit confirmation record does not match recomputation")
    if _real_bool(recorded_control.get("behavioral_repair"), "no-edit repair") is not False:
        _fail("no-edit control was marked as a repair")
    replayed = {}
    for method in METHODS:
        finalist = lock["selection"]["finalists"][method]
        recorded = confirmation["methods"][method]
        if finalist.get("status") != "qualified":
            if recorded != {"status": finalist["status"]}:
                _fail(f"{method} confirmation does not match the locked failure")
            replayed[method] = recorded
            continue
        with np.load(out_dir / "candidates" / finalist["bundle_name"] / "candidate.npz") as handle:
            candidate = np.array(handle["W"], copy=True)
        if array_content_sha256(candidate) != finalist["candidate_content_sha256"]:
            _fail("confirmation candidate does not match the lock")
        scored = _score_matrix(teacher, candidate, baseline_target, baseline_preserve, bank)
        for key in (
            "corrections",
            "losses",
            "baseline_correct",
            "candidate_correct",
            "preservation_flips",
            "preservation_corrections",
            "preservation_regressions",
            "preservation_incorrect_to_incorrect",
        ):
            if recorded.get(key) != scored[key]:
                _fail(f"confirmation {key} does not match recomputation for {method}")
        if _real_bool(recorded.get("behavioral_repair"), f"{method} repair") is not scored["behavioral_repair"]:
            _fail(f"confirmation repair flag does not match recomputation for {method}")
        if recorded.get("target_prediction_sha256") != scored["target_prediction_sha256"]:
            _fail(f"confirmation prediction hash does not match recomputation for {method}")
        replayed[method] = {
            "status": "qualified",
            "behavioral_repair": scored["behavioral_repair"],
            "corrections": scored["corrections"],
        }
    if confirmation_comparison(replayed) != report.get("comparison"):
        _fail("comparison label does not match the confirmation rule")
    if practice["W"].shape != (4, 16):
        _fail("practice matrix shape changed")


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "experiments/proxy_support_v3")
    try:
        verdict = verify_study(out)
    except VerificationFailure as exc:
        print(f"status=rejected reason={exc}", file=sys.stderr)
        return 1
    print(
        f"status={verdict['status']} study_status={verdict['study_status']} "
        f"wall_seconds={verdict['wall_seconds']} comparison={verdict.get('comparison')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
