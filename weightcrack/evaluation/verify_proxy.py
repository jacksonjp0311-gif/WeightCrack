"""Replay and verify one proxy_fault_v1 artifact directory.

    python -m weightcrack.evaluation.verify_proxy experiments/proxy_fault_v1

Exit 0 means the saved report, lock, receipts, and predictions agree.
It does not mean a behavioral repair occurred. Exit 1 means tampering,
disagreement, or an incomplete bundle.

A fresh resave boolean is recorded separately from the historical one.
Those booleans are not required to match: ``np.savez`` creator-system
metadata depends on the operating system. Original file hashes and exact
stored tensor values remain mandatory. This module checks the v1 method
set only. Another study uses its own verifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.localization import rank_columns
from ..canonical import array_content_sha256, file_sha256
from ..errors import VerificationFailure
from ..verify import verify_bundle
from .proxy_experiment import (
    CONTROL_SEED,
    MATCH_TOLERANCE,
    MAX_ABS_DELTA,
    MAX_ENTRIES,
    MAX_FRACTION,
    TENSOR_NAME,
    official_config,
    platform_version,
    _json_bytes,
    _recompute_practice,
)
from .proxy_model import predict, prediction_sha256, preservation_flips, split_counts
from .proxy_proposer import METHODS, PRACTICE_KEYS, SOURCE_KEYS

# The v1 protocol compared these two column-selection arms. A later study
# does not inherit this set.
V1_METHODS = ("heuristic", "random")

BANNED_LOCK_KEYS = {
    "fault_seed",
    "fault_columns",
    "fault_direction",
    "fault_direction_f64_hex",
    "fault_strength",
    "model_seed",
    "data_seed",
    "heldout",
    "W_intact",
    "intact_W",
    "x_heldout_target",
    "y_heldout_target",
    "x_heldout_preserve",
    "y_heldout_preserve",
}


def _fail(message: str) -> None:
    raise VerificationFailure("rejected", message)


def _real_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} is not a boolean")
    return value


def _walk_keys(value: Any, banned: set[str], where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in banned:
                _fail(f"{where} contains {key}")
            _walk_keys(item, banned, where)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, banned, where)


def _load_report(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "experiment_report.json"
    if not path.is_file():
        _fail("experiment report is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_budgets(accounting: dict[str, Any]) -> None:
    if accounting["fraction_denominator_definition"] != "edited_tensor_scalar_count":
        _fail("fraction denominator is not the edited tensor")
    if int(accounting["touched_coordinates"]) > MAX_ENTRIES:
        _fail("touched coordinates exceed max_entries")
    fraction = int(accounting["touched_coordinates"]) / int(accounting["fraction_denominator"])
    if fraction > MAX_FRACTION:
        _fail("touched fraction exceeds max_fraction")
    if float(accounting["max_abs_stored_delta"]) > MAX_ABS_DELTA:
        _fail("stored absolute delta exceeds the policy")
    if int(accounting["operation_count"]) != 1:
        _fail("operation count is not one project_remove")


def _historical_resave_flags(report: dict[str, Any], applied: int) -> list[bool]:
    historical = report.get("resave_byte_identical")
    if not isinstance(historical, list) or len(historical) != applied:
        _fail("historical resave observations are missing")
    if any(type(item) is not bool for item in historical):
        _fail("historical resave observations are not booleans")
    return historical


def _audit_practice_records(out_dir: Path, selection: dict[str, Any]) -> None:
    """Recompute every practice count stored on an applied slot or finalist."""
    if selection.get("status") != "ok":
        return
    proposal = out_dir / "proposal_inputs"
    with np.load(proposal / "practice_export.npz") as handle:
        practice = {key: np.array(handle[key], copy=True) for key in handle.files}
    baseline_target = predict(
        practice["A"], practice["b"], practice["W"], practice["c"], practice["x_practice_target"]
    )
    baseline_preserve = predict(
        practice["A"], practice["b"], practice["W"], practice["c"], practice["x_practice_preserve"]
    )
    finalists = selection.get("finalists")
    if not isinstance(finalists, dict) or set(finalists) != set(V1_METHODS):
        _fail("lock finalists are not the v1 method set")
    methods = selection.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(V1_METHODS):
        _fail("lock methods are not the v1 method set")
    for method in V1_METHODS:
        payload = methods[method]
        winner = None
        for slot in payload["slots"]:
            if slot.get("status") != "applied":
                continue
            bundle = out_dir / "candidates" / slot["bundle_name"]
            with np.load(bundle / "candidate.npz") as handle:
                candidate = np.array(handle["W"], copy=True)
            target = predict(practice["A"], practice["b"], candidate, practice["c"], practice["x_practice_target"])
            preserve = predict(
                practice["A"], practice["b"], candidate, practice["c"], practice["x_practice_preserve"]
            )
            counts = split_counts(baseline_target, target, practice["y_practice_target"])
            flips = preservation_flips(baseline_preserve, preserve)
            recorded = slot.get("practice")
            if not isinstance(recorded, dict):
                _fail(f"practice record is missing for {method}")
            expected = {
                "corrections": counts["corrections"],
                "losses": counts["losses"],
                "baseline_correct": counts["baseline_correct"],
                "candidate_correct": counts["candidate_correct"],
                "count": counts["count"],
                "preservation_flips": flips,
                "preservation_count": int(practice["y_practice_preserve"].shape[0]),
                "target_prediction_sha256": prediction_sha256(target),
                "preserve_prediction_sha256": prediction_sha256(preserve),
            }
            for key, value in expected.items():
                if recorded.get(key) != value:
                    _fail(f"practice {key} does not match the saved candidate for {method}")
            eligible = counts["corrections"] > 0 and counts["losses"] == 0 and flips == 0
            if _real_bool(recorded.get("eligible"), f"{method} practice eligibility") is not eligible:
                _fail(f"practice eligibility does not match the saved candidate for {method}")
            if eligible and (winner is None or (
                -int(recorded["corrections"]),
                float(slot["stored_update_frobenius"]),
                int(slot["dose_index"]),
            ) < (
                -int(winner["practice"]["corrections"]),
                float(winner["stored_update_frobenius"]),
                int(winner["dose_index"]),
            )):
                winner = slot
        finalist = finalists[method]
        status = finalist.get("status")
        if type(status) is not str:
            _fail(f"finalist status is not a string for {method}")
        if winner is None:
            if finalist != {"status": "NO_QUALIFIED_CANDIDATE"}:
                _fail(f"lock finalist does not match an empty practice result for {method}")
            continue
        if status != "qualified":
            _fail(f"lock dropped a practice-qualified candidate for {method}")
        if finalist.get("bundle_name") != winner["bundle_name"]:
            _fail(f"lock finalist bundle does not match the practice rule for {method}")
        if finalist.get("practice") != winner["practice"]:
            _fail(f"lock finalist practice record does not match the winning slot for {method}")


def _score_heldout_matrix(secret, matrix, baseline_target, baseline_preserve) -> dict[str, Any]:
    target = predict(secret["A"], secret["b"], matrix, secret["c"], secret["x_heldout_target"])
    preserve = predict(secret["A"], secret["b"], matrix, secret["c"], secret["x_heldout_preserve"])
    counts = split_counts(baseline_target, target, secret["y_heldout_target"])
    flips = preservation_flips(baseline_preserve, preserve)
    return {
        "counts": counts,
        "flips": flips,
        "target": target,
        "preserve": preserve,
        "preservation_count": int(secret["x_heldout_preserve"].shape[0]),
        "repair": counts["corrections"] > 0 and counts["losses"] == 0 and flips == 0,
    }


def _expect_heldout_fields(scored: dict[str, Any], replay: dict[str, Any], label: str) -> None:
    counts = replay["counts"]
    for key in ("corrections", "losses", "baseline_correct", "candidate_correct", "count"):
        if scored.get(key) != counts[key]:
            _fail(f"held-out {key} disagrees for {label}")
    if scored.get("preservation_flips") != replay["flips"]:
        _fail(f"held-out preservation flips disagree for {label}")
    if scored.get("preservation_count") != replay["preservation_count"]:
        _fail(f"held-out preservation count disagrees for {label}")
    if scored.get("target_prediction_sha256") != prediction_sha256(replay["target"]):
        _fail(f"held-out target prediction hash disagrees for {label}")
    if scored.get("preserve_prediction_sha256") != prediction_sha256(replay["preserve"]):
        _fail(f"held-out preservation prediction hash disagrees for {label}")
    if _real_bool(scored.get("behavioral_repair"), f"{label} behavioral repair") is not replay["repair"]:
        _fail(f"behavioral repair flag disagrees for {label}")


def _replay_heldout(
    out_dir: Path,
    report: dict[str, Any],
    lock: dict[str, Any],
    verdicts: dict[str, dict[str, Any]],
) -> None:
    with np.load(out_dir / "host_only" / "host_secret.npz") as handle:
        secret = {key: np.array(handle[key], copy=True) for key in handle.files}
    meta = json.loads((out_dir / "host_only" / "host_secret.json").read_text(encoding="utf-8"))
    if report["host_disclosure"] != meta:
        _fail("host disclosure does not match the secret record")
    if list(secret["fault_columns"].tolist()) != list(meta["fault_columns"]):
        _fail("secret fault columns do not match the disclosure")
    if array_content_sha256(secret["W_damaged"]) != meta["damaged_W_sha256"]:
        _fail("damaged source identity does not match the disclosure")
    if array_content_sha256(secret["W_intact"]) != meta["intact_W_sha256"]:
        _fail("intact teacher identity does not match the disclosure")
    with np.load(out_dir / "proposal_inputs" / "damaged_source.npz") as handle:
        source_W = np.array(handle["W"], copy=True)
    if not np.array_equal(source_W, secret["W_damaged"]):
        _fail("rollback source is not the damaged matrix")
    baseline_target = predict(secret["A"], secret["b"], source_W, secret["c"], secret["x_heldout_target"])
    baseline_preserve = predict(secret["A"], secret["b"], source_W, secret["c"], secret["x_heldout_preserve"])
    recorded = report["heldout"]["no_edit"]
    no_edit = _score_heldout_matrix(secret, source_W, baseline_target, baseline_preserve)
    if no_edit["counts"]["corrections"] != 0 or no_edit["flips"] != 0:
        _fail("replayed no-edit control moved")
    _expect_heldout_fields(recorded, no_edit, "no-edit")
    heldout = report["heldout"]
    if heldout.get("used_for_selection") is not False or heldout.get("opened_after_lock") is not True:
        _fail("held-out selection flags are not real booleans")
    methods = heldout.get("methods")
    if not isinstance(methods, dict):
        _fail("held-out method table is missing")
    selection = lock["selection"]
    if selection.get("status") != "ok":
        if methods:
            _fail("report scores methods the lock did not run")
        return
    if set(methods) != set(V1_METHODS):
        _fail("held-out method set is not the v1 method set")
    finalists = selection["finalists"]
    for method in V1_METHODS:
        finalist = finalists[method]
        scored = methods[method]
        if type(scored.get("status")) is not str or scored["status"] != finalist.get("status"):
            _fail(f"reported finalist status does not match the lock for {method}")
        if finalist.get("status") == "NO_QUALIFIED_CANDIDATE":
            if scored != {"status": "NO_QUALIFIED_CANDIDATE"} or finalist != {"status": "NO_QUALIFIED_CANDIDATE"}:
                _fail(f"unqualified method carries another outcome for {method}")
            continue
        if finalist.get("status") != "qualified":
            _fail(f"unknown finalist status for {method}")
        for key in ("bundle_name", "dose_index", "candidate_content_sha256", "candidate_file_sha256"):
            if scored.get(key) != finalist.get(key):
                _fail(f"reported {key} does not match the locked finalist for {method}")
        bundle_name = finalist["bundle_name"]
        if bundle_name not in verdicts:
            _fail(f"locked finalist bundle was not verified for {method}")
        with np.load(out_dir / "candidates" / bundle_name / "candidate.npz") as handle:
            candidate = np.array(handle["W"], copy=True)
        if array_content_sha256(candidate) != finalist["candidate_content_sha256"]:
            _fail(f"locked candidate content does not match the bundle for {method}")
        replay = _score_heldout_matrix(secret, candidate, baseline_target, baseline_preserve)
        _expect_heldout_fields(scored, replay, method)
        verdict = verdicts[bundle_name]
        if scored.get("accounting") != verdict["accounting"] or finalist.get("accounting") != verdict["accounting"]:
            _fail(f"reported accounting does not match reconstruction for {method}")
        _check_budgets(verdict["accounting"])
        reconstructed_norm = float(verdict["accounting"]["stored_update_frobenius"])
        if float(finalist["stored_update_frobenius"]) != reconstructed_norm:
            _fail(f"finalist update norm does not match reconstruction for {method}")
        if scored.get("practice") != finalist.get("practice"):
            _fail(f"duplicated practice record disagrees for {method}")


def verify_study(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    report = _load_report(out_dir)
    if report.get("schema") != "weightcrack.proxy_report.v1":
        _fail("unexpected report schema")
    if report.get("study_id") != "proxy_fault_v1":
        _fail("unexpected study id")
    if tuple(METHODS) != V1_METHODS:
        _fail("v1 proposer method set changed")
    timing = report.get("timing")
    if not isinstance(timing, dict) or "wall_seconds" not in timing or timing.get("limit_seconds") != 60.0:
        _fail("timing evidence is missing")
    if report["status"] == "RESOURCE_LIMIT":
        if report.get("heldout_opened") is True and report.get("invalidated_by_time_limit") is not True:
            _fail("a timed-out study opened held-out data as a result")
        return {"status": "verified", "study_status": "RESOURCE_LIMIT", "wall_seconds": timing["wall_seconds"]}
    if report["status"] == "VERIFICATION_FAILED":
        return {"status": "verified", "study_status": "VERIFICATION_FAILED", "wall_seconds": timing["wall_seconds"]}
    if report["status"] not in {"complete", "NO_LOCALIZATION_SIGNAL"}:
        _fail("unknown study status")
    lock_path = out_dir / "proposal_lock.json"
    if file_sha256(lock_path) != report["lock_sha256"]:
        _fail("lock file hash does not match the report")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("heldout_used_for_selection") is not False:
        _fail("lock says held-out data was used for selection")
    _walk_keys(lock, BANNED_LOCK_KEYS, "proposal lock")
    if report.get("selection") != lock.get("selection"):
        _fail("duplicated report selection does not match the lock")
    config = json.loads((out_dir / "proposal_inputs" / "public_config.json").read_text(encoding="utf-8"))
    if config != json.loads(_json_bytes(official_config()).decode("utf-8")):
        _fail("public config does not match the frozen study")
    _walk_keys(config, BANNED_LOCK_KEYS, "public config")
    if file_sha256(out_dir / "proposal_inputs" / "public_config.json") != report["public_config_sha256"]:
        _fail("public config hash does not match the report")
    if file_sha256(out_dir / "proposal_inputs" / "practice_export.npz") != report["practice_export_sha256"]:
        _fail("practice export hash does not match the report")
    if file_sha256(out_dir / "proposal_inputs" / "damaged_source.npz") != report["source_file_sha256"]:
        _fail("source hash does not match the report")
    with np.load(out_dir / "proposal_inputs" / "practice_export.npz") as handle:
        if tuple(handle.files) != PRACTICE_KEYS and set(handle.files) != set(PRACTICE_KEYS):
            _fail("practice export is not the allowlist")
        if set(handle.files) != set(PRACTICE_KEYS):
            _fail("practice export is not the allowlist")
    with np.load(out_dir / "proposal_inputs" / "damaged_source.npz") as handle:
        if set(handle.files) != set(SOURCE_KEYS):
            _fail("source file tensor set changed")
        source_W = np.array(handle["W"], copy=True)
        if source_W.dtype != np.dtype(np.float32):
            _fail("source dtype was not preserved")
    _recompute_practice(out_dir, lock["selection"])
    _audit_practice_records(out_dir, lock["selection"])
    events = [
        json.loads(line)
        for line in (out_dir / "study_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = [event["stage"] for event in events]
    if "lock_written" not in stages or "heldout_opened" not in stages:
        _fail("event record is missing the lock or held-out stage")
    if stages.index("lock_written") > stages.index("heldout_opened"):
        _fail("held-out stage was recorded before the lock")
    applied = 0
    byte_flags = []
    verdicts: dict[str, dict[str, Any]] = {}
    if lock["selection"]["status"] == "ok":
        with np.load(out_dir / "proposal_inputs" / "practice_export.npz") as handle:
            practice_W = np.array(handle["W"], copy=True)
            x_target = np.array(handle["x_practice_target"], copy=True)
            y_target = np.array(handle["y_practice_target"], copy=True)
            A = np.array(handle["A"], copy=True)
            b = np.array(handle["b"], copy=True)
            c = np.array(handle["c"], copy=True)
        from .proxy_model import derive_direction

        derived = derive_direction(A, b, practice_W, c, x_target, y_target)
        ranked = rank_columns(practice_W, derived["direction"], top_k=2)
        if ranked["columns"] != lock["selection"]["methods"]["heuristic"]["columns"]:
            _fail("heuristic columns do not match the practice ranking")
        control = np.random.default_rng(CONTROL_SEED)
        random_columns = [int(column) for column in control.choice(practice_W.shape[1], size=2, replace=False)]
        if random_columns != lock["selection"]["methods"]["random"]["columns"]:
            _fail("random columns do not match the frozen control seed")
        for method, payload in lock["selection"]["methods"].items():
            for slot in payload["slots"]:
                if slot["status"] != "applied":
                    continue
                applied += 1
                bundle = out_dir / "candidates" / slot["bundle_name"]
                verdict = verify_bundle(
                    bundle,
                    out_dir / "proposal_inputs" / "damaged_source.npz",
                    TENSOR_NAME,
                    source_path=out_dir / "proposal_inputs" / "damaged_source.npz",
                )
                if verdict["value_exact_reconstruction"] is not True:
                    _fail("reconstruction is not value-exact")
                flag = verdict["resave_byte_identical"]
                if type(flag) is not bool:
                    _fail("current resave observation is not a boolean")
                byte_flags.append(flag)
                verdicts[slot["bundle_name"]] = verdict
                _check_budgets(verdict["accounting"])
                with np.load(bundle / "candidate.npz") as handle:
                    saved = np.array(handle["W"], copy=True)
                if saved.dtype != source_W.dtype:
                    _fail("saved tensor dtype changed")
                edited = set(slot["columns"])
                for column in range(saved.shape[1]):
                    if column in edited:
                        continue
                    if not np.array_equal(saved[:, column], source_W[:, column]):
                        _fail("an unedited column changed")
                reconstructed_norm = float(verdict["accounting"]["stored_update_frobenius"])
                if float(slot["stored_update_frobenius"]) != reconstructed_norm:
                    _fail("slot update norm does not match reconstruction")
                relative = abs(reconstructed_norm - float(slot["target_norm"])) / float(slot["target_norm"])
                if relative > MATCH_TOLERANCE:
                    _fail("stored update norm is outside the frozen tolerance")
    historical_flags = _historical_resave_flags(report, applied)
    reported_applied = report.get("verified_applied_bundles")
    if type(reported_applied) is not int or reported_applied != applied:
        _fail("reported applied bundle count does not match the lock")
    if applied:
        if report.get("value_exact_reconstruction") is not True:
            _fail("report does not record value-exact reconstruction")
    elif report.get("value_exact_reconstruction") is not None:
        _fail("report records reconstruction without an applied bundle")
    _replay_heldout(out_dir, report, lock, verdicts)
    return {
        "status": "verified",
        "study_status": report["status"],
        "applied_bundles": applied,
        "wall_seconds": timing["wall_seconds"],
        "value_exact_reconstruction": True if applied else None,
        "historical_resave_byte_identical": historical_flags,
        "current_resave_byte_identical": byte_flags,
        "current_runtime": {"python": platform_version(), "numpy": np.__version__},
    }


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "experiments/proxy_fault_v1")
    try:
        verdict = verify_study(out)
    except VerificationFailure as exc:
        print(f"status=rejected detail={exc}")
        return 1
    except Exception as exc:
        print(f"status=rejected detail={type(exc).__name__}: {exc}")
        return 1
    current = verdict.get("current_resave_byte_identical")
    historical = verdict.get("historical_resave_byte_identical")
    print(
        f"status={verdict['status']} study_status={verdict['study_status']} "
        f"wall_seconds={verdict['wall_seconds']}"
    )
    if current is not None:
        print(
            "current_resave_byte_identical="
            + json.dumps(current)
            + " historical_resave_byte_identical="
            + json.dumps(historical)
            + " resave_matches_historical="
            + json.dumps(current == historical)
        )
        runtime = verdict.get("current_runtime")
        if isinstance(runtime, dict):
            print(f"python={runtime.get('python')} numpy={runtime.get('numpy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
