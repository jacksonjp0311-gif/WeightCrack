"""Replay and verify one proxy_fault_v1 artifact directory.

    python -m weightcrack.evaluation.verify_proxy experiments/proxy_fault_v1

Exit 0 means the saved report, lock, receipts, and predictions agree.
It does not mean a behavioral repair occurred. Exit 1 means tampering,
disagreement, or an incomplete bundle.
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
    _json_bytes,
    _recompute_practice,
)
from .proxy_model import predict, prediction_sha256, preservation_flips, split_counts
from .proxy_proposer import PRACTICE_KEYS, SOURCE_KEYS

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


def _replay_heldout(out_dir: Path, report: dict[str, Any]) -> None:
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
    counts = split_counts(baseline_target, baseline_target, secret["y_heldout_target"])
    if counts["corrections"] != 0 or preservation_flips(baseline_preserve, baseline_preserve) != 0:
        _fail("replayed no-edit control moved")
    if recorded["corrections"] != 0 or recorded["losses"] != 0 or recorded["preservation_flips"] != 0:
        _fail("report claims the no-edit control changed predictions")
    if recorded["baseline_correct"] != counts["baseline_correct"]:
        _fail("held-out baseline accuracy does not replay")
    if prediction_sha256(baseline_target) != recorded["target_prediction_sha256"]:
        _fail("held-out baseline prediction hash does not replay")
    for method, scored in report["heldout"]["methods"].items():
        if scored["status"] != "qualified":
            if scored["status"] != "NO_QUALIFIED_CANDIDATE":
                _fail("unknown finalist status")
            continue
        with np.load(out_dir / "candidates" / scored["bundle_name"] / "candidate.npz") as handle:
            candidate = np.array(handle["W"], copy=True)
        target = predict(secret["A"], secret["b"], candidate, secret["c"], secret["x_heldout_target"])
        preserve = predict(secret["A"], secret["b"], candidate, secret["c"], secret["x_heldout_preserve"])
        replay = split_counts(baseline_target, target, secret["y_heldout_target"])
        flips = preservation_flips(baseline_preserve, preserve)
        if replay["corrections"] != scored["corrections"] or replay["losses"] != scored["losses"] or flips != scored["preservation_flips"]:
            _fail(f"held-out replay disagrees for {method}")
        repair = replay["corrections"] > 0 and replay["losses"] == 0 and flips == 0
        if bool(scored["behavioral_repair"]) != repair:
            _fail(f"behavioral repair flag disagrees for {method}")
        _check_budgets(scored["accounting"])


def verify_study(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    report = _load_report(out_dir)
    if report.get("schema") != "weightcrack.proxy_report.v1":
        _fail("unexpected report schema")
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
                byte_flags.append(bool(verdict["resave_byte_identical"]))
                _check_budgets(verdict["accounting"])
                with np.load(bundle / "candidate.npz") as handle:
                    saved = np.array(handle["W"], copy=True)
                if saved.dtype != source_W.dtype:
                    _fail("saved tensor dtype changed")
                # Untouched columns outside the declared support must match the source.
                edited = set(slot["columns"])
                for column in range(saved.shape[1]):
                    if column in edited:
                        continue
                    if not np.array_equal(saved[:, column], source_W[:, column]):
                        _fail("an unedited column changed")
                relative = abs(float(slot["stored_update_frobenius"]) - float(slot["target_norm"])) / float(slot["target_norm"])
                if relative > MATCH_TOLERANCE:
                    _fail("stored update norm is outside the frozen tolerance")
    if report.get("resave_byte_identical") != byte_flags:
        _fail("byte-identity flags do not match a fresh verification")
    _replay_heldout(out_dir, report)
    return {
        "status": "verified",
        "study_status": report["status"],
        "applied_bundles": applied,
        "wall_seconds": timing["wall_seconds"],
        "value_exact_reconstruction": True if applied else None,
        "resave_byte_identical": byte_flags,
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
    print(
        f"status={verdict['status']} study_status={verdict['study_status']} "
        f"wall_seconds={verdict['wall_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
