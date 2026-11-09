"""Frozen comparison of two practice directions on the v1 damaged model.

Constants and the direction formula were fixed before the official candidate
trials. Do not edit them after seeing those trials. The proposer subprocess
receives only the practice export, the public config, the damaged source,
the candidates directory, and a deadline.

Reproduce:
    python -m weightcrack.evaluation.direction_experiment experiments/proxy_direction_v2

Verify:
    python -m weightcrack.evaluation.verify_direction experiments/proxy_direction_v2

Exit 0 means the protocol finished, including a negative or empty result.
Exit 2 means the wall-time ceiling was hit. Exit 1 means a harness or
verification failure. A file write is not a behavioral repair.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..bundle import exclusive_create_bytes
from ..canonical import array_content_sha256, file_sha256
from ..errors import VerificationFailure
from .direction_proposer import METHODS, STUDY_ID
from .proxy_experiment import (
    DOSES,
    MATCH_TOLERANCE,
    SECTOR,
    SUPPORT_SIZE,
    TENSOR_NAME,
    WALL_LIMIT_S,
    official_policy,
    platform_version,
    _append_event,
    _json_bytes,
    _save_npz,
)
from .proxy_model import predict, prediction_sha256, split_counts

CONFIRMATION_SEED = 1112
CONFIRMATION_TARGET = 256
CONFIRMATION_PRESERVE = 256
QUESTION = (
    "Does a direction informed by supervised practice loss produce a better "
    "compensating edit than the existing difference between mean incorrect "
    "and mean correct logits?"
)
PRACTICE_EXPORT_SHA256 = "43564c2ce9e4a973a517d4eb887bd75c377d72f0785ab4e2692e14cc659d4535"
SOURCE_FILE_SHA256 = "9bb3173af038aa8d58070fd28c3c5c6811e5afe0dbc2ef309dd44c3aebad2728"
DAMAGED_W_SHA256 = "0f264954eccef5288d489cbfba305a80a0f8635d476e30b0739d35e338c59e5e"
INTACT_W_SHA256 = "c8da464c0dc7412cc928e10fd61b00f544a7897a65d669b226c6997fcf2bc5a7"
PROTECTED_DIRECTORIES = {"proxy_fault_v1", "synthetic_v1", "synthetic_strong", "synthetic_search"}
ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "experiments" / "proxy_fault_v1"


def official_config() -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "question": QUESTION,
        "tensor_name": TENSOR_NAME,
        "sector": SECTOR,
        "activation_site": "post_linear_logits",
        "storage_dtype": "float32",
        "working_dtype": "float64",
        "shapes": {"A": [16, 8], "b": [16], "W": [4, 16], "c": [4], "x": [8]},
        "doses": list(DOSES),
        "match_tolerance": MATCH_TOLERANCE,
        "support_size": SUPPORT_SIZE,
        "population": {
            "practice_target": 128,
            "practice_preserve": 128,
            "confirmation_target": CONFIRMATION_TARGET,
            "confirmation_preserve": CONFIRMATION_PRESERVE,
        },
        "confirmation_seed": CONFIRMATION_SEED,
        "confirmation_generator": (
            "numpy.random.default_rng(1112).standard_normal; target rows, then "
            "preservation rows; one cast to float32"
        ),
        "source_practice_identities": {
            "practice_export_sha256": PRACTICE_EXPORT_SHA256,
            "source_file_sha256": SOURCE_FILE_SHA256,
            "damaged_W_sha256": DAMAGED_W_SHA256,
            "intact_W_sha256": INTACT_W_SHA256,
        },
        "methods": list(METHODS),
        "directions": {
            "mean_logit_difference": (
                "normalized mean damaged post_linear_logits on incorrect practice "
                "targets minus the mean on correct practice targets"
            ),
            "loss_sensitive_projection": (
                "unit leading eigenvector of M = (G @ W.T + W @ G.T) / 2, where G is "
                "the mean cross-entropy gradient on the practice targets"
            ),
        },
        "ranking": "absolute projection onto that method's direction; top two columns",
        "selection_rule": (
            "positive practice target corrections, zero practice target losses, "
            "zero practice preservation flips; then more corrections, smaller "
            "stored update norm, earlier dose index"
        ),
        "confirmation_rule": (
            "positive confirmation-target corrections, zero losses of previously "
            "correct confirmation targets, zero confirmation preservation flips"
        ),
        "comparison_rule": (
            "A method confirms only when its locked finalist meets the confirmation "
            "rule. If both confirm, more target corrections decides; an equal count "
            "is NO_CONFIRMATION_WINNER. If neither confirms, the result is "
            "NEITHER_METHOD_CONFIRMED. This comparison does not create candidates."
        ),
        "eigenvalue_rule": "tau = 1e-12 * max(1, frobenius_norm(M)); largest and gap must exceed tau",
        "maximum_candidate_slots": 16,
        "wall_limit_s": WALL_LIMIT_S,
        "policy": official_policy().model_dump(mode="json"),
    }


def draw_confirmation_inputs(seed: int = CONFIRMATION_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Draw the precommitted confirmation bank. Target rows come first."""
    generator = np.random.default_rng(seed)
    target = generator.standard_normal((CONFIRMATION_TARGET, 8)).astype(np.float32)
    preserve = generator.standard_normal((CONFIRMATION_PRESERVE, 8)).astype(np.float32)
    return target, preserve


def preservation_change_kinds(
    baseline: np.ndarray,
    candidate: np.ndarray,
    teacher: np.ndarray,
) -> dict[str, int]:
    """Split prediction changes into corrections, regressions, and wrong-to-wrong."""
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    teacher = np.asarray(teacher)
    changed = baseline != candidate
    base_ok = baseline == teacher
    new_ok = candidate == teacher
    corrections = int(np.count_nonzero(changed & ~base_ok & new_ok))
    regressions = int(np.count_nonzero(changed & base_ok & ~new_ok))
    incorrect = int(np.count_nonzero(changed & ~base_ok & ~new_ok))
    flips = int(np.count_nonzero(changed))
    if corrections + regressions + incorrect != flips:
        raise VerificationFailure("rejected", "preservation change kinds do not add up")
    return {
        "flips": flips,
        "corrections": corrections,
        "regressions": regressions,
        "incorrect_to_incorrect": incorrect,
    }


def confirmation_comparison(methods: dict[str, Any]) -> str:
    """Precommitted reading of the two confirmation results. It does not select candidates."""
    confirmed = {}
    corrections = {}
    for method in METHODS:
        scored = methods[method]
        if scored.get("status") != "qualified" or scored.get("behavioral_repair") is not True:
            confirmed[method] = False
            continue
        confirmed[method] = True
        corrections[method] = int(scored["corrections"])
    loss_name, mean_name = "loss_sensitive_projection", "mean_logit_difference"
    if confirmed[loss_name] and confirmed[mean_name]:
        if corrections[loss_name] > corrections[mean_name]:
            return "LOSS_SENSITIVE_MORE_CORRECTIONS"
        if corrections[mean_name] > corrections[loss_name]:
            return "MEAN_LOGIT_MORE_CORRECTIONS"
        return "NO_CONFIRMATION_WINNER"
    if confirmed[loss_name]:
        return "LOSS_SENSITIVE_CONFIRMED"
    if confirmed[mean_name]:
        return "MEAN_LOGIT_CONFIRMED"
    return "NEITHER_METHOD_CONFIRMED"


def _timing(started: float) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    return {
        "wall_seconds": elapsed,
        "limit_seconds": WALL_LIMIT_S,
        "exceeded": elapsed > WALL_LIMIT_S,
        "clock": "time.perf_counter",
    }


def _check_source_pins(practice_path: Path, source_path: Path, teacher: dict[str, np.ndarray]) -> None:
    if file_sha256(practice_path) != PRACTICE_EXPORT_SHA256:
        raise VerificationFailure("rejected", "practice export is not the frozen v1 file")
    if file_sha256(source_path) != SOURCE_FILE_SHA256:
        raise VerificationFailure("rejected", "damaged source is not the frozen v1 file")
    if array_content_sha256(teacher["W_damaged"]) != DAMAGED_W_SHA256:
        raise VerificationFailure("rejected", "damaged W is not the frozen v1 matrix")
    if array_content_sha256(teacher["W_intact"]) != INTACT_W_SHA256:
        raise VerificationFailure("rejected", "intact teacher is not the frozen v1 matrix")


def _load_teacher() -> dict[str, np.ndarray]:
    secret = V1_DIR / "host_only" / "host_secret.npz"
    with np.load(secret) as handle:
        teacher = {key: np.array(handle[key], copy=True) for key in ("A", "b", "c", "W_intact", "W_damaged")}
    meta = json.loads((V1_DIR / "host_only" / "host_secret.json").read_text(encoding="utf-8"))
    if meta["intact_W_sha256"] != INTACT_W_SHA256 or meta["damaged_W_sha256"] != DAMAGED_W_SHA256:
        raise VerificationFailure("rejected", "v1 disclosure hashes do not match the frozen pins")
    if list(meta["fault_columns"]) != [10, 7] or float(meta["fault_strength"]) != 1.0:
        raise VerificationFailure("rejected", "v1 fault disclosure does not match the frozen study")
    return teacher


def _write_inputs(out_dir: Path, teacher: dict[str, np.ndarray]) -> dict[str, Path]:
    proposal = out_dir / "proposal_inputs"
    host = out_dir / "host_only"
    candidates = out_dir / "candidates"
    for path in (proposal, host, candidates):
        path.mkdir(parents=True, exist_ok=False)
    practice_src = V1_DIR / "proposal_inputs" / "practice_export.npz"
    source_src = V1_DIR / "proposal_inputs" / "damaged_source.npz"
    practice = proposal / "practice_export.npz"
    source = proposal / "damaged_source.npz"
    exclusive_create_bytes(practice, practice_src.read_bytes())
    exclusive_create_bytes(source, source_src.read_bytes())
    _check_source_pins(practice, source, teacher)
    with np.load(practice) as handle:
        copied = {key: np.array(handle[key], copy=True) for key in ("A", "b", "c", "W")}
    if not np.array_equal(copied["W"], teacher["W_damaged"]):
        raise VerificationFailure("rejected", "copied practice W is not the damaged teacher")
    for key in ("A", "b", "c"):
        if not np.array_equal(copied[key], teacher[key]):
            raise VerificationFailure("rejected", "copied practice parameters do not match the teacher")
    config_path = proposal / "public_config.json"
    exclusive_create_bytes(config_path, _json_bytes(official_config()))
    x_target, x_preserve = draw_confirmation_inputs(CONFIRMATION_SEED)
    y_target = predict(teacher["A"], teacher["b"], teacher["W_intact"], teacher["c"], x_target)
    y_preserve = predict(teacher["A"], teacher["b"], teacher["W_intact"], teacher["c"], x_preserve)
    secret_npz = host / "confirmation_bank.npz"
    _save_npz(
        secret_npz,
        A=teacher["A"],
        b=teacher["b"],
        c=teacher["c"],
        W_intact=teacher["W_intact"],
        W_damaged=teacher["W_damaged"],
        x_confirmation_target=x_target,
        y_confirmation_target=y_target,
        x_confirmation_preserve=x_preserve,
        y_confirmation_preserve=y_preserve,
    )
    v1_meta = json.loads((V1_DIR / "host_only" / "host_secret.json").read_text(encoding="utf-8"))
    secret_json = host / "confirmation_bank.json"
    disclosure = {
        "schema": "weightcrack.direction_secret.v1",
        "source_study": "proxy_fault_v1",
        "known_damaged_model": True,
        "blinded_fault_replication": False,
        "fault_columns": list(v1_meta["fault_columns"]),
        "fault_strength": v1_meta["fault_strength"],
        "intact_W_sha256": INTACT_W_SHA256,
        "damaged_W_sha256": DAMAGED_W_SHA256,
        "confirmation_seed": CONFIRMATION_SEED,
        "confirmation_target_sha256": array_content_sha256(x_target),
        "confirmation_preserve_sha256": array_content_sha256(x_preserve),
        "confirmation_target_label_sha256": array_content_sha256(y_target),
        "confirmation_preserve_label_sha256": array_content_sha256(y_preserve),
        "label_rule": "argmax of the intact teacher on confirmation inputs drawn with seed 1112",
        "process_boundary": (
            "The proposer subprocess arguments are the practice export, public config, "
            "damaged source, candidates directory, and a deadline. The confirmation bank "
            "is not an argument. Process separation is not a filesystem sandbox."
        ),
    }
    exclusive_create_bytes(secret_json, _json_bytes(disclosure))
    events = out_dir / "study_events.jsonl"
    _append_event(events, {"stage": "frozen", "study_id": STUDY_ID})
    return {
        "practice": practice,
        "config": config_path,
        "source": source,
        "candidates": candidates,
        "secret_npz": secret_npz,
        "secret_json": secret_json,
        "events": events,
    }


def _score_matrix(teacher, matrix, baseline_target, baseline_preserve, bank) -> dict[str, Any]:
    target = predict(teacher["A"], teacher["b"], matrix, teacher["c"], bank["x_confirmation_target"])
    preserve = predict(teacher["A"], teacher["b"], matrix, teacher["c"], bank["x_confirmation_preserve"])
    counts = split_counts(baseline_target, target, bank["y_confirmation_target"])
    kinds = preservation_change_kinds(baseline_preserve, preserve, bank["y_confirmation_preserve"])
    repair = counts["corrections"] > 0 and counts["losses"] == 0 and kinds["flips"] == 0
    return {
        "corrections": counts["corrections"],
        "losses": counts["losses"],
        "baseline_correct": counts["baseline_correct"],
        "candidate_correct": counts["candidate_correct"],
        "count": counts["count"],
        "preservation_flips": kinds["flips"],
        "preservation_corrections": kinds["corrections"],
        "preservation_regressions": kinds["regressions"],
        "preservation_incorrect_to_incorrect": kinds["incorrect_to_incorrect"],
        "preservation_count": int(bank["x_confirmation_preserve"].shape[0]),
        "behavioral_repair": repair,
        "target_prediction_sha256": prediction_sha256(target),
        "preserve_prediction_sha256": prediction_sha256(preserve),
    }


def evaluate(out_dir: Path) -> dict[str, Any]:
    """Score locked finalists once. Refuses to score when the lock is absent."""
    out_dir = Path(out_dir)
    lock_path = out_dir / "proposal_lock.json"
    if not lock_path.is_file():
        raise VerificationFailure("incomplete", "proposal lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    bank_path = out_dir / "host_only" / "confirmation_bank.npz"
    if file_sha256(bank_path) != lock["confirmation_bank_sha256"]:
        raise VerificationFailure("rejected", "confirmation bank changed after the lock")
    with np.load(bank_path) as handle:
        bank = {key: np.array(handle[key], copy=True) for key in handle.files}
    teacher = {"A": bank["A"], "b": bank["b"], "c": bank["c"], "W": bank["W_damaged"]}
    baseline_target = predict(teacher["A"], teacher["b"], teacher["W"], teacher["c"], bank["x_confirmation_target"])
    baseline_preserve = predict(
        teacher["A"], teacher["b"], teacher["W"], teacher["c"], bank["x_confirmation_preserve"]
    )
    confirmation: dict[str, Any] = {
        "opened_after_lock": True,
        "used_for_selection": False,
        "no_edit": _score_matrix(teacher, teacher["W"], baseline_target, baseline_preserve, bank),
        "methods": {},
    }
    no_edit = confirmation["no_edit"]
    if no_edit["corrections"] != 0 or no_edit["losses"] != 0 or no_edit["preservation_flips"] != 0:
        raise VerificationFailure("rejected", "no-edit control does not reproduce the damaged model")
    for method in METHODS:
        finalist = lock["selection"]["finalists"][method]
        if finalist.get("status") != "qualified":
            confirmation["methods"][method] = {"status": finalist["status"]}
            continue
        with np.load(out_dir / "candidates" / finalist["bundle_name"] / "candidate.npz") as handle:
            candidate = np.array(handle["W"], copy=True)
        if array_content_sha256(candidate) != finalist["candidate_content_sha256"]:
            raise VerificationFailure("rejected", "finalist hash does not match the locked candidate")
        scored = _score_matrix(teacher, candidate, baseline_target, baseline_preserve, bank)
        scored["status"] = "qualified"
        scored["dose_index"] = finalist["dose_index"]
        scored["bundle_name"] = finalist["bundle_name"]
        scored["candidate_content_sha256"] = finalist["candidate_content_sha256"]
        scored["candidate_file_sha256"] = finalist["candidate_file_sha256"]
        scored["accounting"] = finalist["accounting"]
        scored["practice"] = finalist["practice"]
        confirmation["methods"][method] = scored
    confirmation["comparison"] = confirmation_comparison(confirmation["methods"])
    return confirmation


def _base_report(started: float, status: str) -> dict[str, Any]:
    return {
        "schema": "weightcrack.direction_report.v1",
        "study_id": STUDY_ID,
        "status": status,
        "question": QUESTION,
        "timing": _timing(started),
        "runtime": {"python": platform_version(), "numpy": np.__version__},
        "isolation": {
            "implemented": (
                "The proposer is a subprocess whose arguments are the practice export, "
                "public config, damaged source, candidates directory, and a deadline. "
                "The confirmation bank is not an argument."
            ),
            "not_a_sandbox": (
                "Process separation is not a hardened filesystem sandbox. A modified "
                "proposer could open sibling files, including the confirmation bank."
            ),
            "not_a_blinded_replication": (
                "The intact teacher and the original fault are already disclosed. "
                "Confirmation uses fresh inputs for this known damaged model."
            ),
        },
    }


def _slot_budget(selection: dict[str, Any]) -> dict[str, int]:
    recorded = 0
    applied = 0
    rejected = 0
    for method in METHODS:
        payload = selection.get("methods", {}).get(method, {})
        for slot in payload.get("slots", []):
            recorded += 1
            if slot.get("status") == "applied":
                applied += 1
            else:
                rejected += 1
    return {
        "maximum_candidate_slots": 16,
        "slots_recorded": recorded,
        "applied_bundles": applied,
        "rejected_slots": rejected,
        "no_edit_control": 1,
    }


def run(out_dir: Path) -> int:
    out_dir = Path(out_dir)
    if out_dir.name in PROTECTED_DIRECTORIES or out_dir.resolve() == V1_DIR.resolve():
        print(f"{out_dir} is protected evidence. Choose a new directory.", file=sys.stderr)
        return 1
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"refusing to overwrite {out_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report_path = out_dir / "experiment_report.json"
    try:
        teacher = _load_teacher()
        paths = _write_inputs(out_dir, teacher)
        remaining = WALL_LIMIT_S - (time.perf_counter() - started)
        deadline = time.time() + max(0.0, remaining)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "weightcrack.evaluation.direction_proposer",
                    str(paths["practice"]),
                    str(paths["config"]),
                    str(paths["source"]),
                    str(paths["candidates"]),
                    f"{deadline:.6f}",
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=WALL_LIMIT_S + 15,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            report = _base_report(started, "RESOURCE_LIMIT")
            report["timing_evidence"] = {"timeout_expired": True, "stdout": exc.stdout, "stderr": exc.stderr}
            report["confirmation_opened"] = False
            exclusive_create_bytes(report_path, _json_bytes(report))
            return 2
        _append_event(paths["events"], {"stage": "proposer_returned", "code": completed.returncode})
        if completed.returncode not in (0, 2):
            report = _base_report(started, "VERIFICATION_FAILED")
            report["proposer_stdout"] = completed.stdout
            report["proposer_stderr"] = completed.stderr
            report["confirmation_opened"] = False
            exclusive_create_bytes(report_path, _json_bytes(report))
            return 1
        selection_path = paths["candidates"] / "selection.json"
        if not selection_path.is_file():
            raise VerificationFailure("incomplete", "proposer did not write selection.json")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if completed.returncode == 2 or selection.get("status") == "RESOURCE_LIMIT" or _timing(started)["exceeded"]:
            report = _base_report(started, "RESOURCE_LIMIT")
            report["selection_status"] = selection.get("status")
            report["proposer_stdout"] = completed.stdout
            report["confirmation_opened"] = False
            report["budget"] = _slot_budget(selection)
            exclusive_create_bytes(report_path, _json_bytes(report))
            return 2
        lock = {
            "schema": "weightcrack.direction_lock.v1",
            "study_id": STUDY_ID,
            "selection": selection,
            "public_config_sha256": file_sha256(paths["config"]),
            "practice_export_sha256": file_sha256(paths["practice"]),
            "source_file_sha256": file_sha256(paths["source"]),
            "confirmation_bank_sha256": file_sha256(paths["secret_npz"]),
            "confirmation_disclosure_sha256": file_sha256(paths["secret_json"]),
            "confirmation_used_for_selection": False,
            "heldout_used_for_selection": False,
        }
        lock_path = out_dir / "proposal_lock.json"
        exclusive_create_bytes(lock_path, _json_bytes(lock))
        _append_event(paths["events"], {"stage": "lock_written", "lock_sha256": file_sha256(lock_path)})
        if _timing(started)["exceeded"]:
            report = _base_report(started, "RESOURCE_LIMIT")
            report["lock_sha256"] = file_sha256(lock_path)
            report["confirmation_opened"] = False
            report["invalidated_by_time_limit"] = True
            exclusive_create_bytes(report_path, _json_bytes(report))
            return 2
        _append_event(paths["events"], {"stage": "confirmation_opened"})
        confirmation = evaluate(out_dir)
        from ..verify import verify_bundle

        verified = []
        for payload in selection.get("methods", {}).values():
            for slot in payload.get("slots", []):
                if slot.get("status") != "applied":
                    continue
                verified.append(
                    verify_bundle(
                        out_dir / "candidates" / slot["bundle_name"],
                        paths["source"],
                        TENSOR_NAME,
                        source_path=paths["source"],
                    )
                )
        report = _base_report(started, "complete")
        report["lock_sha256"] = file_sha256(lock_path)
        report["source_file_sha256"] = file_sha256(paths["source"])
        report["practice_export_sha256"] = file_sha256(paths["practice"])
        report["public_config_sha256"] = file_sha256(paths["config"])
        report["confirmation_bank_sha256"] = lock["confirmation_bank_sha256"]
        report["confirmation_disclosure_sha256"] = lock["confirmation_disclosure_sha256"]
        report["selection"] = selection
        report["confirmation"] = confirmation
        report["budget"] = _slot_budget(selection)
        report["verified_applied_bundles"] = len(verified)
        report["value_exact_reconstruction"] = (
            all(item["value_exact_reconstruction"] is True for item in verified) if verified else None
        )
        report["resave_byte_identical"] = [bool(item["resave_byte_identical"]) for item in verified]
        report["confirmation_opened"] = True
        report["comparison"] = confirmation["comparison"]
        if report["timing"]["exceeded"]:
            report["status"] = "RESOURCE_LIMIT"
            report["invalidated_by_time_limit"] = True
            exclusive_create_bytes(report_path, _json_bytes(report))
            return 2
        _append_event(paths["events"], {"stage": "report_written", "status": report["status"]})
        exclusive_create_bytes(report_path, _json_bytes(report))
        return 0
    except VerificationFailure as exc:
        report = _base_report(started, "VERIFICATION_FAILED")
        report["error"] = str(exc)
        report["confirmation_opened"] = False
        if not report_path.exists():
            exclusive_create_bytes(report_path, _json_bytes(report))
        return 1
    except Exception as exc:
        report = _base_report(started, "VERIFICATION_FAILED")
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["confirmation_opened"] = False
        if not report_path.exists():
            exclusive_create_bytes(report_path, _json_bytes(report))
        return 1


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "experiments/proxy_direction_v2")
    code = run(out)
    print(f"status_code={code} directory={out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
