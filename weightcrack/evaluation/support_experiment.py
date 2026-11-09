"""Preregistered support comparison on one new synthetic fault.

``prepare`` writes the host files and a hash commitment. It does not build
candidates. ``run`` is the official trial and refuses to redraw those files.
The intact teacher and confirmation labels stay under ``host_only`` until a
later disclosure commit. Process separation is not a filesystem sandbox.

The 60-second ceiling covers ``run`` only.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..bundle import exclusive_create_bytes
from ..canonical import array_content_sha256, f64_hex, file_sha256
from ..errors import VerificationFailure
from ..patches.huggingface import apply_project_remove
from .direction_experiment import preservation_change_kinds
from .proxy_experiment import (
    DOSES,
    FAULT_STRENGTH,
    MATCH_TOLERANCE,
    SECTOR,
    TENSOR_NAME,
    WALL_LIMIT_S,
    _draw_model,
    official_policy,
    platform_version,
)
from .proxy_model import predict, prediction_sha256, split_counts
from .support_proposer import METHODS, STUDY_ID
from .support_rank import CONTRIBUTION_TOLERANCE_SCALE, LOSS_SIGN_TOLERANCE

QUESTION = (
    "Does selecting the two columns with the largest positive predicted loss "
    "reduction produce a better compensating edit than selecting the two columns "
    "with the largest projection magnitude, when both use the same loss-sensitive direction?"
)
POPULATION = {
    "practice_target": 128,
    "practice_preserve": 128,
    "confirmation_target": 256,
    "confirmation_preserve": 256,
}
ROOT = Path(__file__).resolve().parents[2]
PROTECTED_DIRECTORIES = {
    "proxy_fault_v1",
    "proxy_direction_v2",
    "proxy_direction_v2_harness_failure",
    "synthetic_v1",
    "synthetic_strong",
    "synthetic_search",
}
SOURCE_IDENTITY = (
    "weightcrack/evaluation/support_rank.py",
    "weightcrack/evaluation/support_proposer.py",
    "weightcrack/evaluation/support_experiment.py",
)


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
        "support_size": 2,
        "population": dict(POPULATION),
        "methods": list(METHODS),
        "fault_strength": FAULT_STRENGTH,
        "seed_location": (
            "host_only/host_config.json, bound by public_commitment.json before the trial "
            "and disclosed after confirmation"
        ),
        "fault_rule": (
            "numpy Generator(fault_seed).choice of two columns without replacement, "
            "then a normalized standard_normal direction of length 4, then one "
            "project_remove at strength 1.0. The realized columns are host-only."
        ),
        "direction_rule": "leading eigenvector of (G @ W.T + W @ G.T) / 2 with the v2 eigenvalue gate",
        "magnitude_rule": "top two |u.T W[:,j]| with ascending-index ties",
        "signed_rule": (
            "top two a_j = (u.T G[:,j]) * (u.T W[:,j]) among a_j > "
            "1e-12 * max(1, max |a_j|), ascending-index ties; otherwise "
            "INSUFFICIENT_DESCENT_SUPPORT"
        ),
        "contribution_tolerance_scale": CONTRIBUTION_TOLERANCE_SCALE,
        "loss_sign_tolerance": LOSS_SIGN_TOLERANCE,
        "selection_rule": (
            "positive practice target corrections, zero practice target losses, "
            "zero practice preservation flips; then more corrections, smaller "
            "stored update norm, earlier dose index"
        ),
        "confirmation_rule": (
            "positive confirmation target corrections, zero confirmation target "
            "losses, zero confirmation preservation flips"
        ),
        "comparison_rule": (
            "A method confirms only when its locked finalist meets the confirmation "
            "rule. If both confirm, more target corrections decides. An equal count "
            "is NO_CONFIRMATION_WINNER. If neither confirms, the result is "
            "NEITHER_METHOD_CONFIRMED. This comparison does not create candidates."
        ),
        "maximum_candidate_slots": 16,
        "wall_limit_s": WALL_LIMIT_S,
        "policy": official_policy().model_dump(mode="json"),
    }


def draw_practice_inputs(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Practice rows only. This generator does not continue into a held-out bank."""
    generator = np.random.default_rng(seed)
    target = generator.standard_normal((POPULATION["practice_target"], 8)).astype(np.float32)
    preserve = generator.standard_normal((POPULATION["practice_preserve"], 8)).astype(np.float32)
    return target, preserve


def draw_confirmation_inputs(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fresh confirmation bank. Target rows come first. This is not seed 1112."""
    generator = np.random.default_rng(seed)
    target = generator.standard_normal((POPULATION["confirmation_target"], 8)).astype(np.float32)
    preserve = generator.standard_normal((POPULATION["confirmation_preserve"], 8)).astype(np.float32)
    return target, preserve


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
    magnitude, signed = METHODS
    if confirmed[magnitude] and confirmed[signed]:
        if corrections[signed] > corrections[magnitude]:
            return "SIGNED_SUPPORT_MORE_CORRECTIONS"
        if corrections[magnitude] > corrections[signed]:
            return "MAGNITUDE_SUPPORT_MORE_CORRECTIONS"
        return "NO_CONFIRMATION_WINNER"
    if confirmed[signed]:
        return "SIGNED_SUPPORT_CONFIRMED"
    if confirmed[magnitude]:
        return "MAGNITUDE_SUPPORT_CONFIRMED"
    return "NEITHER_METHOD_CONFIRMED"


def source_identity() -> dict[str, str]:
    return {relative: file_sha256(ROOT / relative) for relative in SOURCE_IDENTITY}


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    exclusive_create_bytes(path, buffer.getvalue())


def _append_event(path: Path, record: dict[str, Any]) -> None:
    line = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(str(path), flags)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def _refuse_archive(out_dir: Path) -> bool:
    if out_dir.name in PROTECTED_DIRECTORIES:
        return True
    resolved = out_dir.resolve()
    for name in PROTECTED_DIRECTORIES:
        archive = (ROOT / "experiments" / name).resolve()
        if resolved == archive:
            return True
    return False


def _timing(started: float) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    return {
        "wall_seconds": elapsed,
        "limit_seconds": WALL_LIMIT_S,
        "exceeded": elapsed > WALL_LIMIT_S,
        "clock": "time.perf_counter",
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


def _draw_case(seeds: dict[str, int]) -> dict[str, Any]:
    intact = _draw_model(int(seeds["model_seed"]))
    x_target, x_preserve = draw_practice_inputs(int(seeds["data_seed"]))
    y_target = predict(intact["A"], intact["b"], intact["W"], intact["c"], x_target)
    y_preserve = predict(intact["A"], intact["b"], intact["W"], intact["c"], x_preserve)
    fault_rng = np.random.default_rng(int(seeds["fault_seed"]))
    fault_columns = [int(column) for column in fault_rng.choice(16, size=2, replace=False)]
    fault_direction = fault_rng.standard_normal(4)
    fault_direction = fault_direction / np.linalg.norm(fault_direction)
    damaged_w, fault_accounting = apply_project_remove(
        intact["W"], fault_direction.tolist(), fault_columns, float(FAULT_STRENGTH)
    )
    x_confirmation_target, x_confirmation_preserve = draw_confirmation_inputs(int(seeds["confirmation_seed"]))
    y_confirmation_target = predict(
        intact["A"], intact["b"], intact["W"], intact["c"], x_confirmation_target
    )
    y_confirmation_preserve = predict(
        intact["A"], intact["b"], intact["W"], intact["c"], x_confirmation_preserve
    )
    return {
        "intact": intact,
        "damaged_w": damaged_w,
        "fault_columns": fault_columns,
        "fault_direction": np.asarray(fault_direction, dtype=np.float64),
        "fault_changed_coordinates": fault_accounting["changed_coordinates"],
        "x_practice_target": x_target,
        "x_practice_preserve": x_preserve,
        "y_practice_target": y_target,
        "y_practice_preserve": y_preserve,
        "x_confirmation_target": x_confirmation_target,
        "x_confirmation_preserve": x_confirmation_preserve,
        "y_confirmation_target": y_confirmation_target,
        "y_confirmation_preserve": y_confirmation_preserve,
    }


def _commitment_files(out_dir: Path) -> dict[str, Path]:
    return {
        "host_config.json": out_dir / "host_only" / "host_config.json",
        "intact_model.npz": out_dir / "host_only" / "intact_model.npz",
        "confirmation_bank.npz": out_dir / "host_only" / "confirmation_bank.npz",
        "confirmation_disclosure.json": out_dir / "host_only" / "confirmation_disclosure.json",
        "damaged_source.npz": out_dir / "proposal_inputs" / "damaged_source.npz",
        "practice_export.npz": out_dir / "proposal_inputs" / "practice_export.npz",
        "public_config.json": out_dir / "proposal_inputs" / "public_config.json",
    }


def _load_seeds(path: Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("model_seed", "data_seed", "fault_seed", "confirmation_seed")
    if set(payload) != set(required):
        raise VerificationFailure("rejected", "seed file keys are not the host seed set")
    seeds = {}
    for key in required:
        value = payload[key]
        if type(value) is not int or isinstance(value, bool):
            raise VerificationFailure("rejected", f"{key} is not an integer")
        seeds[key] = value
    if len(set(seeds.values())) != 4:
        raise VerificationFailure("rejected", "host seeds are not four distinct integers")
    forbidden = {1108, 1109, 1110, 1111, 1112}
    if forbidden.intersection(seeds.values()):
        raise VerificationFailure("rejected", "host seeds reuse a v1 or v2 seed")
    return seeds


def prepare(out_dir: Path, seeds_file: Path) -> int:
    """Freeze one case and write its hash commitment. Does not run candidates."""
    out_dir = Path(out_dir)
    if _refuse_archive(out_dir):
        print(f"{out_dir} is protected evidence. Choose a new directory.", file=sys.stderr)
        return 1
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"refusing to overwrite {out_dir}", file=sys.stderr)
        return 1
    proposal = out_dir / "proposal_inputs"
    host = out_dir / "host_only"
    candidates = out_dir / "candidates"
    for path in (proposal, host, candidates):
        path.mkdir(parents=True, exist_ok=True)
    seeds = _load_seeds(seeds_file)
    case = _draw_case(seeds)
    intact = case["intact"]
    damaged = dict(intact)
    damaged["W"] = case["damaged_w"]
    _save_npz(
        proposal / "practice_export.npz",
        A=damaged["A"],
        b=damaged["b"],
        W=damaged["W"],
        c=damaged["c"],
        x_practice_target=case["x_practice_target"],
        y_practice_target=case["y_practice_target"],
        x_practice_preserve=case["x_practice_preserve"],
        y_practice_preserve=case["y_practice_preserve"],
    )
    _save_npz(proposal / "damaged_source.npz", A=damaged["A"], b=damaged["b"], W=damaged["W"], c=damaged["c"])
    exclusive_create_bytes(proposal / "public_config.json", _json_bytes(official_config()))
    _save_npz(
        host / "intact_model.npz",
        A=intact["A"],
        b=intact["b"],
        W=intact["W"],
        c=intact["c"],
    )
    host_config = {
        "schema": "weightcrack.support_host.v1",
        "study_id": STUDY_ID,
        "model_seed": seeds["model_seed"],
        "data_seed": seeds["data_seed"],
        "fault_seed": seeds["fault_seed"],
        "confirmation_seed": seeds["confirmation_seed"],
        "fault_strength": FAULT_STRENGTH,
        "fault_columns": case["fault_columns"],
        "fault_direction_f64_hex": f64_hex(case["fault_direction"]),
        "fault_changed_coordinates": case["fault_changed_coordinates"],
        "intact_W_sha256": array_content_sha256(intact["W"]),
        "damaged_W_sha256": array_content_sha256(case["damaged_w"]),
        "generator": "numpy.random.Generator.standard_normal; parameters and inputs cast once to float32",
        "label_rule": "argmax of intact post_linear_logits",
        "fault_rule": official_config()["fault_rule"],
        "contribution_tolerance_rule": "1e-12 * max(1, max abs a_j)",
        "numpy": np.__version__,
        "python": platform_version(),
    }
    exclusive_create_bytes(host / "host_config.json", _json_bytes(host_config))
    _save_npz(
        host / "confirmation_bank.npz",
        A=intact["A"],
        b=intact["b"],
        c=intact["c"],
        W_intact=intact["W"],
        W_damaged=case["damaged_w"],
        x_confirmation_target=case["x_confirmation_target"],
        y_confirmation_target=case["y_confirmation_target"],
        x_confirmation_preserve=case["x_confirmation_preserve"],
        y_confirmation_preserve=case["y_confirmation_preserve"],
    )
    disclosure = {
        "schema": "weightcrack.support_disclosure.v1",
        "study_id": STUDY_ID,
        "known_generator": True,
        "blinded_fault_replication": False,
        "confirmation_seed": seeds["confirmation_seed"],
        "confirmation_target_sha256": array_content_sha256(case["x_confirmation_target"]),
        "confirmation_preserve_sha256": array_content_sha256(case["x_confirmation_preserve"]),
        "confirmation_target_label_sha256": array_content_sha256(case["y_confirmation_target"]),
        "confirmation_preserve_label_sha256": array_content_sha256(case["y_confirmation_preserve"]),
        "label_rule": "argmax of the intact teacher",
        "process_boundary": (
            "The proposer subprocess arguments are the practice export, public config, "
            "damaged source, candidates directory, and a deadline. The confirmation bank "
            "and the intact teacher are not arguments. Process separation is not a "
            "filesystem sandbox."
        ),
    }
    exclusive_create_bytes(host / "confirmation_disclosure.json", _json_bytes(disclosure))
    files = _commitment_files(out_dir)
    commitment = {
        "schema": "weightcrack.support_commitment.v1",
        "study_id": STUDY_ID,
        "host_files_in_protocol_commit": False,
        "files": {name: file_sha256(path) for name, path in files.items()},
        "source_identity": source_identity(),
    }
    exclusive_create_bytes(out_dir / "public_commitment.json", _json_bytes(commitment))
    _append_event(
        out_dir / "study_events.jsonl",
        {
            "stage": "host_committed",
            "study_id": STUDY_ID,
            "commitment_sha256": file_sha256(out_dir / "public_commitment.json"),
        },
    )
    print(f"status_code=0 directory={out_dir} stage=prepared")
    return 0


def _require_commitment(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "public_commitment.json"
    if not path.is_file():
        raise VerificationFailure("incomplete", "public commitment is missing")
    commitment = json.loads(path.read_text(encoding="utf-8"))
    if commitment.get("schema") != "weightcrack.support_commitment.v1":
        raise VerificationFailure("rejected", "public commitment schema changed")
    if commitment.get("source_identity") != source_identity():
        raise VerificationFailure("rejected", "source identity does not match the commitment")
    files = _commitment_files(out_dir)
    recorded = commitment.get("files")
    if not isinstance(recorded, dict) or set(recorded) != set(files):
        raise VerificationFailure("rejected", "commitment file list changed")
    for name, host_path in files.items():
        if not host_path.is_file():
            raise VerificationFailure("incomplete", f"committed file is missing: {name}")
        if file_sha256(host_path) != recorded[name]:
            raise VerificationFailure("rejected", f"committed file changed: {name}")
    return commitment


def evaluate(out_dir: Path) -> dict[str, Any]:
    """Score locked finalists once. Refuses to score when the lock is absent."""
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


def _base_report(started: float, status: str) -> dict[str, Any]:
    return {
        "schema": "weightcrack.support_report.v1",
        "study_id": STUDY_ID,
        "status": status,
        "question": QUESTION,
        "timing": _timing(started),
        "runtime": {"python": platform_version(), "numpy": np.__version__},
        "isolation": {
            "implemented": (
                "The proposer is a subprocess whose arguments are the practice export, "
                "public config, damaged source, candidates directory, and a deadline. "
                "The confirmation bank and the intact teacher are not arguments."
            ),
            "not_a_sandbox": (
                "Process separation is not a hardened filesystem sandbox. A modified "
                "proposer could open sibling files, including the confirmation bank."
            ),
            "publication_boundary": (
                "The protocol commit records hashes of the host files. The intact teacher "
                "and confirmation labels are disclosed after the lock and the confirmation score."
            ),
        },
    }


def run(out_dir: Path) -> int:
    """Official candidate trial. Requires a prepare commitment and does not redraw it."""
    out_dir = Path(out_dir)
    if _refuse_archive(out_dir):
        print(f"{out_dir} is protected evidence. Choose a new directory.", file=sys.stderr)
        return 1
    started = time.perf_counter()
    report_path = out_dir / "experiment_report.json"
    if report_path.exists() or (out_dir / "candidates" / "selection.json").exists():
        print(f"refusing to overwrite an existing trial in {out_dir}", file=sys.stderr)
        return 1
    try:
        commitment = _require_commitment(out_dir)
        paths = _commitment_files(out_dir)
        remaining = WALL_LIMIT_S - (time.perf_counter() - started)
        deadline = time.time() + max(0.0, remaining)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "weightcrack.evaluation.support_proposer",
                str(paths["practice_export.npz"]),
                str(paths["public_config.json"]),
                str(paths["damaged_source.npz"]),
                str(out_dir / "candidates"),
                f"{deadline:.6f}",
            ],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=WALL_LIMIT_S + 15,
            check=False,
        )
        _append_event(out_dir / "study_events.jsonl", {"stage": "proposer_returned", "code": completed.returncode})
        if completed.returncode not in (0, 2):
            report = _base_report(started, "VERIFICATION_FAILED")
            report["proposer_stdout"] = completed.stdout
            report["proposer_stderr"] = completed.stderr
            report["confirmation_opened"] = False
            exclusive_create_bytes(report_path, _json_bytes(report))
            print(f"status_code=1 directory={out_dir}")
            return 1
        selection_path = out_dir / "candidates" / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if completed.returncode == 2 or selection.get("status") == "RESOURCE_LIMIT" or _timing(started)["exceeded"]:
            report = _base_report(started, "RESOURCE_LIMIT")
            report["selection_status"] = selection.get("status")
            report["proposer_stdout"] = completed.stdout
            report["confirmation_opened"] = False
            report["budget"] = _slot_budget(selection)
            exclusive_create_bytes(report_path, _json_bytes(report))
            print(f"status_code=2 directory={out_dir}")
            return 2
        lock = {
            "schema": "weightcrack.support_lock.v1",
            "study_id": STUDY_ID,
            "selection": selection,
            "public_config_sha256": file_sha256(paths["public_config.json"]),
            "practice_export_sha256": file_sha256(paths["practice_export.npz"]),
            "source_file_sha256": file_sha256(paths["damaged_source.npz"]),
            "confirmation_bank_sha256": file_sha256(paths["confirmation_bank.npz"]),
            "commitment_sha256": file_sha256(out_dir / "public_commitment.json"),
            "confirmation_used_for_selection": False,
            "heldout_used_for_selection": False,
        }
        lock_path = out_dir / "proposal_lock.json"
        exclusive_create_bytes(lock_path, _json_bytes(lock))
        _append_event(out_dir / "study_events.jsonl", {"stage": "lock_written", "lock_sha256": file_sha256(lock_path)})
        if _timing(started)["exceeded"]:
            report = _base_report(started, "RESOURCE_LIMIT")
            report["lock_sha256"] = file_sha256(lock_path)
            report["confirmation_opened"] = False
            report["invalidated_by_time_limit"] = True
            exclusive_create_bytes(report_path, _json_bytes(report))
            print(f"status_code=2 directory={out_dir}")
            return 2
        _append_event(out_dir / "study_events.jsonl", {"stage": "confirmation_opened"})
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
                        paths["damaged_source.npz"],
                        TENSOR_NAME,
                        source_path=paths["damaged_source.npz"],
                    )
                )
        report = _base_report(started, "complete")
        report["lock_sha256"] = file_sha256(lock_path)
        report["source_file_sha256"] = file_sha256(paths["damaged_source.npz"])
        report["practice_export_sha256"] = file_sha256(paths["practice_export.npz"])
        report["public_config_sha256"] = file_sha256(paths["public_config.json"])
        report["confirmation_bank_sha256"] = lock["confirmation_bank_sha256"]
        report["commitment_sha256"] = lock["commitment_sha256"]
        report["source_identity"] = commitment["source_identity"]
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
            print(f"status_code=2 directory={out_dir}")
            return 2
        _append_event(out_dir / "study_events.jsonl", {"stage": "report_written", "status": report["status"]})
        exclusive_create_bytes(report_path, _json_bytes(report))
        print(f"status_code=0 directory={out_dir} comparison={report['comparison']}")
        return 0
    except subprocess.TimeoutExpired as exc:
        report = _base_report(started, "RESOURCE_LIMIT")
        report["timing_evidence"] = {"timeout_expired": True}
        report["confirmation_opened"] = False
        report["proposer_stdout"] = exc.stdout
        report["proposer_stderr"] = exc.stderr
        if not report_path.exists():
            exclusive_create_bytes(report_path, _json_bytes(report))
        print(f"status_code=2 directory={out_dir}")
        return 2
    except VerificationFailure as exc:
        report = _base_report(started, "VERIFICATION_FAILED")
        report["error"] = str(exc)
        report["confirmation_opened"] = False
        if not report_path.exists():
            exclusive_create_bytes(report_path, _json_bytes(report))
        print(f"status_code=1 directory={out_dir}")
        return 1
    except Exception as exc:
        report = _base_report(started, "VERIFICATION_FAILED")
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["confirmation_opened"] = False
        if not report_path.exists():
            exclusive_create_bytes(report_path, _json_bytes(report))
        print(f"status_code=1 directory={out_dir}")
        return 1


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in {"prepare", "run"}:
        print(
            "usage: python -m weightcrack.evaluation.support_experiment "
            "prepare experiments/proxy_support_v3 --seeds-file SEEDS.json",
            file=sys.stderr,
        )
        print(
            "       python -m weightcrack.evaluation.support_experiment run experiments/proxy_support_v3",
            file=sys.stderr,
        )
        return 1
    out = Path(argv[2])
    if argv[1] == "prepare":
        if len(argv) != 5 or argv[3] != "--seeds-file":
            print("prepare requires --seeds-file", file=sys.stderr)
            return 1
        return prepare(out, Path(argv[4]))
    if len(argv) != 3:
        print("run takes only the destination", file=sys.stderr)
        return 1
    return run(out)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
