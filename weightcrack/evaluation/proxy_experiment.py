"""Preregistered proxy fault study.

Constants below were fixed before this study was run. Do not edit them to
chase a behavioral result. The proposer subprocess receives only the practice
export, the public config, the damaged source, and the candidates directory.

Reproduce:
    python -m weightcrack.evaluation.proxy_experiment experiments/proxy_fault_v1

Verify and replay:
    python -m weightcrack.evaluation.verify_proxy experiments/proxy_fault_v1

Exit 0 means the protocol finished, including a negative or empty result.
Exit 2 means the wall-time ceiling was hit. Exit 1 means a harness or
verification failure. A file write is not a behavioral repair.
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
from ..schema import WeightCrackPolicy
from .proxy_model import derive_direction, predict, prediction_sha256, preservation_flips, split_counts

STUDY_ID = "proxy_fault_v1"
MODEL_SEED = 1108
DATA_SEED = 1109
FAULT_SEED = 1110
CONTROL_SEED = 1111
FAULT_STRENGTH = 1.0
DOSES = (0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20)
MATCH_TOLERANCE = 1e-3
MAX_ENTRIES = 8
MAX_FRACTION = 0.125
MAX_ABS_DELTA = 2.0
WALL_LIMIT_S = 60.0
SUPPORT_SIZE = 2
TENSOR_NAME = "W"
SECTOR = "synthetic_classifier"
HISTORICAL_REPORTS = {"synthetic_v1", "synthetic_strong", "synthetic_search"}
POPULATION = {
    "practice_target": 128,
    "practice_preserve": 128,
    "heldout_target": 256,
    "heldout_preserve": 256,
}
ROOT = Path(__file__).resolve().parents[2]


def official_policy() -> WeightCrackPolicy:
    return WeightCrackPolicy(
        allow_tensors=[TENSOR_NAME],
        max_entries=MAX_ENTRIES,
        max_fraction=MAX_FRACTION,
        max_abs_delta=MAX_ABS_DELTA,
        allowed_ops=["project_remove"],
    )


def official_config() -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "tensor_name": TENSOR_NAME,
        "sector": SECTOR,
        "activation_site": "post_linear_logits",
        "storage_dtype": "float32",
        "working_dtype": "float64",
        "doses": list(DOSES),
        "match_tolerance": MATCH_TOLERANCE,
        "control_seed": CONTROL_SEED,
        "support_size": SUPPORT_SIZE,
        "population": dict(POPULATION),
        "shapes": {"A": [16, 8], "b": [16], "W": [4, 16], "c": [4], "x": [8]},
        "wall_limit_s": WALL_LIMIT_S,
        "ranking": "absolute projection onto the shared practice mean-difference direction",
        "random_control": "uniform selection of two columns without replacement",
        "selection_rule": (
            "positive practice target corrections, zero practice target losses, "
            "zero practice preservation flips; then more corrections, smaller "
            "stored update norm, earlier dose index"
        ),
        "policy": official_policy().model_dump(mode="json"),
    }


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


def _draw_model(seed: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    drawn = {
        "A": generator.standard_normal((16, 8)),
        "b": generator.standard_normal(16),
        "W": generator.standard_normal((4, 16)),
        "c": generator.standard_normal(4),
    }
    return {name: value.astype(np.float32) for name, value in drawn.items()}


def _draw_inputs(seed: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    order = (
        ("x_practice_target", POPULATION["practice_target"]),
        ("x_practice_preserve", POPULATION["practice_preserve"]),
        ("x_heldout_target", POPULATION["heldout_target"]),
        ("x_heldout_preserve", POPULATION["heldout_preserve"]),
    )
    return {
        name: generator.standard_normal((count, 8)).astype(np.float32)
        for name, count in order
    }


def freeze_study(out_dir: Path) -> dict[str, Path]:
    """Write host secrets and the proposer allowlist. Does not score candidates."""
    out_dir = Path(out_dir)
    proposal = out_dir / "proposal_inputs"
    host = out_dir / "host_only"
    candidates = out_dir / "candidates"
    for path in (proposal, host, candidates):
        path.mkdir(parents=True, exist_ok=False)
    intact = _draw_model(MODEL_SEED)
    inputs = _draw_inputs(DATA_SEED)
    labels = {
        name.replace("x_", "y_"): predict(intact["A"], intact["b"], intact["W"], intact["c"], value)
        for name, value in inputs.items()
    }
    fault_rng = np.random.default_rng(FAULT_SEED)
    fault_columns = [int(column) for column in fault_rng.choice(16, size=2, replace=False)]
    fault_direction = fault_rng.standard_normal(4)
    fault_direction = fault_direction / np.linalg.norm(fault_direction)
    damaged_W, fault_accounting = apply_project_remove(
        intact["W"], fault_direction.tolist(), fault_columns, float(FAULT_STRENGTH)
    )
    damaged = dict(intact)
    damaged["W"] = damaged_W
    practice_payload = {
        "A": damaged["A"],
        "b": damaged["b"],
        "W": damaged["W"],
        "c": damaged["c"],
        "x_practice_target": inputs["x_practice_target"],
        "y_practice_target": labels["y_practice_target"],
        "x_practice_preserve": inputs["x_practice_preserve"],
        "y_practice_preserve": labels["y_practice_preserve"],
    }
    paths = {
        "practice": proposal / "practice_export.npz",
        "config": proposal / "public_config.json",
        "source": proposal / "damaged_source.npz",
        "candidates": candidates,
        "secret_npz": host / "host_secret.npz",
        "secret_json": host / "host_secret.json",
        "events": out_dir / "study_events.jsonl",
    }
    _save_npz(paths["practice"], **practice_payload)
    _save_npz(paths["source"], A=damaged["A"], b=damaged["b"], W=damaged["W"], c=damaged["c"])
    exclusive_create_bytes(paths["config"], _json_bytes(official_config()))
    _save_npz(
        paths["secret_npz"],
        A=intact["A"],
        b=intact["b"],
        c=intact["c"],
        W_intact=intact["W"],
        W_damaged=damaged_W,
        fault_direction=np.asarray(fault_direction, dtype=np.float64),
        fault_columns=np.asarray(fault_columns, dtype=np.int64),
        **inputs,
        **labels,
    )
    secret = {
        "schema": "weightcrack.proxy_secret.v1",
        "model_seed": MODEL_SEED,
        "data_seed": DATA_SEED,
        "fault_seed": FAULT_SEED,
        "control_seed": CONTROL_SEED,
        "fault_strength": FAULT_STRENGTH,
        "fault_columns": fault_columns,
        "fault_direction_f64_hex": f64_hex(np.asarray(fault_direction, dtype=np.float64)),
        "fault_changed_coordinates": fault_accounting["changed_coordinates"],
        "intact_W_sha256": array_content_sha256(intact["W"]),
        "damaged_W_sha256": array_content_sha256(damaged_W),
        "generator": "numpy.random.Generator.standard_normal; parameters cast once to float32",
        "label_rule": "argmax of intact post_linear_logits",
        "fault_rule": "one project_remove of strength 1 on two fault-rng columns",
        "numpy": np.__version__,
        "python": platform_version(),
    }
    exclusive_create_bytes(paths["secret_json"], _json_bytes(secret))
    _append_event(paths["events"], {"stage": "frozen", "study_id": STUDY_ID})
    return paths


def platform_version() -> str:
    return sys.version.split()[0]


def _load_selection(candidates: Path) -> dict[str, Any]:
    path = candidates / "selection.json"
    if not path.is_file():
        raise VerificationFailure("incomplete", "proposer did not write selection.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_official(config: dict[str, Any]) -> None:
    expected = json.loads(_json_bytes(official_config()).decode("utf-8"))
    if config != expected:
        raise VerificationFailure("rejected", "public config does not match the frozen study")


def _recompute_practice(out_dir: Path, selection: dict[str, Any]) -> None:
    """Repeat practice scores from proposer-visible files only."""
    proposal = out_dir / "proposal_inputs"
    with np.load(proposal / "practice_export.npz") as handle:
        practice = {key: np.array(handle[key], copy=True) for key in handle.files}
    derived = derive_direction(
        practice["A"], practice["b"], practice["W"], practice["c"],
        practice["x_practice_target"], practice["y_practice_target"],
    )
    if derived["status"] != ("ok" if selection["status"] == "ok" else selection["status"]):
        if not (derived["status"] == "NO_LOCALIZATION_SIGNAL" and selection["status"] == "NO_LOCALIZATION_SIGNAL"):
            raise VerificationFailure("rejected", "host practice direction disagrees with the lock")
    if selection["status"] == "NO_LOCALIZATION_SIGNAL":
        return
    if derived["status"] != "ok":
        raise VerificationFailure("rejected", "host practice direction disagrees with the lock")
    host_hash = array_content_sha256(np.asarray(derived["direction"], dtype=np.float64))
    if host_hash != selection["activation"]["direction_sha256"]:
        raise VerificationFailure("rejected", "practice direction hash disagrees with the lock")
    baseline_target = predict(practice["A"], practice["b"], practice["W"], practice["c"], practice["x_practice_target"])
    baseline_preserve = predict(practice["A"], practice["b"], practice["W"], practice["c"], practice["x_practice_preserve"])
    if prediction_sha256(baseline_target) != selection["no_edit"]["practice_target_prediction_sha256"]:
        raise VerificationFailure("rejected", "no-edit practice predictions disagree with the lock")
    for method, payload in selection["methods"].items():
        for slot in payload["slots"]:
            if slot["status"] != "applied":
                continue
            bundle = out_dir / "candidates" / slot["bundle_name"]
            with np.load(bundle / "candidate.npz") as handle:
                candidate = np.array(handle["W"], copy=True)
            target = predict(practice["A"], practice["b"], candidate, practice["c"], practice["x_practice_target"])
            preserve = predict(practice["A"], practice["b"], candidate, practice["c"], practice["x_practice_preserve"])
            counts = split_counts(baseline_target, target, practice["y_practice_target"])
            flips = preservation_flips(baseline_preserve, preserve)
            practice_score = slot["practice"]
            if counts["corrections"] != practice_score["corrections"] or counts["losses"] != practice_score["losses"]:
                raise VerificationFailure("rejected", "practice corrections do not match the saved candidate")
            if flips != practice_score["preservation_flips"]:
                raise VerificationFailure("rejected", "practice preservation flips do not match the saved candidate")
            relative = abs(float(slot["stored_update_frobenius"]) - float(slot["target_norm"])) / float(slot["target_norm"])
            if relative > MATCH_TOLERANCE:
                raise VerificationFailure("rejected", "admitted candidate missed the stored-norm tolerance")


def _score_matrix(model, matrix, baseline_target, baseline_preserve, secret) -> dict[str, Any]:
    target = predict(model["A"], model["b"], matrix, model["c"], secret["x_heldout_target"])
    preserve = predict(model["A"], model["b"], matrix, model["c"], secret["x_heldout_preserve"])
    counts = split_counts(baseline_target, target, secret["y_heldout_target"])
    flips = preservation_flips(baseline_preserve, preserve)
    return {
        "corrections": counts["corrections"],
        "losses": counts["losses"],
        "baseline_correct": counts["baseline_correct"],
        "candidate_correct": counts["candidate_correct"],
        "count": counts["count"],
        "preservation_flips": flips,
        "preservation_count": int(secret["x_heldout_preserve"].shape[0]),
        "behavioral_repair": counts["corrections"] > 0 and counts["losses"] == 0 and flips == 0,
        "target_prediction_sha256": prediction_sha256(target),
        "preserve_prediction_sha256": prediction_sha256(preserve),
    }


def evaluate(out_dir: Path) -> dict[str, Any]:
    """Score locked finalists. Refuses to open the secret when the lock is absent."""
    out_dir = Path(out_dir)
    lock_path = out_dir / "proposal_lock.json"
    if not lock_path.is_file():
        raise VerificationFailure("incomplete", "proposal lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _recompute_practice(out_dir, lock["selection"])
    secret_path = out_dir / "host_only" / "host_secret.npz"
    with np.load(secret_path) as handle:
        secret = {key: np.array(handle[key], copy=True) for key in handle.files}
    secret_meta = json.loads((out_dir / "host_only" / "host_secret.json").read_text(encoding="utf-8"))
    model = {"A": secret["A"], "b": secret["b"], "c": secret["c"], "W": secret["W_damaged"]}
    baseline_target = predict(model["A"], model["b"], model["W"], model["c"], secret["x_heldout_target"])
    baseline_preserve = predict(model["A"], model["b"], model["W"], model["c"], secret["x_heldout_preserve"])
    heldout: dict[str, Any] = {
        "opened_after_lock": True,
        "used_for_selection": False,
        "no_edit": _score_matrix(model, model["W"], baseline_target, baseline_preserve, secret),
        "methods": {},
    }
    if heldout["no_edit"]["corrections"] != 0 or heldout["no_edit"]["losses"] != 0 or heldout["no_edit"]["preservation_flips"] != 0:
        raise VerificationFailure("rejected", "no-edit control does not reproduce the damaged model")
    selection = lock["selection"]
    if selection["status"] == "ok":
        for method, finalist in selection["finalists"].items():
            if finalist["status"] != "qualified":
                heldout["methods"][method] = {"status": "NO_QUALIFIED_CANDIDATE"}
                continue
            bundle = out_dir / "candidates" / finalist["bundle_name"]
            with np.load(bundle / "candidate.npz") as handle:
                candidate = np.array(handle["W"], copy=True)
            if array_content_sha256(candidate) != finalist["candidate_content_sha256"]:
                raise VerificationFailure("rejected", "finalist hash does not match the locked candidate")
            scored = _score_matrix(model, candidate, baseline_target, baseline_preserve, secret)
            scored["status"] = "qualified"
            scored["dose_index"] = finalist["dose_index"]
            scored["bundle_name"] = finalist["bundle_name"]
            scored["candidate_content_sha256"] = finalist["candidate_content_sha256"]
            scored["candidate_file_sha256"] = finalist["candidate_file_sha256"]
            scored["accounting"] = finalist["accounting"]
            scored["practice"] = finalist["practice"]
            heldout["methods"][method] = scored
    return {"lock": lock, "heldout": heldout, "host_disclosure": secret_meta}


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    exclusive_create_bytes(path, _json_bytes(payload))


def _timing(started: float) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    return {
        "wall_seconds": elapsed,
        "limit_seconds": WALL_LIMIT_S,
        "exceeded": elapsed > WALL_LIMIT_S,
        "clock": "time.perf_counter",
    }


def _base_report(started: float, status: str) -> dict[str, Any]:
    return {
        "schema": "weightcrack.proxy_report.v1",
        "study_id": STUDY_ID,
        "status": status,
        "timing": _timing(started),
        "runtime": {"python": platform_version(), "numpy": np.__version__},
        "isolation": {
            "implemented": (
                "The proposer is a subprocess whose arguments are the practice export, "
                "public config, damaged source, candidates directory, and a deadline. "
                "The host secret is not an argument."
            ),
            "not_a_sandbox": (
                "Process separation is not a hardened filesystem sandbox. A modified "
                "proposer could open sibling files."
            ),
        },
        "question": (
            "Does absolute-projection column selection repair held-out predictions "
            "better than a matched random column pair, at one shared practice direction?"
        ),
    }


def run(out_dir: Path) -> int:
    out_dir = Path(out_dir)
    if out_dir.name in HISTORICAL_REPORTS:
        print(f"{out_dir.name} is historical evidence. Choose a new directory.", file=sys.stderr)
        return 1
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"refusing to overwrite {out_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report_path = out_dir / "experiment_report.json"
    try:
        paths = freeze_study(out_dir)
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        _check_official(config)
        remaining = WALL_LIMIT_S - (time.perf_counter() - started)
        deadline = time.time() + max(0.0, remaining)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "weightcrack.evaluation.proxy_proposer",
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
            _write_report(report_path, report)
            return 2
        _append_event(paths["events"], {"stage": "proposer_returned", "code": completed.returncode})
        if completed.returncode not in (0, 2):
            report = _base_report(started, "VERIFICATION_FAILED")
            report["proposer_stdout"] = completed.stdout
            report["proposer_stderr"] = completed.stderr
            _write_report(report_path, report)
            return 1
        selection = _load_selection(paths["candidates"])
        if completed.returncode == 2 or selection.get("status") == "RESOURCE_LIMIT" or _timing(started)["exceeded"]:
            report = _base_report(started, "RESOURCE_LIMIT")
            report["selection_status"] = selection.get("status")
            report["proposer_stdout"] = completed.stdout
            _write_report(report_path, report)
            return 2
        _recompute_practice(out_dir, selection)
        lock = {
            "schema": "weightcrack.proxy_lock.v1",
            "study_id": STUDY_ID,
            "selection": selection,
            "public_config_sha256": file_sha256(paths["config"]),
            "practice_export_sha256": file_sha256(paths["practice"]),
            "source_file_sha256": file_sha256(paths["source"]),
            "heldout_used_for_selection": False,
        }
        lock_path = out_dir / "proposal_lock.json"
        exclusive_create_bytes(lock_path, _json_bytes(lock))
        _append_event(paths["events"], {"stage": "lock_written", "lock_sha256": file_sha256(lock_path)})
        if _timing(started)["exceeded"]:
            report = _base_report(started, "RESOURCE_LIMIT")
            report["lock_sha256"] = file_sha256(lock_path)
            report["heldout_opened"] = False
            _write_report(report_path, report)
            return 2
        _append_event(paths["events"], {"stage": "heldout_opened"})
        judged = evaluate(out_dir)
        from ..verify import verify_bundle

        verified = []
        if selection["status"] == "ok":
            for payload in selection["methods"].values():
                for slot in payload["slots"]:
                    if slot["status"] != "applied":
                        continue
                    verified.append(
                        verify_bundle(
                            out_dir / "candidates" / slot["bundle_name"],
                            paths["source"],
                            TENSOR_NAME,
                            source_path=paths["source"],
                        )
                    )
        report = _base_report(started, "NO_LOCALIZATION_SIGNAL" if selection["status"] == "NO_LOCALIZATION_SIGNAL" else "complete")
        report["lock_sha256"] = file_sha256(lock_path)
        report["source_file_sha256"] = file_sha256(paths["source"])
        report["practice_export_sha256"] = file_sha256(paths["practice"])
        report["public_config_sha256"] = file_sha256(paths["config"])
        report["selection"] = selection
        report["heldout"] = judged["heldout"]
        report["host_disclosure"] = judged["host_disclosure"]
        report["verified_applied_bundles"] = len(verified)
        report["value_exact_reconstruction"] = all(item["value_exact_reconstruction"] for item in verified) if verified else None
        report["resave_byte_identical"] = [bool(item["resave_byte_identical"]) for item in verified]
        report["heldout_opened"] = True
        if report["timing"]["exceeded"]:
            report["status"] = "RESOURCE_LIMIT"
            report["invalidated_by_time_limit"] = True
            _write_report(report_path, report)
            return 2
        _append_event(paths["events"], {"stage": "report_written", "status": report["status"]})
        _write_report(report_path, report)
        return 0
    except VerificationFailure as exc:
        report = _base_report(started, "VERIFICATION_FAILED")
        report["error"] = str(exc)
        if not report_path.exists():
            _write_report(report_path, report)
        return 1
    except Exception as exc:
        report = _base_report(started, "VERIFICATION_FAILED")
        report["error"] = f"{type(exc).__name__}: {exc}"
        if not report_path.exists():
            _write_report(report_path, report)
        return 1


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "experiments/proxy_fault_v1")
    code = run(out)
    print(f"status_code={code} directory={out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
