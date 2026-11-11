"""Open the sealed math holdout once at each wider dose.

Columns stay 2056 and 1309. The direction is the float64 vector stored on
the strength-0.5 receipt. Each strength is built with construct_candidate.
A rejected strength is recorded and its prompts are not tokenized. An
admitted strength has its candidate hash locked before any holdout prompt
is rendered, and each prompt is generated at most once. Baseline labels
come from the strength-0.5 holdout and are not regenerated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "real_matrix"))

import capture_qwen35_traces as cap
from weightcrack.canonical import array_content_sha256, f64_from_hex
from weightcrack.errors import EditRejected, PolicyViolation
from weightcrack.patches.huggingface import apply_patch, construct_candidate
from weightcrack.schema import PatchChange, WeightCrackPatch, WeightCrackPolicy

DOSE = ROOT / "experiments" / "real_probe_dose"
RECEIPT = ROOT / "experiments" / "real_probe" / "finalist.wcbundle" / "receipt.json"
BASELINE_PATH = ROOT / "experiments" / "real_probe" / "holdout.json"
STARTED = DOSE / "dose_holdout_started.json"
RESULT = DOSE / "dose_holdout.json"
PROGRESS = DOSE / "dose_progress.json"
COLUMNS = [2056, 1309]
STRENGTHS = [1.0, 1.5, 2.0]
EXPECTED_FAIL = (8, 1024)


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def mark(payload: dict) -> None:
    dump(PROGRESS, payload)
    print(json.dumps(payload), flush=True)


def slug(strength: float) -> str:
    return f"{strength:.1f}"


def bundle_for(strength: float) -> Path:
    return DOSE / f"strength-{slug(strength)}.wcbundle"


def load_direction() -> tuple[list[float], str]:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    normalized = receipt["normalized_patch"]
    if normalized["columns"] != COLUMNS:
        raise SystemExit(f"receipt columns are not {COLUMNS}")
    if float(normalized["strength"]) != 0.5:
        raise SystemExit("receipt is not the strength 0.5 finalist")
    decoded = f64_from_hex(normalized["direction_f64_hex"])
    if decoded.shape != (EXPECTED_FAIL[1],):
        raise SystemExit(f"direction length {decoded.shape} is not {EXPECTED_FAIL[1]}")
    if not np.isfinite(decoded).all():
        raise SystemExit("recovered direction is nonfinite")
    return [float(value) for value in decoded.tolist()], receipt["base_sha256"]


def policy() -> WeightCrackPolicy:
    return WeightCrackPolicy(
        allow_tensors=["weight"],
        max_entries=10000,
        max_fraction=0.05,
        max_abs_delta=1.0,
    )


def make_patch(base_sha: str, direction: list[float], strength: float) -> WeightCrackPatch:
    return WeightCrackPatch(
        base_sha256=base_sha,
        reason="real-matrix dose holdout",
        sector="math-loop",
        changes=[
            PatchChange(
                op="project_remove",
                tensor="weight",
                direction=direction,
                index=COLUMNS,
                strength=float(strength),
            )
        ],
    )


def load_inputs() -> tuple[np.ndarray, list[float], list[dict], int]:
    traces = np.load(cap.TRACES)
    fail = traces["fail"]
    if tuple(fail.shape) != EXPECTED_FAIL:
        raise SystemExit(f"fail shape {tuple(fail.shape)} is not {EXPECTED_FAIL}")
    matrix = np.load(cap.OUT)["weight"]
    if matrix.shape != (1024, 3584) or matrix.dtype != np.float32:
        raise SystemExit(f"unexpected matrix {matrix.shape} {matrix.dtype}")
    direction, receipt_base = load_direction()
    base_sha = array_content_sha256(matrix)
    if base_sha != receipt_base:
        raise SystemExit("matrix hash does not match the strength 0.5 receipt")
    centered = fail.astype(np.float64) - fail.astype(np.float64).mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pca = vh[0]
    pca = pca / (np.linalg.norm(pca) + 1e-12)
    recovered = np.asarray(direction, dtype=np.float64)
    recovered = recovered / np.linalg.norm(recovered)
    cosine = float(np.dot(pca, recovered))
    if abs(cosine) < 0.999:
        raise SystemExit(f"receipt direction does not match the failure PCA, cosine {cosine}")
    baseline_doc = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if baseline_doc.get("strength") != 0.5:
        raise SystemExit("baseline file is not the strength 0.5 holdout")
    baseline = baseline_doc["baseline"]
    if [row["id"] for row in baseline] != [item["id"] for item in cap.HOLDOUT]:
        raise SystemExit("baseline ids do not match the sealed holdout")
    baseline_loops = sum(1 for row in baseline if row["loop"])
    if baseline_loops != baseline_doc["baseline_loops"]:
        raise SystemExit("baseline loop count does not match its rows")
    return matrix, direction, baseline, baseline_loops


def admit_one(matrix: np.ndarray, direction: list[float], strength: float) -> dict:
    patch = make_patch(array_content_sha256(matrix), direction, strength)
    try:
        candidate, accounting, _normalized = construct_candidate(
            matrix, patch, policy(), "weight"
        )
    except (EditRejected, PolicyViolation) as exc:
        return {
            "strength": strength,
            "admitted": False,
            "reason": f"{exc.__class__.__name__}: {exc}",
            "holdout_opened": False,
        }
    return {
        "strength": strength,
        "admitted": True,
        "candidate_content_sha256": array_content_sha256(candidate),
        "touched_coordinates": accounting["touched_coordinates"],
        "max_abs_stored_delta": accounting["max_abs_stored_delta"],
        "bundle": str(bundle_for(strength)),
        "holdout_opened": False,
    }


def publish(matrix_path: Path, direction: list[float], item: dict) -> None:
    strength = float(item["strength"])
    destination = bundle_for(strength)
    expected = item["candidate_content_sha256"]
    if not destination.exists():
        receipt = apply_patch(
            matrix_path,
            make_patch(array_content_sha256(np.load(matrix_path)["weight"]), direction, strength),
            policy(),
            destination,
            tensor_name="weight",
            task_id=f"real-matrix-dose-{slug(strength)}",
        )
        if receipt["candidate_content_sha256"] != expected:
            raise SystemExit(f"published hash drifted for strength {strength}")
    receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
    if receipt["candidate_content_sha256"] != expected:
        raise SystemExit(f"bundle hash does not match the lock for strength {strength}")
    if receipt["normalized_patch"]["columns"] != COLUMNS:
        raise SystemExit(f"bundle columns drifted for strength {strength}")
    if float(receipt["normalized_patch"]["strength"]) != strength:
        raise SystemExit(f"bundle strength drifted for strength {strength}")


def lock_finalists(matrix: np.ndarray, direction: list[float]) -> dict:
    fresh = [admit_one(matrix, direction, strength) for strength in STRENGTHS]
    if STARTED.exists():
        started = json.loads(STARTED.read_text(encoding="utf-8"))
        previous = {float(item["strength"]): item for item in started["finalists"]}
        for item in fresh:
            prior = previous.get(float(item["strength"]))
            if prior is None:
                raise SystemExit("started file is missing a strength")
            if prior.get("admitted") != item["admitted"]:
                raise SystemExit(f"admission changed for strength {item['strength']}")
            if item["admitted"] and prior.get("candidate_content_sha256") != item["candidate_content_sha256"]:
                raise SystemExit(f"candidate hash changed for strength {item['strength']}")
        finalists = started["finalists"]
    else:
        started = {
            "status": "locked",
            "direction_source": "experiments/real_probe/finalist.wcbundle/receipt.json normalized_patch.direction_f64_hex",
            "columns": COLUMNS,
            "fail_shape": list(EXPECTED_FAIL),
            "finalists": fresh,
        }
        dump(STARTED, started)
        finalists = fresh
        mark({"status": "locked", "strengths": [item["strength"] for item in fresh]})
    for item in finalists:
        if item["admitted"]:
            publish(cap.OUT, direction, item)
    return started


def load_result(baseline: list[dict], baseline_loops: int) -> dict:
    if RESULT.exists():
        return json.loads(RESULT.read_text(encoding="utf-8"))
    return {
        "status": "in_progress",
        "loop_rule": "a 4-word sequence occurs at least 3 times and the completion has at least 12 words",
        "max_new_tokens": cap.MAX_NEW_TOKENS,
        "columns": COLUMNS,
        "direction_source": "experiments/real_probe/finalist.wcbundle/receipt.json normalized_patch.direction_f64_hex",
        "baseline_source": "experiments/real_probe/holdout.json",
        "baseline_loops": baseline_loops,
        "baseline_regenerated": False,
        "holdout_count": len(cap.HOLDOUT),
        "baseline": baseline,
        "strengths": [],
    }


def score_missing(model, tokenizer, module, item: dict, result: dict) -> None:
    import torch

    strength = float(item["strength"])
    rows = next(entry for entry in result["strengths"] if float(entry["strength"]) == strength)
    done = {row["id"] for row in rows["edited"]}
    with np.load(bundle_for(strength) / "candidate.npz") as handle:
        candidate = np.array(handle["weight"], dtype=np.float32, copy=True)
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(candidate))
    for prompt in cap.HOLDOUT:
        if prompt["id"] in done:
            continue
        rendered = cap.render_prompt(tokenizer, prompt["prompt"])
        text = cap.generate(model, tokenizer, rendered)
        rows["edited"].append({
            "id": prompt["id"],
            "prompt": prompt["prompt"],
            "loop": cap.is_loop(text),
            "completion": text,
        })
        rows["edited_loops"] = sum(1 for row in rows["edited"] if row["loop"])
        dump(RESULT, result)
        mark({
            "status": "prompt_saved",
            "strength": strength,
            "id": prompt["id"],
            "saved": len(rows["edited"]),
        })
    if [row["id"] for row in rows["edited"]] != [prompt["id"] for prompt in cap.HOLDOUT]:
        raise SystemExit(f"holdout ids incomplete for strength {strength}")
    labels_moved = [
        row["id"]
        for row, base in zip(rows["edited"], result["baseline"])
        if bool(row["loop"]) != bool(base["loop"])
    ]
    rows["edited_loops"] = sum(1 for row in rows["edited"] if row["loop"])
    rows["behavior_moved"] = bool(labels_moved) or rows["edited_loops"] != result["baseline_loops"]
    rows["labels_moved"] = labels_moved
    rows["holdout_opened"] = True
    item["holdout_opened"] = True
    dump(RESULT, result)
    mark({
        "status": "strength_complete",
        "strength": strength,
        "edited_loops": rows["edited_loops"],
        "baseline_loops": result["baseline_loops"],
        "behavior_moved": rows["behavior_moved"],
    })


def generate(started: dict, result: dict) -> None:
    from huggingface_hub import snapshot_download

    mark({"status": "loading_model"})

    folder = Path(snapshot_download(
        cap.MODEL_ID,
        revision=cap.REVISION,
        cache_dir=str(cap.CACHE),
        local_files_only=True,
    ))
    matrix = np.load(cap.OUT)["weight"]
    tokenizer = cap.load_tokenizer(folder)
    model, module = cap.load_model(folder, matrix)
    for item in started["finalists"]:
        if not item["admitted"]:
            continue
        score_missing(model, tokenizer, module, item, result)


def main() -> int:
    gate_only = "--gate-only" in sys.argv
    matrix, direction, baseline, baseline_loops = load_inputs()
    started = lock_finalists(matrix, direction)
    result = load_result(baseline, baseline_loops)
    by_strength = {float(entry["strength"]): entry for entry in result["strengths"]}
    for item in started["finalists"]:
        strength = float(item["strength"])
        if strength in by_strength:
            continue
        entry = {
            "strength": strength,
            "admitted": item["admitted"],
            "candidate_content_sha256": item.get("candidate_content_sha256"),
            "touched_coordinates": item.get("touched_coordinates"),
            "max_abs_stored_delta": item.get("max_abs_stored_delta"),
            "reason": item.get("reason"),
            "bundle": item.get("bundle"),
            "holdout_opened": False,
            "edited": [],
        }
        result["strengths"].append(entry)
        by_strength[strength] = entry
    dump(RESULT, result)
    if gate_only:
        mark({"status": "gate_only", "admitted": [item["strength"] for item in started["finalists"] if item["admitted"]]})
        return 0
    pending = [
        item for item in started["finalists"]
        if item["admitted"] and not by_strength[float(item["strength"])].get("holdout_opened")
    ]
    if pending:
        generate(started, result)
    result["status"] = "opened_once"
    result["criterion_met"] = any(entry.get("behavior_moved") for entry in result["strengths"])
    dump(RESULT, result)
    mark({"status": "DOSE_DONE", "criterion_met": result["criterion_met"]})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        mark({"status": "FAILED", "error": f"{exc.__class__.__name__}: {exc}"})
        raise
