"""Open the sealed math holdout once for the four-column ranking.

Columns stay 1309, 2633, 1775, and 1022. The direction is the top PCA
component of the eight failure traces, normalized the same way as the
real-matrix probe. Each strength goes through construct_candidate. A
rejected strength is recorded and its prompts are not tokenized. An
admitted strength has its candidate hash locked before any holdout prompt
is rendered, and each prompt is generated at most once. Baseline labels
come from the recorded strength-0.5 holdout and are not regenerated.
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
from weightcrack.canonical import array_content_sha256
from weightcrack.errors import EditRejected, PolicyViolation
from weightcrack.patches.huggingface import apply_patch, construct_candidate
from weightcrack.schema import PatchChange, WeightCrackPatch, WeightCrackPolicy

RANK = ROOT / "experiments" / "real_probe_rank"
BASELINE_PATH = ROOT / "experiments" / "real_probe" / "holdout.json"
STARTED = RANK / "ranking_holdout_started.json"
RESULT = RANK / "ranking_holdout.json"
PROGRESS = RANK / "ranking_progress.json"
COLUMNS = [1309, 2633, 1775, 1022]
STRENGTHS = [0.5, 1.0]
EXPECTED_FAIL = (8, 1024)
RECORDED_DELTA = {0.5: 0.002945561893284321, 1.0: 0.005891123786568642}


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
    return RANK / f"strength-{slug(strength)}.wcbundle"


def top_direction(fail: np.ndarray) -> list[float]:
    centered = fail.astype(np.float32) - fail.astype(np.float32).mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    second = vh[1]
    second = second / (np.linalg.norm(second) + 1e-12)
    return [float(value) for value in direction.tolist()], second


def ranked_columns(matrix: np.ndarray, first: np.ndarray, second: np.ndarray) -> list[int]:
    score = np.abs(matrix.T @ first) + np.abs(matrix.T @ second)
    return [int(column) for column in np.argsort(score)[::-1][:4].tolist()]


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
        reason="real-matrix ranking holdout",
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
    direction, second = top_direction(fail)
    first = np.asarray(direction, dtype=np.float64)
    found = ranked_columns(matrix, first, second)
    if found != COLUMNS:
        raise SystemExit(f"ranking columns changed: {found}")
    recorded = json.loads((RANK / "ranking.json").read_text(encoding="utf-8"))
    if recorded["columns"] != COLUMNS:
        raise SystemExit("ranking.json columns differ from the fixed ranking")
    baseline_doc = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if baseline_doc.get("strength") != 0.5:
        raise SystemExit("baseline file is not the recorded holdout")
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
        _candidate, accounting, _normalized = construct_candidate(
            matrix, patch, policy(), "weight"
        )
    except (EditRejected, PolicyViolation) as exc:
        return {
            "strength": strength,
            "admitted": False,
            "reason": f"{exc.__class__.__name__}: {exc}",
            "holdout_opened": False,
        }
    recorded = RECORDED_DELTA[strength]
    measured = accounting["max_abs_stored_delta"]
    if measured != recorded:
        raise SystemExit(
            f"strength {strength} delta {measured} does not match the ranking record {recorded}"
        )
    return {
        "strength": strength,
        "admitted": True,
        "candidate_content_sha256": array_content_sha256(_candidate),
        "touched_coordinates": accounting["touched_coordinates"],
        "max_abs_stored_delta": measured,
        "bundle": str(bundle_for(strength)),
        "holdout_opened": False,
    }


def publish(direction: list[float], item: dict) -> None:
    strength = float(item["strength"])
    destination = bundle_for(strength)
    expected = item["candidate_content_sha256"]
    matrix = np.load(cap.OUT)["weight"]
    if not destination.exists():
        receipt = apply_patch(
            cap.OUT,
            make_patch(array_content_sha256(matrix), direction, strength),
            policy(),
            destination,
            tensor_name="weight",
            task_id=f"real-matrix-ranking-{slug(strength)}",
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
            "direction_source": "top PCA component of the eight centered failure traces",
            "columns": COLUMNS,
            "fail_shape": list(EXPECTED_FAIL),
            "finalists": fresh,
        }
        dump(STARTED, started)
        finalists = fresh
        mark({"status": "locked", "strengths": [item["strength"] for item in fresh]})
    for item in finalists:
        if item["admitted"]:
            publish(direction, item)
    return started


def load_result(baseline: list[dict], baseline_loops: int) -> dict:
    if RESULT.exists():
        return json.loads(RESULT.read_text(encoding="utf-8"))
    return {
        "status": "in_progress",
        "loop_rule": "a 4-word sequence occurs at least 3 times and the completion has at least 12 words",
        "max_new_tokens": cap.MAX_NEW_TOKENS,
        "columns": COLUMNS,
        "direction_source": "top PCA component of the eight centered failure traces",
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
    flipped = [
        row["id"]
        for row, base in zip(rows["edited"], result["baseline"])
        if bool(row["loop"]) != bool(base["loop"])
    ]
    rows["edited_loops"] = sum(1 for row in rows["edited"] if row["loop"])
    rows["labels_flipped"] = flipped
    rows["behavior_moved"] = bool(flipped)
    rows["holdout_opened"] = True
    dump(RESULT, result)
    mark({
        "status": "strength_complete",
        "strength": strength,
        "edited_loops": rows["edited_loops"],
        "baseline_loops": result["baseline_loops"],
        "labels_flipped": flipped,
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
    by_strength = {float(entry["strength"]): entry for entry in result["strengths"]}
    for item in started["finalists"]:
        if not item["admitted"]:
            continue
        if by_strength[float(item["strength"])].get("holdout_opened"):
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
        result["strengths"].append({
            "strength": strength,
            "admitted": item["admitted"],
            "candidate_content_sha256": item.get("candidate_content_sha256"),
            "touched_coordinates": item.get("touched_coordinates"),
            "max_abs_stored_delta": item.get("max_abs_stored_delta"),
            "reason": item.get("reason"),
            "bundle": item.get("bundle"),
            "holdout_opened": False,
            "edited": [],
        })
    dump(RESULT, result)
    if gate_only:
        mark({
            "status": "gate_only",
            "admitted": [item["strength"] for item in started["finalists"] if item["admitted"]],
        })
        return 0
    generate(started, result)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    result["status"] = "opened_once"
    result["criterion_met"] = any(entry.get("behavior_moved") for entry in result["strengths"])
    dump(RESULT, result)
    mark({"status": "RANK_DONE", "criterion_met": result["criterion_met"]})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        mark({"status": "FAILED", "error": f"{exc.__class__.__name__}: {exc}"})
        raise
