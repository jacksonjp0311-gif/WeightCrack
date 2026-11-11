"""Open the sealed math holdout once after the admitted probe candidate.

The candidate is rebuilt with the same PCA, columns, strength, and policy
the probe used, and it is published through apply_patch. Generation starts
only after holdout_started.json records the candidate hash.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import capture_qwen35_traces as cap
from weightcrack.patches.huggingface import apply_patch
from weightcrack.schema import PatchChange, WeightCrackPatch, WeightCrackPolicy

MATRIX = cap.OUT
TRACES = cap.TRACES
PROBE = ROOT / "experiments" / "real_probe"
BUNDLE = PROBE / "finalist.wcbundle"
STARTED = PROBE / "holdout_started.json"
HOLDOUT_OUT = PROBE / "holdout.json"
EXPECTED_COLUMNS = [2056, 1309]
STRENGTH = 0.5


def build_patch(matrix: np.ndarray) -> tuple[WeightCrackPatch, WeightCrackPolicy, list[int]]:
    traces = np.load(TRACES)
    fail = traces["fail"].astype(np.float32)
    centered = fail - fail.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    write = np.abs(matrix.T @ direction)
    columns = [int(column) for column in np.argsort(write)[::-1][:2].tolist()]
    if columns != EXPECTED_COLUMNS:
        raise SystemExit(f"probe columns changed: {columns}")
    source_sha = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
    patch = WeightCrackPatch(
        base_sha256=source_sha,
        reason="real-matrix probe",
        sector="math-loop",
        changes=[
            PatchChange(
                op="project_remove",
                tensor="weight",
                direction=[float(value) for value in direction.tolist()],
                index=columns,
                strength=float(STRENGTH),
            )
        ],
    )
    policy = WeightCrackPolicy(
        allow_tensors=["weight"],
        max_entries=10000,
        max_fraction=0.05,
        max_abs_delta=1.0,
    )
    return patch, policy, columns


def score(model, tokenizer, module, matrix: np.ndarray) -> list[dict]:
    import torch

    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(np.array(matrix, dtype=np.float32, copy=True)))
    rows = []
    for item in cap.HOLDOUT:
        rendered = cap.render_prompt(tokenizer, item["prompt"])
        text = cap.generate(model, tokenizer, rendered)
        rows.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "loop": cap.is_loop(text),
            "completion": text,
        })
        print(f"{item['id']} loop={rows[-1]['loop']}", flush=True)
    return rows


def main() -> int:
    if STARTED.exists() or HOLDOUT_OUT.exists() or BUNDLE.exists():
        raise SystemExit("holdout already started")
    matrix = np.load(MATRIX)["weight"]
    patch, policy, columns = build_patch(matrix)
    receipt = apply_patch(
        MATRIX,
        patch,
        policy,
        BUNDLE,
        tensor_name="weight",
        task_id="real-matrix-dense-holdout",
    )
    candidate_hash = receipt["candidate_content_sha256"]
    STARTED.write_text(json.dumps({
        "status": "started",
        "candidate_content_sha256": candidate_hash,
        "selected_columns": columns,
        "strength": STRENGTH,
        "bundle": str(BUNDLE),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"locked": candidate_hash, "columns": columns}), flush=True)
    from huggingface_hub import snapshot_download
    folder = Path(snapshot_download(cap.MODEL_ID, revision=cap.REVISION, cache_dir=str(cap.CACHE)))
    tokenizer = cap.load_tokenizer(folder)
    model, module = cap.load_model(folder, matrix)
    print("baseline", flush=True)
    baseline = score(model, tokenizer, module, matrix)
    with np.load(BUNDLE / "candidate.npz") as handle:
        candidate = np.array(handle["weight"], dtype=np.float32, copy=True)
    print("edited", flush=True)
    edited = score(model, tokenizer, module, candidate)
    base_loops = sum(1 for row in baseline if row["loop"])
    edited_loops = sum(1 for row in edited if row["loop"])
    HOLDOUT_OUT.write_text(json.dumps({
        "status": "opened_once",
        "loop_rule": "a 4-word sequence occurs at least 3 times",
        "max_new_tokens": cap.MAX_NEW_TOKENS,
        "candidate_content_sha256": candidate_hash,
        "selected_columns": columns,
        "strength": STRENGTH,
        "baseline_loops": base_loops,
        "edited_loops": edited_loops,
        "holdout_count": len(cap.HOLDOUT),
        "baseline": baseline,
        "edited": edited,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline_loops": base_loops,
        "edited_loops": edited_loops,
        "holdout_count": len(cap.HOLDOUT),
        "candidate_content_sha256": candidate_hash,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
