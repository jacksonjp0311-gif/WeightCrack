"""Extract one Qwen MLP down-projection into experiments/real_matrix.

Uses a local snapshot when it is already cached. A failed download is written
to docs/REAL_MATRIX_STATUS.md and the process stops.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weightcrack.canonical import array_content_sha256, file_sha256

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
CACHE = ROOT / ".venv-real" / "hf-cache"
OUT = ROOT / "experiments" / "real_matrix" / "mlp_down_proj.npz"
META = ROOT / "experiments" / "real_matrix" / "matrix_meta.json"
STATUS = ROOT / "docs" / "REAL_MATRIX_STATUS.md"


def snapshot_file() -> Path:
    return (
        CACHE
        / "models--Qwen--Qwen2.5-0.5B"
        / "snapshots"
        / REVISION
        / "model.safetensors"
    )


def ensure_checkpoint() -> Path:
    path = snapshot_file()
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                MODEL_ID,
                "model.safetensors",
                revision=REVISION,
                cache_dir=str(CACHE),
            )
        )
    except Exception as exc:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        block = (
            "\n\n## Download attempt during real-matrix extraction\n\n"
            f"Model: {MODEL_ID} revision {REVISION}.\n\n"
            "Result: DOWNLOAD_FAILED.\n\n"
            "```\n"
            f"{exc.__class__.__name__}: {exc}\n"
            "```\n\n"
            "```\n"
            f"{traceback.format_exc()}\n"
            "```\n\n"
            "No new matrix was saved by this attempt. The extraction stopped.\n"
        )
        previous = STATUS.read_text(encoding="utf-8") if STATUS.exists() else ""
        STATUS.write_text(previous + block, encoding="utf-8")
        print(f"DOWNLOAD_FAILED {exc.__class__.__name__}: {exc}", flush=True)
        raise SystemExit(1) from exc
    if not downloaded.is_file():
        raise SystemExit(1)
    return downloaded


def choose_tensor(path: Path):
    from safetensors import safe_open

    chosen = None
    chosen_index = -1
    with safe_open(path, framework="pt") as handle:
        for key in handle.keys():
            if not key.endswith("mlp.down_proj.weight"):
                continue
            marker = ".layers."
            start = key.find(marker)
            if start < 0:
                continue
            layer_text = key[start + len(marker):].split(".", 1)[0]
            if not layer_text.isdigit():
                continue
            layer_index = int(layer_text)
            if layer_index > chosen_index:
                chosen = key
                chosen_index = layer_index
    if chosen is None:
        raise RuntimeError("no mlp.down_proj.weight key in the checkpoint")
    return chosen, chosen_index


def main() -> int:
    path = ensure_checkpoint()
    key, layer_index = choose_tensor(path)
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        tensor = handle.get_tensor(key)
        original_dtype = str(tensor.dtype).replace("torch.", "")
        matrix = tensor.detach().float().contiguous().cpu().numpy()
    if matrix.dtype != np.float32 or matrix.ndim != 2:
        raise RuntimeError("host cast did not produce a float32 matrix")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, weight=matrix)
    meta = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "checkpoint_file_sha256": file_sha256(path),
        "source_key": key,
        "layer_index": layer_index,
        "checkpoint_dtype": original_dtype,
        "stored_dtype": "float32",
        "shape": list(matrix.shape),
        "content_sha256": array_content_sha256(matrix),
        "npz": str(OUT),
        "npz_key": "weight",
    }
    META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"key": key, "shape": meta["shape"], "content_sha256": meta["content_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
