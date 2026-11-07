"""Attempt to load one MLP matrix from a public checkpoint for the transfer probe.

If the download succeeds, the matrix is saved under experiments/real_matrix/.
If it fails, the reason is written and the probe is not run.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import safetensors.numpy as st


def try_load_one_matrix(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B",
            filename="model.safetensors",
            cache_dir=str(out_dir / "cache"),
        )
        tensors = st.load_file(path)
        # Prefer a later-layer down_proj; fall back to any down_proj
        key = None
        for k in sorted(tensors.keys()):
            if "mlp.down_proj.weight" in k:
                key = k
                break
        if key is None:
            return {"status": "NO_MATRIX", "reason": "no down_proj found"}
        matrix = tensors[key]
        npz_path = out_dir / "mlp_down_proj.npz"
        import numpy as np
        np.savez(npz_path, weight=matrix.astype("float32"))
        return {
            "status": "loaded",
            "key": key,
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "path": str(npz_path),
        }
    except Exception as exc:
        return {"status": "DOWNLOAD_FAILED", "reason": str(exc)}


if __name__ == "__main__":
    result = try_load_one_matrix(Path("experiments/real_matrix"))
    print(json.dumps(result, indent=2))
    Path("experiments/real_matrix/load_status.json").write_text(json.dumps(result, indent=2))
