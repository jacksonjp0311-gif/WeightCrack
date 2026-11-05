"""
Sparse patch applicator (stub).
Applies a gated patch to a checkpoint copy and emits a receipt.
"""

from pathlib import Path
from typing import Any
import hashlib
import json
from .schema import WeightCrackPatch


def apply_patch(
    base_path: Path,
    patch: WeightCrackPatch,
    out_path: Path,
    policy_hash: str,
) -> dict[str, Any]:
    """
    Write a new candidate and return a receipt.
    Real implementation loads safetensors/npz, applies changes, writes new file.
    """
    # Stub: just record the intent
    receipt = {
        "schema": "weightcrack.receipt.v1",
        "base_sha256": patch.base_sha256,
        "patch_hash": patch.compute_hash(),
        "policy_hash": policy_hash,
        "sector": patch.sector,
        "reason": patch.reason,
        "out_path": str(out_path),
        "status": "stub_applied",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = out_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return receipt
