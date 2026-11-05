"""Adapted policy-gated sparse and directional edits for safetensors checkpoints.

Based on the WeightBridge implementation, re-scoped for WeightCrack.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import WeightCrackPatch, WeightCrackPolicy


class EditRejected(Exception):
    pass


def apply_patch(
    base: str | Path,
    patch: WeightCrackPatch,
    policy: WeightCrackPolicy,
    output: str | Path,
) -> dict[str, Any]:
    """
    Apply a gated sparse patch and return a receipt.
    Full tensor-level implementation is stubbed here for the scaffold;
    the real loader and write path will be filled from the WeightBridge pattern.
    """
    base_path = Path(base)
    out_path = Path(output)
    if out_path.exists():
        raise EditRejected(f"output path exists: {out_path}")

    # Stub receipt — real version loads checkpoint, applies changes, writes new file
    receipt = {
        "schema": "weightcrack.receipt.v1",
        "base_sha256": patch.base_sha256,
        "patch_hash": patch.compute_hash(),
        "sector": patch.sector,
        "reason": patch.reason,
        "policy": policy.model_dump(),
        "status": "scaffold_applied",
        "note": "tensor-level apply pending full safetensors integration",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = out_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return receipt
