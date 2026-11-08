"""Canonical hashes for tensors, policies, and patches.

Array hashes cover contiguous payload bytes. Shape and dtype live in the
manifest so two different arrays cannot hide behind the same raw hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from pydantic import BaseModel


def canonical_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def tensor_manifest(name: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "content_sha256": array_content_sha256(array),
    }


def f64_hex(values: np.ndarray) -> str:
    raw = np.ascontiguousarray(values, dtype="<f8")
    return raw.tobytes().hex()


def f64_from_hex(text: str) -> np.ndarray:
    data = bytes.fromhex(text)
    if len(data) % 8 != 0:
        raise ValueError("direction encoding is not a sequence of float64 values")
    return np.frombuffer(data, dtype="<f8").astype(np.float64, copy=True)


def model_payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def model_sha256(model: BaseModel) -> str:
    return canonical_sha256(model_payload(model))
