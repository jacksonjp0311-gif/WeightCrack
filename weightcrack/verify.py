"""Replay a completed bundle and fail closed on tampering or a torn publish."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bundle import read_completed_bundle, resave_bytes
from .canonical import array_content_sha256, canonical_sha256, f64_from_hex, file_sha256, model_sha256
from .errors import VerificationFailure
from .patches.huggingface import apply_project_remove
from .policy import enforce_realized
from .schema import WeightCrackPatch, WeightCrackPolicy


def _fail(status: str, message: str) -> None:
    raise VerificationFailure(status, message)


def verify_bundle(bundle, source, tensor_name: str, source_path=None) -> dict:
    """Verify hashes, policy, source identity, and exact stored-value reconstruction.

    ``resave_byte_identical`` is measured separately. Equal array values do not
    imply that a new ``np.savez`` reproduces the file bytes.
    """
    loaded = read_completed_bundle(Path(bundle))
    receipt = loaded["receipt"]
    event = loaded["event"]
    if receipt.get("status") != "applied" or receipt.get("disposition") != "applied":
        _fail("rejected", "bundle disposition is not applied")
    if receipt.get("behavioral_qualification") is not None:
        _fail("rejected", "a successful file write is not a behavioral pass")
    if event.get("disposition") != "applied" or event.get("behavioral_qualification") is not None:
        _fail("rejected", "event record disposition does not match an applied edit")
    try:
        patch = WeightCrackPatch.model_validate(receipt["patch"])
        policy = WeightCrackPolicy.model_validate(receipt["policy"])
    except Exception as exc:
        _fail("rejected", f"patch or policy does not validate: {exc}")
    if model_sha256(patch) != receipt.get("patch_sha256"):
        _fail("rejected", "patch hash mismatch")
    if model_sha256(policy) != receipt.get("policy_sha256"):
        _fail("rejected", "policy hash mismatch")
    normalized = receipt.get("normalized_patch")
    if not isinstance(normalized, dict) or canonical_sha256(normalized) != receipt.get("normalized_patch_sha256"):
        _fail("rejected", "normalized patch hash mismatch")
    if patch.changes[0].tensor != tensor_name or normalized.get("tensor") != tensor_name:
        _fail("rejected", "tensor identity mismatch")
    if isinstance(source, np.ndarray):
        source_matrix = source
        source_arrays = {tensor_name: np.array(source, copy=True)}
    else:
        source_path = Path(source)
        with np.load(source_path) as handle:
            if tensor_name not in handle.files:
                _fail("rejected", "source is missing the bound tensor")
            source_arrays = {key: np.array(handle[key], copy=True) for key in handle.files}
        source_matrix = source_arrays[tensor_name]
    if array_content_sha256(source_matrix) != receipt["source_manifest"]["content_sha256"]:
        _fail("rejected", "source tensor identity mismatch")
    if source_path is not None and file_sha256(source_path) != receipt.get("source_file_sha256"):
        _fail("rejected", "source file hash mismatch")
    try:
        direction = f64_from_hex(normalized["direction_f64_hex"])
        strength = float(f64_from_hex(normalized["strength_f64_hex"])[0])
        raw_columns = normalized["columns"]
        if not isinstance(raw_columns, list) or any(type(column) is not int for column in raw_columns):
            _fail("rejected", "normalized columns are not integers")
        columns = raw_columns
    except VerificationFailure:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        _fail("rejected", f"normalized patch cannot be decoded: {exc}")
    submitted = np.asarray(patch.changes[0].direction, dtype=np.float64)
    if not np.array_equal(submitted, direction):
        _fail("rejected", "submitted direction does not match the normalized patch")
    if float(patch.changes[0].strength) != strength or list(patch.changes[0].index) != columns:
        _fail("rejected", "submitted patch does not match the normalized patch")
    try:
        reconstructed, accounting = apply_project_remove(source_matrix, direction, columns, strength)
        enforce_realized(accounting, policy, int(source_matrix.size))
    except VerificationFailure:
        raise
    except Exception as exc:
        _fail("rejected", f"reconstruction rejected the saved edit: {exc}")
    saved = loaded["arrays"].get(tensor_name)
    if saved is None or not np.array_equal(reconstructed, saved):
        _fail("rejected", "reconstruction does not match the stored tensor")
    if saved.dtype != source_matrix.dtype:
        _fail("rejected", "stored dtype does not match the source")
    for name, value in source_arrays.items():
        if name == tensor_name:
            continue
        if name not in loaded["arrays"] or not np.array_equal(loaded["arrays"][name], value):
            _fail("rejected", f"untouched tensor changed: {name}")
    if accounting != receipt.get("accounting"):
        _fail("rejected", "stored accounting does not match reconstruction")
    resaved = resave_bytes(loaded["arrays"])
    return {
        "status": "verified",
        "disposition": "applied",
        "behavioral_qualification": None,
        "value_exact_reconstruction": True,
        "resave_byte_identical": resaved == loaded["candidate_bytes"],
        "candidate_file_sha256": receipt["candidate_file_sha256"],
        "source_content_sha256": receipt["source_manifest"]["content_sha256"],
        "accounting": accounting,
    }
