"""Replay a completed bundle and fail closed on tampering or a torn publish."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bundle import read_completed_bundle, resave_bytes
from .canonical import (
    array_content_sha256,
    canonical_sha256,
    f64_from_hex,
    file_sha256,
    model_sha256,
    tensor_manifest,
)
from .errors import EditRejected, VerificationFailure
from .patches.huggingface import construct_candidate
from .policy import PolicyViolation
from .schema import WeightCrackPatch, WeightCrackPolicy


def _fail(status: str, message: str) -> None:
    raise VerificationFailure(status, message)


def verify_bundle(bundle, source, tensor_name: str, source_path=None) -> dict:
    """Verify the application contract and exact stored-value reconstruction.

    Reconstruction uses ``construct_candidate``, the same gate as a normal
    application. ``resave_byte_identical`` is a fresh ``np.savez`` comparison.
    Zip creator metadata can differ across operating systems, so that boolean
    is an observation. It is not proof that the stored values differ, and an
    unsigned file hash does not show who authorized a rewritten archive.
    """
    loaded = read_completed_bundle(Path(bundle))
    receipt = loaded["receipt"]
    event = loaded["event"]
    if receipt.get("status") != "applied" or receipt.get("disposition") != "applied":
        _fail("rejected", "bundle disposition is not applied")
    if receipt.get("behavioral_qualification") is not None:
        _fail("rejected", "a successful file write is not a behavioral pass")
    if event.get("schema") != "weightcrack.event.v1":
        _fail("rejected", "event schema is not weightcrack.event.v1")
    if event.get("disposition") != "applied" or event.get("behavioral_qualification") is not None:
        _fail("rejected", "event record disposition does not match an applied edit")
    if event.get("status") != "applied":
        _fail("rejected", "event status is not applied")
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
    if len(patch.changes) != 1:
        _fail("rejected", "bundle does not contain exactly one change")
    if patch.changes[0].tensor != tensor_name or normalized.get("tensor") != tensor_name:
        _fail("rejected", "tensor identity mismatch")
    if event.get("task_id") != receipt.get("task_id"):
        _fail("rejected", "event task id does not match the receipt")
    if event.get("tensor") != tensor_name:
        _fail("rejected", "event tensor does not match the bound tensor")
    if event.get("patch_sha256") != receipt.get("patch_sha256"):
        _fail("rejected", "event patch hash does not match the receipt")
    if event.get("policy_sha256") != receipt.get("policy_sha256"):
        _fail("rejected", "event policy hash does not match the receipt")
    if event.get("candidate_file_sha256") != receipt.get("candidate_file_sha256"):
        _fail("rejected", "event candidate file hash does not match the receipt")
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
    if set(loaded["arrays"]) != set(source_arrays):
        _fail("rejected", "candidate tensor names do not match the source")
    source_record = tensor_manifest(tensor_name, source_matrix)
    if receipt.get("source_manifest") != source_record:
        _fail("rejected", "source manifest does not match the supplied tensor")
    if receipt.get("base_sha256") != source_record["content_sha256"] or patch.base_sha256 != source_record["content_sha256"]:
        _fail("rejected", "submitted base identity does not match the supplied source")
    if source_path is not None and file_sha256(source_path) != receipt.get("source_file_sha256"):
        _fail("rejected", "source file hash mismatch")
    try:
        direction = f64_from_hex(normalized["direction_f64_hex"])
        strength = float(f64_from_hex(normalized["strength_f64_hex"])[0])
        raw_columns = normalized["columns"]
        if not isinstance(raw_columns, list) or any(type(column) is not int for column in raw_columns):
            _fail("rejected", "normalized columns are not integers")
    except VerificationFailure:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        _fail("rejected", f"normalized patch cannot be decoded: {exc}")
    submitted = np.asarray(patch.changes[0].direction, dtype=np.float64)
    if not np.array_equal(submitted, direction):
        _fail("rejected", "submitted direction does not match the normalized patch")
    if float(patch.changes[0].strength) != strength or list(patch.changes[0].index) != raw_columns:
        _fail("rejected", "submitted patch does not match the normalized patch")
    try:
        reconstructed, accounting, regenerated = construct_candidate(
            source_matrix, patch, policy, tensor_name
        )
    except VerificationFailure:
        raise
    except (EditRejected, PolicyViolation) as exc:
        _fail("rejected", f"application contract rejected the saved edit: {exc}")
    except Exception as exc:
        _fail("rejected", f"reconstruction rejected the saved edit: {exc}")
    if canonical_sha256(regenerated) != receipt.get("normalized_patch_sha256"):
        _fail("rejected", "regenerated normalized patch does not match the receipt")
    saved = loaded["arrays"].get(tensor_name)
    if saved is None or saved.dtype != reconstructed.dtype or not np.array_equal(reconstructed, saved):
        _fail("rejected", "reconstruction does not match the stored tensor")
    if saved.dtype != source_matrix.dtype or saved.shape != source_matrix.shape:
        _fail("rejected", "stored dtype or shape does not match the source")
    candidate_record = tensor_manifest(tensor_name, saved)
    if receipt.get("candidate_manifest") != candidate_record:
        _fail("rejected", "candidate manifest does not match the stored tensor")
    if receipt.get("candidate_content_sha256") != candidate_record["content_sha256"]:
        _fail("rejected", "candidate content hash does not match the stored tensor")
    if array_content_sha256(saved) != candidate_record["content_sha256"]:
        _fail("rejected", "candidate content hash does not match the stored tensor")
    untouched_records = [
        tensor_manifest(name, value) for name, value in source_arrays.items() if name != tensor_name
    ]
    if receipt.get("untouched_tensor_manifests") != untouched_records:
        _fail("rejected", "untouched tensor manifest does not match the source")
    for name, value in source_arrays.items():
        stored = loaded["arrays"][name]
        if stored.dtype != value.dtype or stored.shape != value.shape:
            _fail("rejected", f"tensor dtype or shape changed: {name}")
        if name == tensor_name:
            continue
        if not np.array_equal(stored, value):
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
        "source_content_sha256": source_record["content_sha256"],
        "candidate_content_sha256": candidate_record["content_sha256"],
        "accounting": accounting,
    }
