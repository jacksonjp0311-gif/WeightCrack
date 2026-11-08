"""Exclusive candidate bundles.

A bundle is a new directory. Candidate bytes, the receipt, and the event
record are written first. The completion manifest is published last. Readers
reject a directory that has no valid manifest. Removing the directory on a
failed attempt is cleanup of this attempt only; it is not a transaction that
covers any other file.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_dumps, file_sha256
from .errors import EditRejected, VerificationFailure

BUNDLE_SUFFIX = ".wcbundle"
CANDIDATE_NAME = "candidate.npz"
RECEIPT_NAME = "receipt.json"
EVENTS_NAME = "events.jsonl"
MANIFEST_NAME = "manifest.json"
REQUIRED_FILES = (CANDIDATE_NAME, RECEIPT_NAME, EVENTS_NAME)


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def exclusive_create_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise EditRejected(f"destination already exists: {path}")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(str(path), flags)
    try:
        view = payload
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Give np.savez a file handle so it cannot append a suffix to the path."""
    if path.exists() or path.is_symlink():
        raise EditRejected(f"destination already exists: {path}")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(str(path), flags)
    owned = fd
    try:
        with os.fdopen(owned, "wb") as handle:
            owned = -1
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if owned != -1:
            os.close(owned)


def write_completion_manifest(path: Path, manifest: dict[str, Any]) -> None:
    exclusive_create_bytes(path, json_bytes(manifest))


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": file_sha256(path), "nbytes": path.stat().st_size}


def publish_bundle(
    bundle: Path,
    arrays: dict[str, np.ndarray],
    receipt: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    """Create ``bundle`` and publish its manifest last.

    On failure, delete only this directory when this call created it.
    """
    if bundle.exists() or bundle.is_symlink():
        raise EditRejected(f"destination already exists: {bundle}")
    if bundle.suffix != BUNDLE_SUFFIX:
        raise EditRejected("output bundle must use an explicit .wcbundle directory name")
    created = False
    try:
        os.mkdir(bundle)
        created = True
        exclusive_savez(bundle / CANDIDATE_NAME, arrays)
        # The receipt names the saved candidate bytes, so it follows that write.
        receipt = dict(receipt)
        receipt["candidate_file_sha256"] = file_sha256(bundle / CANDIDATE_NAME)
        receipt["candidate_nbytes"] = (bundle / CANDIDATE_NAME).stat().st_size
        event = dict(event)
        event["candidate_file_sha256"] = receipt["candidate_file_sha256"]
        exclusive_create_bytes(bundle / RECEIPT_NAME, json_bytes(receipt))
        exclusive_create_bytes(bundle / EVENTS_NAME, json_bytes(event))
        manifest = {
            "schema": "weightcrack.bundle.v1",
            "status": "complete",
            "files": {name: _file_record(bundle / name) for name in REQUIRED_FILES},
        }
        write_completion_manifest(bundle / MANIFEST_NAME, manifest)
        return receipt
    except Exception:
        if created:
            shutil.rmtree(bundle)
        raise


def read_completed_bundle(bundle: Path) -> dict[str, Any]:
    """Load a bundle or fail closed when the completion manifest is absent or stale."""
    manifest_path = bundle / MANIFEST_NAME
    if not bundle.is_dir() or bundle.is_symlink() or not manifest_path.is_file():
        raise VerificationFailure("incomplete", "bundle has no completion manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationFailure("incomplete", "completion manifest is not JSON") from exc
    if manifest.get("schema") != "weightcrack.bundle.v1" or manifest.get("status") != "complete":
        raise VerificationFailure("incomplete", "completion manifest is not complete")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(REQUIRED_FILES):
        raise VerificationFailure("rejected", "completion manifest file set is wrong")
    present = {path.name for path in bundle.iterdir() if path.is_file()}
    if present != set(REQUIRED_FILES) | {MANIFEST_NAME}:
        raise VerificationFailure("rejected", "bundle contains an unexpected file set")
    loaded_files = {}
    for name, record in files.items():
        path = bundle / name
        if not path.is_file() or path.is_symlink():
            raise VerificationFailure("incomplete", f"missing {name}")
        digest = file_sha256(path)
        if digest != record.get("sha256") or path.stat().st_size != record.get("nbytes"):
            raise VerificationFailure("rejected", f"hash mismatch for {name}")
        loaded_files[name] = path.read_bytes()
    try:
        receipt = json.loads(loaded_files[RECEIPT_NAME].decode("utf-8"))
        event = json.loads(loaded_files[EVENTS_NAME].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationFailure("rejected", "receipt or event record is not JSON") from exc
    with np.load(bundle / CANDIDATE_NAME) as handle:
        arrays = {key: np.array(handle[key], copy=True) for key in handle.files}
    if receipt.get("candidate_file_sha256") != files[CANDIDATE_NAME]["sha256"]:
        raise VerificationFailure("rejected", "receipt candidate hash does not match the saved file")
    if event.get("candidate_file_sha256") != receipt.get("candidate_file_sha256"):
        raise VerificationFailure("rejected", "event record does not match the saved candidate")
    return {
        "manifest": manifest,
        "receipt": receipt,
        "event": event,
        "arrays": arrays,
        "candidate_bytes": loaded_files[CANDIDATE_NAME],
    }


def resave_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize arrays again. Zip metadata can differ from the saved file."""
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()
