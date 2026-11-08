"""Host-owned practice session.

Task identity, tensor binding, source identity, policy, sector, and output
root come from ``SessionConfig``. A proposal selects columns and strength.
It cannot replace the host path, policy, sector, or task.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..canonical import array_content_sha256
from ..errors import EditRejected
from ..patches.huggingface import apply_patch
from ..policy import PolicyViolation
from ..schema import PatchChange, WeightCrackPatch, WeightCrackPolicy
from .localization import activation_difference, direction_trial, rank_columns


@dataclass(frozen=True)
class SessionConfig:
    task_id: str
    tensor_name: str
    sector: str
    policy: WeightCrackPolicy
    output_root: Path


class Session:
    def __init__(
        self,
        matrix: np.ndarray,
        practice_fail: np.ndarray,
        practice_success: np.ndarray,
        config: SessionConfig,
        practice_score_fn: Callable[[np.ndarray], float],
    ):
        if not isinstance(config, SessionConfig):
            raise TypeError("Session requires a host SessionConfig")
        if not isinstance(config.task_id, str) or not config.task_id:
            raise EditRejected("task_id is required")
        if any(character in config.task_id for character in "\\/:"):
            raise EditRejected("task_id cannot contain a path separator")
        if not isinstance(config.tensor_name, str) or not config.tensor_name:
            raise EditRejected("tensor_name is required")
        if not isinstance(config.sector, str) or not config.sector:
            raise EditRejected("sector is required")
        self.task_id = config.task_id
        self.tensor_name = config.tensor_name
        self.sector = config.sector
        self.policy = WeightCrackPolicy.model_validate(config.policy.model_dump())
        self.output_root = Path(config.output_root)
        if not self.output_root.is_dir():
            raise EditRejected("output root must be an existing host directory")
        self.matrix = np.array(matrix, copy=True)
        self.practice_fail = np.array(practice_fail, copy=True)
        self.practice_success = np.array(practice_success, copy=True)
        self.practice_score_fn = practice_score_fn
        self.base_hash = array_content_sha256(self.matrix)
        self.ledger: list[dict[str, Any]] = []
        self._attempt = 0

    def _direction(self) -> list[float]:
        report = activation_difference(self.practice_fail, self.practice_success)
        if report["status"] != "ok":
            raise EditRejected("NO_LOCALIZATION_SIGNAL")
        if len(report["direction"]) != self.matrix.shape[0]:
            raise EditRejected("activation direction does not match the edited matrix")
        return report["direction"]

    def _patch(self, columns: list[int], strength: float) -> WeightCrackPatch:
        return WeightCrackPatch(
            base_sha256=self.base_hash,
            reason="practice direction removal",
            sector=self.sector,
            evidence_ids=[self.task_id],
            changes=[
                PatchChange(
                    tensor=self.tensor_name,
                    op="project_remove",
                    index=list(columns),
                    direction=self._direction(),
                    strength=strength,
                )
            ],
        )

    def _record(self, record: dict[str, Any]) -> None:
        self.ledger.append(record)
        path = self.output_root / "session_events.jsonl"
        line = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd = os.open(str(path), flags)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    def tools(self) -> dict[str, Any]:
        """Practice-only reports. Holdout paths are not accepted or returned."""
        diff = activation_difference(self.practice_fail, self.practice_success)
        ranked = None
        if diff["status"] == "ok":
            ranked = rank_columns(self.matrix, diff["direction"])
        return {
            "activation_difference": diff,
            "rank_columns": ranked,
            "base_hash": self.base_hash,
            "task_id": self.task_id,
            "tensor_name": self.tensor_name,
            "allowed_tensors": list(self.policy.allow_tensors),
        }

    def trial(self, columns: list[int], strength: float) -> dict[str, Any]:
        try:
            result = direction_trial(
                self.matrix,
                self._direction(),
                columns,
                strength,
                self.practice_score_fn,
                self.policy,
                self.tensor_name,
            )
        except (EditRejected, PolicyViolation) as exc:
            self._record(
                {
                    "schema": "weightcrack.event.v1",
                    "status": "rejected",
                    "disposition": "rejected",
                    "behavioral_qualification": None,
                    "task_id": self.task_id,
                    "stage": "trial",
                    "error": str(exc),
                }
            )
            raise
        return result

    def propose(self, columns: list[int], strength: float) -> dict[str, Any]:
        """Apply under the frozen host configuration. The caller does not choose a path."""
        self._attempt += 1
        bundle = self.output_root / f"{self.task_id}-{self._attempt:04d}.wcbundle"
        try:
            patch = self._patch(columns, strength)
            receipt = apply_patch(
                self.matrix,
                patch,
                self.policy,
                bundle,
                tensor_name=self.tensor_name,
                task_id=self.task_id,
            )
        except (EditRejected, PolicyViolation) as exc:
            record = {
                "schema": "weightcrack.event.v1",
                "status": "rejected",
                "disposition": "rejected",
                "behavioral_qualification": None,
                "task_id": self.task_id,
                "stage": "propose",
                "error": str(exc),
            }
            self._record(record)
            return record
        record = {
            "schema": "weightcrack.event.v1",
            "status": "applied",
            "disposition": "applied",
            "behavioral_qualification": None,
            "task_id": self.task_id,
            "bundle": str(bundle),
            "candidate_file_sha256": receipt["candidate_file_sha256"],
        }
        self._record(record)
        return {"status": "applied", "disposition": "applied", "receipt": receipt, "bundle": str(bundle)}
