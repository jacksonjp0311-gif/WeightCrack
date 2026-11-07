"""Minimal agent session under governance.

Exposes only practice-only tools. Host gates, applies, and evaluates.
No holdout access for the agent.
"""

from __future__ import annotations

from typing import Any, Callable
import hashlib
import json
from pathlib import Path

import numpy as np

from ..schema import WeightCrackPatch, WeightCrackPolicy, PatchChange
from ..policy import gate_patch
from ..patches.huggingface import apply_patch
from .localization import activation_difference, rank_columns, direction_trial


class Session:
    def __init__(
        self,
        matrix: np.ndarray,
        practice_fail: np.ndarray,
        practice_success: np.ndarray,
        policy: WeightCrackPolicy,
        practice_score_fn: Callable[[np.ndarray], float],
    ):
        self.matrix = matrix
        self.practice_fail = practice_fail
        self.practice_success = practice_success
        self.policy = policy
        self.practice_score_fn = practice_score_fn
        self.base_hash = hashlib.sha256(matrix.tobytes()).hexdigest()
        self.ledger: list[dict[str, Any]] = []

    def tools(self) -> dict[str, Any]:
        """Practice-only tools the agent may call."""
        diff = activation_difference(self.practice_fail, self.practice_success)
        ranked = rank_columns(self.matrix, diff["direction"])
        return {
            "activation_difference": diff,
            "rank_columns": ranked,
            "base_hash": self.base_hash,
            "allowed_tensors": self.policy.allow_tensors,
        }

    def trial(self, columns: list[int], strength: float) -> dict[str, Any]:
        diff = activation_difference(self.practice_fail, self.practice_success)
        return direction_trial(
            self.matrix,
            diff["direction"],
            columns,
            strength,
            self.practice_score_fn,
            policy=self.policy,
        )

    def propose_and_gate(
        self,
        columns: list[int],
        strength: float,
        out_path: Path,
    ) -> dict[str, Any]:
        """Agent proposes; host gates and applies if accepted."""
        diff = activation_difference(self.practice_fail, self.practice_success)
        patch = WeightCrackPatch(
            base_sha256=self.base_hash,
            reason="agent-proposed direction removal",
            sector="math_loop",
            changes=[
                PatchChange(
                    tensor=self.policy.allow_tensors[0],
                    op="project_remove",
                    index=columns,
                    direction=diff["direction"],
                    strength=strength,
                )
            ],
        )
        try:
            gate_patch(patch, self.policy, self.matrix.size)
            receipt = apply_patch(self.matrix, patch, self.policy, out_path)
            record = {"status": "applied", "receipt": receipt}
        except Exception as exc:
            record = {"status": "rejected", "error": str(exc)}
        self.ledger.append(record)
        return record
