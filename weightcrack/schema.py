"""
WeightCrack core schemas.

Patch and policy contracts that the agent proposes against and the host enforces.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import hashlib


class PatchChange(BaseModel):
    tensor: str
    op: Literal["set", "add", "scale", "project_remove", "low_rank"]
    index: Optional[list[int]] = None
    value: Optional[float] = None
    direction: Optional[list[float]] = None
    strength: Optional[float] = None
    rank: Optional[int] = None

    @field_validator("op")
    @classmethod
    def check_op_fields(cls, v, info):
        # Basic validation; host will enforce full constraints
        return v


class WeightCrackPatch(BaseModel):
    schema_id: Literal["weightcrack.patch.v1"] = "weightcrack.patch.v1"
    base_sha256: str
    reason: str
    sector: str  # e.g. "math", "coding", "refusal", "style"
    evidence_ids: list[str] = Field(default_factory=list)
    changes: list[PatchChange]

    def compute_hash(self) -> str:
        payload = self.model_dump_json(exclude={"base_sha256"}).encode()
        return hashlib.sha256(payload).hexdigest()


class WeightCrackPolicy(BaseModel):
    schema_id: Literal["weightcrack.policy.v1"] = "weightcrack.policy.v1"
    allow_tensors: list[str]
    max_entries: int = 64
    max_fraction: float = 0.01
    max_abs_delta: float = 1.0
    max_rank: int = 8
    allowed_ops: list[str] = Field(default_factory=lambda: ["set", "add", "scale", "project_remove"])

    def allows(self, change: PatchChange) -> bool:
        if change.tensor not in self.allow_tensors:
            return False
        if change.op not in self.allowed_ops:
            return False
        return True
