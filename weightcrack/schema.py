"""
WeightCrack core schemas.

Patch and policy contracts that the agent proposes against and the host enforces.
Column indexes and numeric fields are not coerced: True, "0", and 1.0 are rejected.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .canonical import model_sha256


def _reject_loose_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _reject_loose_real(value: object, label: str) -> int | float:
    if type(value) not in {int, float}:
        raise ValueError(f"{label} must be a real number")
    return value


class PatchChange(BaseModel):
    tensor: str
    op: str
    index: Optional[list[int]] = None
    value: Optional[float] = None
    direction: Optional[list[float]] = None
    strength: Optional[float] = None
    rank: Optional[int] = None

    @field_validator("op")
    @classmethod
    def operation_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("op must be a non-empty string")
        return value

    @field_validator("index", mode="before")
    @classmethod
    def strict_index(cls, value):
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("columns must be a list of integers")
        return [_reject_loose_int(item, "column") for item in value]

    @field_validator("direction", mode="before")
    @classmethod
    def strict_direction(cls, value):
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("direction must be a list of real numbers")
        return [_reject_loose_real(item, "direction entry") for item in value]

    @field_validator("value", "strength", mode="before")
    @classmethod
    def strict_optional_real(cls, value):
        if value is None:
            return None
        return _reject_loose_real(value, "numeric field")

    @field_validator("rank", mode="before")
    @classmethod
    def strict_rank(cls, value):
        if value is None:
            return None
        return _reject_loose_int(value, "rank")


class WeightCrackPatch(BaseModel):
    schema_id: str = "weightcrack.patch.v1"
    base_sha256: str
    reason: str
    sector: str
    evidence_ids: list[str] = Field(default_factory=list)
    changes: list[PatchChange]

    @field_validator("schema_id")
    @classmethod
    def schema_name(cls, value: str) -> str:
        if value != "weightcrack.patch.v1":
            raise ValueError("unsupported patch schema")
        return value

    def compute_hash(self) -> str:
        """Hash the complete patch, including the base identity."""
        return model_sha256(self)


class WeightCrackPolicy(BaseModel):
    schema_id: str = "weightcrack.policy.v1"
    allow_tensors: list[str]
    max_entries: int = 64
    max_fraction: float = 0.01
    max_abs_delta: float = 1.0
    max_rank: int = 8
    allowed_ops: list[str] = Field(
        default_factory=lambda: ["set", "add", "scale", "project_remove"]
    )

    @field_validator("schema_id")
    @classmethod
    def schema_name(cls, value: str) -> str:
        if value != "weightcrack.policy.v1":
            raise ValueError("unsupported policy schema")
        return value

    @field_validator("max_entries", "max_rank", mode="before")
    @classmethod
    def strict_counts(cls, value):
        return _reject_loose_int(value, "policy count")

    @field_validator("max_fraction", "max_abs_delta", mode="before")
    @classmethod
    def strict_limits(cls, value):
        return _reject_loose_real(value, "policy limit")

    def allows(self, change: PatchChange) -> bool:
        if change.tensor not in self.allow_tensors:
            return False
        if change.op not in self.allowed_ops:
            return False
        return True
