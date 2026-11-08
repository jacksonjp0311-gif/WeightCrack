"""Host-side policy gate.

Budgets count distinct scalar coordinates, not operation records.
``max_abs_delta`` is compared with the realized stored values.
"""

from __future__ import annotations

import math

from .errors import PolicyViolation
from .schema import WeightCrackPatch, WeightCrackPolicy

SUPPORTED_OPERATION = "project_remove"


def validate_policy_limits(policy: WeightCrackPolicy) -> None:
    entries = policy.max_entries
    if type(entries) is not int or entries < 1:
        raise PolicyViolation("invalid max_entries")
    fraction = policy.max_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise PolicyViolation("invalid max_fraction")
    if not math.isfinite(float(fraction)) or not 0 < float(fraction) <= 1:
        raise PolicyViolation("invalid max_fraction")
    delta = policy.max_abs_delta
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise PolicyViolation("invalid max_abs_delta")
    if not math.isfinite(float(delta)) or float(delta) <= 0:
        raise PolicyViolation("invalid max_abs_delta")
    names = policy.allow_tensors
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise PolicyViolation("allow_tensors must name at least one tensor")
    if len(set(names)) != len(names):
        raise PolicyViolation("allow_tensors contains a duplicate")


def gate_patch(patch: WeightCrackPatch, policy: WeightCrackPolicy, base_param_count: int) -> None:
    """Structural gate. Coordinate and stored-delta checks happen after construction."""
    validate_policy_limits(policy)
    if type(base_param_count) is not int or base_param_count < 1:
        raise PolicyViolation("invalid fraction denominator")
    if not patch.changes:
        raise PolicyViolation("empty patch")
    if len(patch.changes) != 1:
        raise PolicyViolation("this iteration accepts exactly one change")
    change = patch.changes[0]
    if change.op != SUPPORTED_OPERATION:
        raise PolicyViolation(f"unsupported operation: {change.op}")
    if not policy.allows(change):
        raise PolicyViolation(f"disallowed change on {change.tensor} op={change.op}")


def enforce_realized(accounting: dict, policy: WeightCrackPolicy, base_param_count: int | None = None) -> None:
    """Enforce budgets on the realized candidate."""
    validate_policy_limits(policy)
    touched = accounting["touched_coordinates"]
    denominator = accounting["fraction_denominator"]
    if type(touched) is not int or touched < 1:
        raise PolicyViolation("touched coordinate count is invalid")
    if type(denominator) is not int or denominator < 1:
        raise PolicyViolation("fraction denominator is invalid")
    if base_param_count is not None and base_param_count != denominator:
        raise PolicyViolation("fraction denominator does not match the edited tensor")
    if touched > policy.max_entries:
        raise PolicyViolation(f"touched coordinates {touched} exceed max_entries {policy.max_entries}")
    fraction = touched / denominator
    if fraction > float(policy.max_fraction):
        raise PolicyViolation(f"fraction {fraction:.6f} exceeds max_fraction")
    stored_delta = accounting["max_abs_stored_delta"]
    if not math.isfinite(stored_delta) or stored_delta > float(policy.max_abs_delta):
        raise PolicyViolation(
            f"realized max delta {stored_delta} exceeds max_abs_delta"
        )
