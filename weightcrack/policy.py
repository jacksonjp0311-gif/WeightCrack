"""
Host-side policy gate.
Counts coordinates, not operation records.
"""

from .schema import WeightCrackPatch, WeightCrackPolicy


class PolicyViolation(Exception):
    pass


def gate_patch(patch: WeightCrackPatch, policy: WeightCrackPolicy, base_param_count: int) -> None:
    """Structural gate before construction. Realized checks happen after."""
    if not patch.changes:
        raise PolicyViolation("empty patch")
    if len(patch.changes) != 1:
        raise PolicyViolation("this iteration accepts exactly one change")
    change = patch.changes[0]
    if not policy.allows(change):
        raise PolicyViolation(f"disallowed change on {change.tensor} op={change.op}")
    if change.op != "project_remove":
        raise PolicyViolation("only project_remove is supported in this iteration")


def enforce_realized(accounting: dict, policy: WeightCrackPolicy, base_param_count: int) -> None:
    """Enforce budgets on the realized candidate."""
    touched = accounting["touched_coordinates"]
    if touched > policy.max_entries:
        raise PolicyViolation(f"touched coordinates {touched} exceed max_entries {policy.max_entries}")
    fraction = touched / max(base_param_count, 1)
    if fraction > policy.max_fraction:
        raise PolicyViolation(f"fraction {fraction:.6f} exceeds max_fraction")
    if accounting["max_abs_stored_delta"] > policy.max_abs_delta:
        raise PolicyViolation(
            f"realized max delta {accounting['max_abs_stored_delta']} exceeds max_abs_delta"
        )
