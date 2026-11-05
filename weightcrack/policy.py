"""
Host-side policy gate.
Validates a proposed patch against the active policy before any application.
"""

from .schema import WeightCrackPatch, WeightCrackPolicy, PatchChange


class PolicyViolation(Exception):
    pass


def gate_patch(patch: WeightCrackPatch, policy: WeightCrackPolicy, base_param_count: int) -> None:
    """
    Enforce policy. Raises PolicyViolation on any breach.
    Does not mutate anything.
    """
    if not patch.changes:
        raise PolicyViolation("empty patch")

    if len(patch.changes) > policy.max_entries:
        raise PolicyViolation(f"too many entries: {len(patch.changes)} > {policy.max_entries}")

    changed_estimate = 0
    for change in patch.changes:
        if not policy.allows(change):
            raise PolicyViolation(f"disallowed change on {change.tensor} op={change.op}")

        if change.op in ("set", "add", "scale") and change.value is not None:
            if abs(change.value) > policy.max_abs_delta:
                raise PolicyViolation(f"delta {change.value} exceeds max_abs_delta")

        if change.op == "low_rank" and (change.rank or 0) > policy.max_rank:
            raise PolicyViolation(f"rank {change.rank} exceeds max_rank")

        # Rough entry count; real implementation will count coordinates
        changed_estimate += 1

    fraction = changed_estimate / max(base_param_count, 1)
    if fraction > policy.max_fraction:
        raise PolicyViolation(f"fraction {fraction:.6f} exceeds max_fraction")
