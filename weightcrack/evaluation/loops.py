"""Practice/holdout evaluation loop (scaffold).

Sector targets must improve; sentinel controls must not regress.
"""

from typing import Any
from pathlib import Path
import json


def evaluate_candidate(
    base_path: Path,
    candidate_path: Path,
    suite: dict[str, Any],
    min_target_gain: int = 1,
    max_sentinel_flips: int = 0,
) -> dict[str, Any]:
    """
    Compare base vs candidate on a frozen suite.
    Real implementation loads both models and runs the probe set.
    """
    report = {
        "schema": "weightcrack.eval.v1",
        "base": str(base_path),
        "candidate": str(candidate_path),
        "sector": suite.get("sector"),
        "target_before": None,
        "target_after": None,
        "sentinel_flips": None,
        "passed": False,
        "note": "evaluation runner pending model load integration",
    }
    report_path = candidate_path.with_suffix(".eval.json")
    report_path.write_text(json.dumps(report, indent=2))
    return report
