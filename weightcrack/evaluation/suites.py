"""Minimal sector probe suite (math loop reduction).

Practice, holdout, and sentinel examples are hash-pinned.
Scoring is a stub that can be replaced with real model generation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _hash_examples(examples: list[dict[str, Any]]) -> str:
    payload = json.dumps(examples, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


# Minimal pinned suite for math loop reduction.
# Real prompts would be longer; these are structural placeholders.
MATH_LOOP_SUITE = {
    "schema": "weightcrack.suite.v1",
    "sector": "math_loop",
    "practice": [
        {"id": "p1", "prompt": "Solve: 2 + 2 =", "expected": "4"},
        {"id": "p2", "prompt": "Solve: 3 * 3 =", "expected": "9"},
    ],
    "holdout": [
        {"id": "h1", "prompt": "Solve: 5 + 5 =", "expected": "10"},
        {"id": "h2", "prompt": "Solve: 4 * 4 =", "expected": "16"},
    ],
    "sentinels": [
        {"id": "s1", "prompt": "The capital of France is", "expected": "Paris"},
        {"id": "s2", "prompt": "2 + 2 is not", "expected": "5"},
    ],
}


# Minimal pinned suite for coding precision (second sector).
CODING_PRECISION_SUITE = {
    "schema": "weightcrack.suite.v1",
    "sector": "coding_precision",
    "practice": [
        {"id": "cp1", "prompt": "def add(a, b): return", "expected": "a + b"},
        {"id": "cp2", "prompt": "def mul(a, b): return", "expected": "a * b"},
    ],
    "holdout": [
        {"id": "ch1", "prompt": "def sub(a, b): return", "expected": "a - b"},
        {"id": "ch2", "prompt": "def div(a, b): return", "expected": "a / b"},
    ],
    "sentinels": [
        {"id": "cs1", "prompt": "The capital of France is", "expected": "Paris"},
        {"id": "cs2", "prompt": "2 + 2 is not", "expected": "5"},
    ],
}


def suite_hashes(suite: dict[str, Any]) -> dict[str, str]:
    return {
        "practice": _hash_examples(suite["practice"]),
        "holdout": _hash_examples(suite["holdout"]),
        "sentinels": _hash_examples(suite["sentinels"]),
    }


def all_suite_hashes() -> dict[str, dict[str, str]]:
    return {
        "math_loop": suite_hashes(MATH_LOOP_SUITE),
        "coding_precision": suite_hashes(CODING_PRECISION_SUITE),
    }


def score_stub(outputs: list[str], expected: list[str]) -> int:
    """Count exact matches. Real runner will use model generation."""
    return sum(1 for o, e in zip(outputs, expected) if o.strip() == e.strip())
