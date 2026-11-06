"""Phase 3 verification: suite hashes are stable and scoring works."""

from weightcrack.evaluation.suites import MATH_LOOP_SUITE, suite_hashes, score_stub


def test_suite_pins_and_scoring():
    hashes = suite_hashes()
    assert len(hashes["practice"]) == 64
    assert len(hashes["holdout"]) == 64
    assert len(hashes["sentinels"]) == 64
    # Recompute to confirm stability
    assert suite_hashes() == hashes

    practice_expected = [ex["expected"] for ex in MATH_LOOP_SUITE["practice"]]
    perfect = score_stub(practice_expected, practice_expected)
    assert perfect == 2
    partial = score_stub(["4", "wrong"], practice_expected)
    assert partial == 1
    print("Phase 3 suite pins and scoring: ok")
    return True


if __name__ == "__main__":
    test_suite_pins_and_scoring()
