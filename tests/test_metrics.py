import pytest

from distill.metrics import summarize, summarize_splits


def row(correct, attempts, latency, split="dev", failure=None):
    return {
        "correct": correct,
        "attempts": [{"ok": ok} for ok in attempts],
        "n_attempts": len(attempts),
        "latency_s": latency,
        "split": split,
        "failure": failure,
    }


ROWS = [
    row(True, [True], 10.0),
    row(True, [False, True], 30.0, "heldout"),  # fixed by one retry
    row(False, [False, False, False, False], 50.0, "heldout", "error_unfixed"),
    row(True, [], 2.0, "heldout"),  # correct refusal: no code, not counted in retries
]


def test_summary():
    s = summarize(ROWS)
    assert s["n_correct"] == 3 and s["accuracy"] == pytest.approx(0.75)
    assert s["questions_that_ran_code"] == 3 and s["avg_retries"] == pytest.approx((0 + 1 + 3) / 3)
    assert s["first_run_ok"] == 1 and s["fixed_by_retry"] == 1
    assert s["latency_s_p50"] == pytest.approx(20.0) and s["failures"] == {"error_unfixed": 1}


def test_splits_are_reported_separately():
    s = summarize_splits(ROWS)
    assert s["all"]["n"] == 4 and s["dev"]["n"] == 1 and s["heldout"]["n"] == 3
    assert s["heldout"]["n_correct"] == 2


def test_fisher_exact_matches_known_values():
    from distill.metrics import fisher_exact

    assert fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)  # classic tea-tasting table
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
