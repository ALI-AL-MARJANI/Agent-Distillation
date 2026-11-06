"""The numbers of the results table, computed from per-question rows (one row = one agent run)."""

from __future__ import annotations

from collections import Counter

import numpy as np


def summarize(rows: list[dict]) -> dict:
    """Accuracy, retries and latency. Retries are counted only on questions where code ran
    (a refusal runs no code and has no retry to count)."""
    ran = [r for r in rows if r["n_attempts"] > 0]
    latencies = [r["latency_s"] for r in rows]
    return {
        "n": len(rows),
        "n_correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "failures": dict(Counter(r["failure"] for r in rows if r["failure"])),
        "questions_that_ran_code": len(ran),
        "avg_retries": float(np.mean([r["n_attempts"] - 1 for r in ran])) if ran else None,
        "first_run_ok": sum(r["attempts"][0]["ok"] for r in ran),
        "fixed_by_retry": sum(not r["attempts"][0]["ok"] and r["attempts"][-1]["ok"] for r in ran),
        "latency_s_p50": float(np.percentile(latencies, 50)),
        "latency_s_p90": float(np.percentile(latencies, 90)),
    }


def summarize_splits(rows: list[dict]) -> dict:
    """All questions, then dev and held-out on their own lines."""
    out = {"all": summarize(rows)}
    for split in ("dev", "heldout"):
        if part := [r for r in rows if r["split"] == split]:
            out[split] = summarize(part)
    return out


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p-value for the 2x2 table [[a, b], [c, d]]
    (rows: two question groups, columns: correct / wrong). Small samples, so no normal approximation."""
    from math import comb

    n, r1, c1 = a + b + c + d, a + b, a + c

    def p(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9))
