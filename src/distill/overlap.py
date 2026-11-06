"""Leakage check between training and evaluation questions.

Two rules, both must hold:
1. no training question equals an evaluation question after normalization (case, punctuation, spaces);
2. no training question is a near-copy: word-set Jaccard similarity below MAX_JACCARD with every eval question.
Jaccard = |words in both| / |words in either|, from 0 (no shared word) to 1 (same words).
"""

from __future__ import annotations

import re

MAX_JACCARD = 0.6


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def words(text: str) -> set[str]:
    return set(normalize(text).split())


def jaccard(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 1.0


def check(train: list[str], evaluation: list[str], max_jaccard: float = MAX_JACCARD) -> dict:
    """Return exact matches, near-copies above the threshold, and the closest training question per eval one."""
    eval_norm = {normalize(e) for e in evaluation}
    exact = sorted({t for t in train if normalize(t) in eval_norm})
    closest = []
    for e in evaluation:
        score, t = max(((jaccard(t, e), t) for t in train), default=(0.0, ""))
        closest.append({"eval": e, "closest_train": t, "jaccard": round(score, 3)})
    closest.sort(key=lambda r: -r["jaccard"])
    near = [r for r in closest if r["jaccard"] >= max_jaccard]
    return {"max_jaccard_allowed": max_jaccard, "exact": exact, "near": near, "closest": closest}
