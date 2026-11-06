"""Rejection sampling: keep only the teacher traces whose final result matches the pandas reference answer,
then turn each kept trace into chat-format training examples (one per LLM call).

python scripts/filter_traces.py      # ~20 s (tokenizes every example once)
-> data/sft/train.jsonl, data/sft/valid.jsonl, results/filter/report.json
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from distill.runinfo import (
    DATA,
    REPO_ROOT,
    RESULTS,
    read_jsonl,
    run_metadata,
    sha256_file,
    write_json,
    write_jsonl,
)
from distill.sft import examples_from_trace, filter_traces

STUDENT = REPO_ROOT / "models/qwen2.5-coder-1.5b-instruct"
MAX_SEQ_LENGTH = 2048  # mlx_lm.lora default; longer examples would be truncated


def token_check(tok, examples: list[dict]) -> dict:
    """Token lengths, and whether the prompt's tokens are an exact prefix of the full example's tokens
    (what --mask-prompt assumes when it hides the prompt from the loss)."""
    lengths, not_prefix = [], 0
    for ex in examples:
        full = tok.apply_chat_template(ex["messages"], return_dict=False)
        prompt = tok.apply_chat_template(ex["messages"][:-1], add_generation_prompt=True, return_dict=False)
        lengths.append(len(full))
        not_prefix += full[: len(prompt)] != prompt
    return {
        "tokens_max": int(max(lengths)),
        "tokens_p50": float(np.percentile(lengths, 50)),
        "tokens_p95": float(np.percentile(lengths, 95)),
        "n_longer_than_max_seq_length": sum(n > MAX_SEQ_LENGTH for n in lengths),
        "n_prompt_not_a_token_prefix": not_prefix,
        "total_tokens": int(sum(lengths)),
    }


def main() -> None:
    traces = read_jsonl(DATA / "traces/teacher.jsonl")
    split = {q["id"]: q["split"] for q in read_jsonl(DATA / "train/questions.jsonl")}
    kept, reasons = filter_traces(traces)
    examples = {"train": [], "valid": []}
    for t in kept:
        examples[split[t["id"]]] += examples_from_trace(t)
    for name, rows in examples.items():
        write_jsonl(DATA / f"sft/{name}.jsonl", rows)

    tok = AutoTokenizer.from_pretrained(STUDENT)
    by_template = Counter(t["template"] for t in traces)
    kept_template = Counter(t["template"] for t in kept)
    report = {
        **run_metadata(),
        "traces_sha256": sha256_file(DATA / "traces/teacher.jsonl"),
        "n_traces": len(traces),
        "n_kept": len(kept),
        "kept_rate": len(kept) / len(traces),
        "rejected_by_reason": dict(reasons),
        "kept_by_category": dict(Counter(t["category"] for t in kept)),
        "kept_by_template": {k: f"{kept_template[k]}/{n}" for k, n in sorted(by_template.items())},
        "kept_traces_fixed_by_a_retry": sum(t["n_attempts"] > 1 for t in kept),
        "n_examples": {name: len(rows) for name, rows in examples.items()},
        "examples_by_node": {name: dict(Counter(e["node"] for e in rows)) for name, rows in examples.items()},
        "tokens": {name: token_check(tok, rows) for name, rows in examples.items() if rows},
    }
    write_json(RESULTS / "filter/report.json", report)
    print({k: report[k] for k in ("n_traces", "n_kept", "rejected_by_reason", "n_examples", "tokens")})


if __name__ == "__main__":
    main()
