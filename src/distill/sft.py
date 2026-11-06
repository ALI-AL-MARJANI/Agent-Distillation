"""Rejection sampling (keep only correct teacher traces) and conversion to chat-format training examples.

One training example = one LLM call of a kept trace: the exact messages the teacher saw, plus its raw output as
the assistant message. MLX-LM reads {"messages": [...]} lines and ignores the extra keys (id, node).
"""

from __future__ import annotations

from collections import Counter

from distill.chat_format import collate


def reject_reason(trace: dict) -> str | None:
    """None if the trace is kept, else why it is rejected (the agent's own failure categories)."""
    if trace.get("crash"):
        return "crash"
    if not trace["correct"]:
        return trace["failure"]
    if any(c["output"] is None for c in trace["calls"]):
        return "missing_llm_output"
    return None


def examples_from_trace(trace: dict) -> list[dict]:
    """Training examples from a kept trace. write_code calls map one-to-one, in order, to run attempts;
    a write_code output whose code failed is skipped (it is a bug, not something to imitate)."""
    attempts = iter(trace["attempts"])
    examples = []
    for call in trace["calls"]:
        if call["node"] == "write_code" and not next(attempts)["ok"]:
            continue
        examples.append(
            {
                "id": trace["id"],
                "node": call["node"],
                # same merge as Ollama at serving time (retry = two user messages in a row)
                "messages": collate([*call["messages"], {"role": "assistant", "content": call["output"]}]),
            }
        )
    return examples


def filter_traces(traces: list[dict]) -> tuple[list[dict], Counter]:
    kept, reasons = [], Counter()
    for t in traces:
        if (reason := reject_reason(t)) is None:
            kept.append(t)
        else:
            reasons[reason] += 1
    return kept, reasons
