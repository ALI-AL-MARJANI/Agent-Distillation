"""Run the agent on one question and record every LLM call (exact input messages and raw output).

The agent's code is not touched: a LangChain callback handler is passed in the graph's config, and LangGraph
propagates it to every model call made inside the nodes. `metadata["langgraph_node"]` tells which node called.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from analyst.scoring import score
from langchain_core.callbacks import BaseCallbackHandler

ROLES = {"system": "system", "human": "user", "ai": "assistant"}


class Recorder(BaseCallbackHandler):
    """Collects one entry per chat-model call: node name, messages in, raw text out (or the error)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._open: dict[UUID, dict] = {}

    def on_chat_model_start(
        self, serialized: dict, messages: list[list[Any]], *, run_id: UUID, metadata: dict | None = None, **kw
    ) -> None:
        call = {
            "node": (metadata or {}).get("langgraph_node"),
            "messages": [{"role": ROLES.get(m.type, m.type), "content": str(m.content)} for m in messages[0]],
            "output": None,
            "error": None,
        }
        self._open[run_id] = call
        self.calls.append(call)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kw) -> None:
        if call := self._open.pop(run_id, None):
            call["output"] = response.generations[0][0].text

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw) -> None:
        if call := self._open.pop(run_id, None):
            call["error"] = f"{type(error).__name__}: {error}"[:500]


def run_one(graph: Any, q: dict, truth: dict) -> dict:
    """Run the agent on question `q`, score it against `truth`, return the full trace."""
    recorder = Recorder()
    t0 = time.perf_counter()
    try:
        state = graph.invoke({"question": q["question"]}, config={"callbacks": [recorder]})
        crash = None
    except Exception as e:  # e.g. Ollama unreachable: recorded, not hidden
        state, crash = {}, f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0
    correct, category = (False, "crash") if crash else score(state, truth)
    return {
        "id": q["id"],
        "split": q.get("split", "dev"),
        "category": q["category"],
        "template": q.get("template"),
        "question": q["question"],
        "truth": truth["value"],
        "kind": truth["kind"],
        "correct": correct,
        "failure": None if correct else category,
        "answerable": state.get("answerable"),
        "restated": state.get("restated"),
        "result": state.get("result"),
        "answer": state.get("answer"),
        "attempts": state.get("attempts", []),
        "n_attempts": len(state.get("attempts", [])),
        "calls": recorder.calls,
        "latency_s": round(latency, 2),
        "crash": crash,
    }
