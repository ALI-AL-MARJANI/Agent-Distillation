"""Trace filtering (rejection sampling), conversion to training examples, and the trace recorder."""

from types import SimpleNamespace
from uuid import uuid4

from distill.sft import examples_from_trace, filter_traces, reject_reason
from distill.traces import Recorder

SYS = {"role": "system", "content": "sys"}


def call(node, user, output):
    return {
        "node": node,
        "messages": [SYS, {"role": "user", "content": user}],
        "output": output,
        "error": None,
    }


def trace(correct=True, failure=None, attempts=(True,), crash=None, calls=None):
    calls = calls or [
        call("understand", "q", '{"restated": "q", "answerable": true, "reason": "r"}'),
        call("write_code", "q", '{"plan": "p", "code": "result = 1"}'),
        call("answer", "result 1", "It is 1."),
    ]
    return {
        "id": "t001",
        "correct": correct,
        "failure": failure,
        "crash": crash,
        "attempts": [{"ok": ok} for ok in attempts],
        "n_attempts": len(attempts),
        "calls": calls,
    }


def test_only_correct_traces_are_kept_with_reasons():
    traces = [trace(), trace(False, "wrong_value"), trace(False, "crash", crash="ConnectionError")]
    kept, reasons = filter_traces(traces)
    assert len(kept) == 1 and reasons == {"wrong_value": 1, "crash": 1}


def test_correct_trace_with_a_missing_output_is_rejected():
    t = trace()
    t["calls"][1]["output"] = None
    assert reject_reason(t) == "missing_llm_output"


def test_one_example_per_call_in_chat_format():
    examples = examples_from_trace(trace())
    assert [e["node"] for e in examples] == ["understand", "write_code", "answer"]
    assert examples[1]["messages"][-1] == {
        "role": "assistant",
        "content": '{"plan": "p", "code": "result = 1"}',
    }
    assert examples[1]["messages"][:-1] == [SYS, {"role": "user", "content": "q"}]


def test_failed_code_is_not_a_training_target_but_its_fix_is():
    calls = [
        call("understand", "q", "{}"),
        call("write_code", "q", '{"code": "result = df[\'Revenue\'].sum()"}'),
        call("write_code", "q + error KeyError", '{"code": "result = 1"}'),
        call("answer", "result 1", "It is 1."),
    ]
    examples = examples_from_trace(trace(attempts=(False, True), calls=calls))
    targets = [e["messages"][-1]["content"] for e in examples]
    assert "Revenue" not in "".join(targets) and '{"code": "result = 1"}' in targets
    assert [e["node"] for e in examples] == ["understand", "write_code", "answer"]


def test_correct_refusal_gives_a_single_understand_example():
    refusal = [
        call("understand", "profit?", '{"restated": "profit", "answerable": false, "reason": "no cost"}')
    ]
    examples = examples_from_trace(trace(attempts=(), calls=refusal))
    assert [e["node"] for e in examples] == ["understand"]


def test_recorder_captures_node_messages_and_raw_output():
    rec, run = Recorder(), uuid4()
    msgs = [[SimpleNamespace(type="system", content="sys"), SimpleNamespace(type="human", content="hi")]]
    rec.on_chat_model_start({}, msgs, run_id=run, metadata={"langgraph_node": "understand"})
    rec.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(text='{"a": 1}')]]), run_id=run)
    assert rec.calls == [
        {
            "node": "understand",
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            "output": '{"a": 1}',
            "error": None,
        }
    ]


def test_recorder_keeps_errors():
    rec, run = Recorder(), uuid4()
    rec.on_chat_model_start({}, [[SimpleNamespace(type="human", content="hi")]], run_id=run, metadata={})
    rec.on_llm_error(ConnectionError("refused"), run_id=run)
    assert rec.calls[0]["output"] is None and rec.calls[0]["error"] == "ConnectionError: refused"


def test_retry_example_is_collated_like_ollama():
    calls = [
        call("understand", "q", "{}"),
        call("write_code", "q", "{bad}"),
        {
            "node": "write_code",
            "messages": [SYS, {"role": "user", "content": "q"}, {"role": "user", "content": "fix"}],
            "output": "{good}",
            "error": None,
        },
        call("answer", "r", "ok"),
    ]
    retry = examples_from_trace(trace(attempts=(False, True), calls=calls))[1]
    assert retry["messages"][1:] == [
        {"role": "user", "content": "q\n\nfix"},
        {"role": "assistant", "content": "{good}"},
    ]
