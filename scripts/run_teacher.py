"""Teacher traces: run the agent (7B teacher) on the training questions and save every step.

python scripts/run_teacher.py --batch 1      # questions t001-t100, ~35 min (estimate)
python scripts/run_teacher.py --batch 2      # t101-t200
python scripts/run_teacher.py --batch 3      # t201-t300
python scripts/run_teacher.py --ids t001 t002

Resumable: each trace is appended to data/traces/teacher.jsonl as soon as it finishes, and questions already
in the file are skipped. A summary with run metadata goes to results/traces/teacher_batch_<n>.json.
"""

from __future__ import annotations

import argparse
import json
import time

from analyst.graph import build_agent
from analyst.llm import make_chat

from distill.runinfo import DATA, REPO_ROOT, RESULTS, load_config, read_jsonl, run_metadata, write_json
from distill.traces import run_one

BATCH_SIZE = 100
TRACES = DATA / "traces/teacher.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/teacher.yaml")
    ap.add_argument("--batch", type=int, choices=[1, 2, 3])
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    questions = read_jsonl(DATA / "train/questions.jsonl")
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    elif args.batch:
        questions = questions[(args.batch - 1) * BATCH_SIZE : args.batch * BATCH_SIZE]
    done = {t["id"] for t in read_jsonl(TRACES)} if TRACES.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(questions)} questions, {len(questions) - len(todo)} already done, {len(todo)} to run")

    chat = make_chat(cfg["llm"])
    chat.invoke("Say OK.")  # load the model so it does not count in the first latency
    graph = build_agent(cfg, str(REPO_ROOT / cfg["data"]["csv"]), chat=chat)
    TRACES.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for q in todo:
        trace = run_one(graph, q, q["truth"])
        with open(TRACES, "a") as f:
            f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
        print(
            f"{q['id']} {'OK ' if trace['correct'] else 'BAD'} attempts={trace['n_attempts']} "
            f"calls={len(trace['calls'])} {trace['latency_s']:.1f}s {trace['failure'] or ''}",
            flush=True,
        )

    if args.batch:
        ids = {q["id"] for q in questions}
        rows = [t for t in read_jsonl(TRACES) if t["id"] in ids]
        write_json(
            RESULTS / f"traces/teacher_batch_{args.batch}.json",
            {
                **run_metadata(config=cfg),
                "batch": args.batch,
                "n": len(rows),
                "n_correct": sum(r["correct"] for r in rows),
                "n_crash": sum(r["crash"] is not None for r in rows),
                "wall_clock_s_this_run": round(time.perf_counter() - t0, 1),
                "sum_latency_s": round(sum(r["latency_s"] for r in rows), 1),
            },
        )


if __name__ == "__main__":
    main()
