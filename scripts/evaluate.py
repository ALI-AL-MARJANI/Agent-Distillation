"""Evaluate one model on the 50 frozen questions: same agent, same sandbox, same config; only the model changes.

python scripts/evaluate.py --config configs/teacher.yaml             # ~17 min (estimate)
python scripts/evaluate.py --config configs/student_base.yaml
python scripts/evaluate.py --config configs/student_distilled.yaml

-> results/eval/<name>.json (summary + metadata), results/eval/<name>_per_question.jsonl
Memory = what Ollama reports for the loaded model (/api/ps: size, and size_vram = the part in GPU memory).
"""

from __future__ import annotations

import argparse
import json

from analyst.graph import build_agent
from analyst.llm import make_chat

from distill.metrics import summarize_splits
from distill.ollama_api import model_info, unload_all
from distill.runinfo import (
    DATA,
    REPO_ROOT,
    RESULTS,
    load_config,
    load_yaml,
    run_metadata,
    sha256_file,
    write_json,
    write_jsonl,
)
from distill.traces import run_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", nargs="*", help="only these questions (quick check, not a reported result)")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    questions = load_yaml(DATA / "eval/questions.yaml")
    truth = load_yaml(DATA / "eval/truth.json")["truth"]
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]

    unload_all()
    chat = make_chat(cfg["llm"])
    chat.invoke("Say OK.")  # load the model so it does not count in the first latency
    graph = build_agent(cfg, str(REPO_ROOT / cfg["data"]["csv"]), chat=chat)
    rows = []
    for q in questions:
        row = run_one(graph, q, truth[q["id"]])
        rows.append(row)
        print(
            f"{q['id']} {'OK ' if row['correct'] else 'BAD'} attempts={row['n_attempts']} "
            f"{row['latency_s']:.1f}s {row['failure'] or ''}",
            flush=True,
        )
    info = model_info(cfg["llm"]["model"])  # read while the model is still loaded

    name = cfg["name"] + ("_subset" if args.ids else "")
    write_jsonl(RESULTS / f"eval/{name}_per_question.jsonl", rows)
    summary = summarize_splits(rows)
    write_json(
        RESULTS / f"eval/{name}.json",
        {
            **run_metadata(config=cfg),
            "ollama": info,
            "eval_questions_sha256": sha256_file(DATA / "eval/questions.yaml"),
            "eval_truth_sha256": sha256_file(DATA / "eval/truth.json"),
            "question_ids": [q["id"] for q in questions],
            "summary": summary,
        },
    )
    print(json.dumps({"ollama": info, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
