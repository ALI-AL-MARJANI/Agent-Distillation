"""Generate the training questions from templates, with their reference answers, and check for leakage.

python scripts/make_train_questions.py     # ~10 s
-> data/train/questions.jsonl, results/split/overlap.json
"""

from __future__ import annotations

from collections import Counter

from distill.overlap import check
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
from distill.templates import assign_splits, generate, load_csv


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs/data.yaml")
    questions, skipped = generate(load_csv(), cfg["n_questions"], cfg["seed"])
    questions = assign_splits(questions, cfg["n_valid"], cfg["seed"])
    write_jsonl(DATA / "train/questions.jsonl", questions)

    evaluation = [q["question"] for q in load_yaml(DATA / "eval/questions.yaml")]
    report = check([q["question"] for q in questions], evaluation, cfg["max_jaccard"])
    write_json(
        RESULTS / "split/overlap.json",
        {
            **run_metadata(config=cfg),
            "eval_questions_sha256": sha256_file(DATA / "eval/questions.yaml"),
            "train_questions_sha256": sha256_file(DATA / "train/questions.jsonl"),
            "n_train": sum(q["split"] == "train" for q in questions),
            "n_valid": sum(q["split"] == "valid" for q in questions),
            "per_template": dict(Counter(q["template"] for q in questions)),
            "per_category": dict(Counter(q["category"] for q in questions)),
            "skipped_candidates": skipped,  # ties or NaN answers
            **report,
        },
    )
    print(f"{len(questions)} questions, per category {dict(Counter(q['category'] for q in questions))}")
    print(f"skipped: {skipped}")
    print(f"exact matches: {len(report['exact'])}, near-copies: {len(report['near'])}")
    for r in report["closest"][:5]:
        print(f"  {r['jaccard']:.2f}  eval: {r['eval']}\n        train: {r['closest_train']}")


if __name__ == "__main__":
    main()
