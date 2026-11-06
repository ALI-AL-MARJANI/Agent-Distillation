"""Compute the reference answer of every evaluation question with plain pandas.

python scripts/build_eval_truth.py       # ~5 s
-> data/eval/truth.json
"""

from __future__ import annotations

from importlib.metadata import version

from distill.reference import EVAL_REFERENCE, to_plain
from distill.runinfo import DATA, load_yaml, sha256_file, write_json
from distill.templates import load_csv


def main() -> None:
    df = load_csv()
    questions = load_yaml(DATA / "eval/questions.yaml")
    assert {q["id"] for q in questions} == set(EVAL_REFERENCE), "every question needs one reference function"

    truth = {}
    for q in questions:
        value = to_plain(EVAL_REFERENCE[q["id"]](df))
        truth[q["id"]] = {"kind": q["kind"], "value": value}
        print(f"{q['id']} {q['kind']:>12}  {value}")
    write_json(
        DATA / "eval/truth.json",
        {
            "csv_sha256": sha256_file(DATA / "raw/online_retail_ii.csv"),
            "pandas": version("pandas"),
            "truth": truth,
        },
    )


if __name__ == "__main__":
    main()
