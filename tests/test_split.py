"""No leakage between training and evaluation questions, and the evaluation set is frozen."""

import hashlib

import pytest

from distill.overlap import check, jaccard, normalize
from distill.runinfo import DATA, REPO_ROOT, agent_commit, load_config, load_yaml, read_jsonl

# Frozen on 2026-11-06, before any teacher trace or training run. Changing an eval question fails this test.
EVAL_QUESTIONS_SHA256 = "df4da862847999f90ba70f2a5fde2264dfef9076e1aab2bc3373dfa073d9e7a4"
EVAL_TRUTH_SHA256 = "5d7e214d4233ef664b2b1c93d28039e2178c9d0b4ca5a013e60b560cf2b76fe0"

EVAL = load_yaml(DATA / "eval/questions.yaml")
TRUTH = load_yaml(DATA / "eval/truth.json")["truth"]  # JSON is valid YAML
TRAIN = read_jsonl(DATA / "train/questions.jsonl")
CFG = load_config(REPO_ROOT / "configs/data.yaml")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_eval_set_is_frozen():
    assert sha256(DATA / "eval/questions.yaml") == EVAL_QUESTIONS_SHA256
    assert sha256(DATA / "eval/truth.json") == EVAL_TRUTH_SHA256


def test_eval_set_shape():
    ids = [q["id"] for q in EVAL]
    assert len(ids) == len(set(ids)) == 50 and set(ids) == set(TRUTH)
    assert sum(q.get("split", "dev") == "heldout" for q in EVAL) == 30


def test_no_training_question_copies_an_eval_question():
    report = check([q["question"] for q in TRAIN], [q["question"] for q in EVAL], CFG["max_jaccard"])
    assert report["exact"] == [] and report["near"] == []


def test_ids_and_splits_are_disjoint():
    assert not {q["id"] for q in TRAIN} & {q["id"] for q in EVAL}
    assert len({q["question"] for q in TRAIN}) == len(TRAIN) == CFG["n_questions"]
    assert sum(q["split"] == "valid" for q in TRAIN) == CFG["n_valid"]


@pytest.mark.parametrize(
    ("template", "param", "eval_value"),
    [
        ("revenue_country", "c", "Netherlands"),  # q07
        ("revenue_country", "c", "Australia"),  # h14
        ("customers_country", "c", "Spain"),  # h09
        ("rows_country", "c", "Germany"),  # h16
        ("revenue_change_country", "c", "Germany"),  # q15
        ("invoice_value", "inv", "489434"),  # q17
        ("rows_quantity_above", "x", 100),  # h17
        ("revenue_price_above", "x", 100),  # q11
        ("invoices_weekday_year", "d", "Sunday"),  # h23
    ],
)
def test_eval_parameters_are_excluded_from_templates(template, param, eval_value):
    assert not [q for q in TRAIN if q["template"] == template and q["params"][param] == eval_value]


def test_every_training_question_has_a_reference_answer():
    for q in TRAIN:
        assert (q["truth"]["value"] is None) == (q["kind"] == "unanswerable"), q["id"]


def test_check_catches_copies_and_near_copies():
    evaluation = ["How many rows have a negative Quantity?"]
    report = check(
        ["how many ROWS have a negative quantity", "Count rows whose Quantity exceeds 5."], evaluation
    )
    assert report["exact"] == ["how many ROWS have a negative quantity"]
    near = check(["How many rows have a negative Quantity value?"], evaluation)["near"]
    assert len(near) == 1 and near[0]["jaccard"] == pytest.approx(7 / 8, abs=1e-3)  # 7 shared words out of 8
    assert check(["Average of Price for Japan rows?"], evaluation)["near"] == []


def test_normalize_and_jaccard():
    assert normalize("  What's the  TOTAL? ") == "what s the total"
    assert jaccard("a b c", "a b d") == pytest.approx(0.5)


def test_the_agent_is_the_pinned_commit():
    assert agent_commit() == "ad9ed1871af245b50c966e0b6f12c79005e71e60"
