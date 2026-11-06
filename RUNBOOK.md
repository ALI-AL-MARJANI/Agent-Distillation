# RUNBOOK

Exact commands to reproduce every number in the README. Times were measured on a MacBook with an Apple M4 (16 GB
unified memory) unless marked "estimate".

## 0. Setup

```bash
uv venv --python 3.12 .venv
uv sync --all-extras                 # pinned by uv.lock; installs the agent from GitHub at commit ad9ed18
source .venv/bin/activate
pytest                               # 27 fast tests, <1 s: no dataset, no Ollama, no model download
```

The agent (`analyst` package: LangGraph graph, sandbox, scoring) is installed unchanged from
[Data-Analyst-Agent](https://github.com/ALI-AL-MARJANI/Data-Analyst-Agent) at commit `ad9ed18`.
`tests/test_split.py` checks that this exact commit is the one installed.

## 1. Data and questions

| Step | Command | Time (M4) | Output |
|---|---|---|---|
| CSV (same file and sha256 as the agent repo) | `python scripts/get_data.py --zip ../Data-Analyst-Agent/data/raw/online_retail_ii.zip` | 27 s | `data/raw/online_retail_ii.csv`, `data/manifest.json` |
| Evaluation answers | `python scripts/build_eval_truth.py` | 2 s | `data/eval/truth.json` |
| Training questions + leakage report | `python scripts/make_train_questions.py` | 7 s | `data/train/questions.jsonl`, `results/split/overlap.json` |

Without a local zip, `python scripts/get_data.py` downloads it from UCI (it took ~15 min, the server is slow).

- **Evaluation set (50 questions, never trained on):** the agent's 20 dev questions and 10 held-out questions,
  copied unchanged, plus 20 new held-out questions (h11-h30) written on 2026-11-06 before any trace or training.
  Their sha256 is pinned in `tests/test_split.py`, so editing one fails the tests.
- **Training set (300 questions = 270 train + 30 validation):** 24 templates x parameters (`src/distill/templates.py`),
  seed 0, answers computed by pandas. Regenerating gives the same file byte for byte.
- **Leakage check:** no training question equals an eval question after normalization, and none shares 60 % or
  more of its words with one (Jaccard). Eval parameters are also excluded from the matching templates
  (e.g. no "revenue for the Netherlands", which is q07).
