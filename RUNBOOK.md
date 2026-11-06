# RUNBOOK

Exact commands to reproduce every number in the README. Times were measured on a MacBook with an Apple M4 (16 GB
unified memory) unless marked "estimate".

## 0. Setup

```bash
uv venv --python 3.12 .venv
uv sync --all-extras                 # agent from GitHub at commit ad9ed18 + `train` extra (mlx-lm, Apple Silicon)
source .venv/bin/activate
pytest                               # 39 fast tests, ~2 s: no dataset, no Ollama, no model download
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

## 2. Teacher traces, filtering, training, evaluation

Start Ollama **0.18.2** with `ollama serve` (the menu bar app can auto-update; 0.40.15 no longer merges consecutive
user messages, which changes the retry prompt, see `tests/test_ollama_template.py`). Times marked "not measured yet" are filled in once run.

| Step | Command | Time (M4) | Output |
|---|---|---|---|
| Teacher traces (3 batches of 100, resumable) | `python scripts/run_teacher.py --batch 1` (then 2, 3) | 35 min 48 s, 34 min 28 s, 38 min 28 s | `data/traces/teacher.jsonl`, `results/traces/teacher_batch_<n>.json` |
| Rejection sampling + training examples | `python scripts/filter_traces.py` | 2 s | `data/sft/{train,valid}.jsonl`, `results/filter/report.json` |
| Student weights (Hugging Face, 2.9 GB) | see below | 1 min 50 s | `models/qwen2.5-coder-1.5b-instruct/` |
| Base student into Ollama (Q4_K_M) | `python scripts/import_ollama.py models/qwen2.5-coder-1.5b-instruct student-base` | 1 min 01 s | Ollama model `student-base` |
| Timing run (20 steps) | `python scripts/train.py --timing` | 5 min 18 s | `results/train/timing.json` |
| LoRA training | `python scripts/train.py` | 51 min (3,058.5 s) | `results/train/lora.json`, `loss.png`, `adapter/` |
| Fuse adapter into the weights | `mlx_lm.fuse --model models/qwen2.5-coder-1.5b-instruct --adapter-path models/adapters/distilled --save-path models/fused-distilled` | 6 s | `models/fused-distilled/` |
| Distilled student into Ollama | `python scripts/import_ollama.py models/fused-distilled student-distilled` | not measured yet | Ollama model `student-distilled` |
| Evaluation, 50 questions, per model | `python scripts/evaluate.py --config configs/<teacher, student_base, student_distilled>.yaml` | 18 min, 12 min 35 s, 6 min 34 s | `results/eval/<name>.json` + `_per_question.jsonl` |
| Tables, held-out split, error analysis | `python scripts/analyze_eval.py` | ~1 s | `results/tables.md`, `results/error_analysis.md`, `results/eval/breakdown.json` |
| Ollama template check (slow test) | `pytest -m slow tests/test_ollama_template.py` | 5 s | - |

Student weights:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-1.5B-Instruct', revision='2e1fd397ee46e1388853d2af2c993145b0f1098a', local_dir='models/qwen2.5-coder-1.5b-instruct')"
```

Disk: `ollama create` from safetensors keeps a raw copy (3.1 GB) and an f16 copy (3.1 GB) of the weights in
`~/.ollama/models/blobs` next to the 1 GB Q4_K_M model. Plan ~7 GB free per import.
