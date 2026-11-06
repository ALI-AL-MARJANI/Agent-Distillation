# Devlog

What happened while building this project, including what broke and how it was fixed. Numbers come from the
files named next to them.

## 2026-11-06

**Setup.** New repo; the agent is installed unchanged from Data-Analyst-Agent at commit `ad9ed18`, and a test checks
that this exact commit is installed. The evaluation set grew from 30 to 50 questions: 20 new held-out questions
(h11-h30) were written before any trace or training, and their hash is pinned in a test.

**Leakage check caught a real near-copy.** A training template produced "How many unique invoices were created on a
Sunday in 2011?", 67 % word overlap with the new held-out h23. Sunday was removed from that template; the final
training set has 0 exact and 0 near-copies (`results/split/overlap.json`).

**Copied script wrote into the venv.** `get_data.py`, copied from the agent repo, imported the agent's `REPO_ROOT`.
Once the agent is an installed package, that path points inside `.venv`, so the CSV landed there. Fixed by using
this repo's own paths everywhere.

**Teacher traces.** 300 questions in 3 batches (35 min 48 s, 34 min 28 s, 38 min 28 s), 0 crashes, 227 correct
(`results/traces/`, `results/filter/report.json`). Of the 73 rejected traces, 34 are false refusals: the teacher
only sees 5 sample rows and decides that "the data does not contain any rows with 'Brazil'". Worst template:
revenue by country, 2 of 13 kept.

**Training and serving prompts differed on retries.** The real Ollama server saw 202 prompt tokens on the retry
call, the Hugging Face template produced 207. Ollama's source (`template/template.go`, `collate`) merges
consecutive messages with the same role, joined by a blank line, and the agent's retry sends two user messages in a
row. Training examples now get the same merge, and a slow test compares token counts with the real server.

**Out of GPU memory, twice.** The first training step failed because Ollama still held the 7B teacher (5.3 GB);
`train.py` now unloads Ollama models first. Batch 4 still did not fit under the 11.45 GB Metal limit, so
`grad_checkpoint` was turned on (same hyperparameters, same math, slower). Peak memory then 7.50 GB.

**mlx-lm 0.32 changed its log format** to a styled table. Instead of parsing printed text, `train.py` calls mlx-lm's
own functions and records losses through its `TrainingCallback`.

**Training.** 287 steps (2 epochs of 574 examples), 51 min, validation loss 1.098 -> 0.0713
(`results/train/lora.json`, `results/train/loss.png`).

**Disk.** `ollama create` from safetensors keeps a raw copy and an f16 copy of the weights (about 6.2 GB) until the
server restarts. That blocked the distilled import until Ollama was restarted.

**Ollama updated itself.** The restart upgraded Ollama from 0.18.2 to 0.40.15, which no longer merges consecutive
messages: the retry prompt changed again. Traces, training data and the first two evaluations were made with
0.18.2, so 0.18.2 was reinstalled and run with `ollama serve` (no auto-updater). The token test passed again.

**Evaluation.** Student before fine-tuning 19/50, distilled student 37/50, teacher 43/50, all on Ollama 0.18.2
(`results/tables.md`). Split of the held-out score: distilled student 8/10 on h01-h10 and 17/20 on h11-h30, teacher
8/10 and 18/20, so the new questions did not favour the student.

**A hypothesis that did not hold.** The student is weaker on dev (12/20) than on held-out (25/30). I expected dev
questions to be further from the training questions; they are not (mean closest Jaccard 0.364 vs 0.345 and
0.351), and accuracy does not rise with similarity. Fisher exact test p = 0.10: possibly chance.

**Dirty SHAs.** `git_sha()` flagged any changed tracked file, including result files, so a script always looked
"dirty" once an earlier script had written its output. `results/` is now excluded from the check, and the cheap
steps were rerun on a clean tree: identical numbers, and the rebuilt training data has the same sha256 as the one
used for training.
