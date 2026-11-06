"""LoRA fine-tuning of the student with mlx-lm, then save the loss curve and the adapter.

python scripts/train.py --timing      # 20 steps, to measure time per step before the real run
python scripts/train.py               # full run with configs/lora.yaml

-> results/train/<run>.json (losses, timing, peak memory, metadata), results/train/loss.png,
   results/train/adapter/ (adapter_config.json + adapters.safetensors, a few MB)

Same steps as `mlx_lm.lora -c configs/lora.yaml` (mlx-lm 0.32.2, lora.py main/run), called from Python so the
losses come from mlx-lm's training callback instead of being parsed from its printed table.
"""

from __future__ import annotations

import argparse
import shutil
import time
from types import SimpleNamespace

import numpy as np
from mlx_lm.lora import CONFIG_DEFAULTS, load_dataset, train_model
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.utils import load

from distill.ollama_api import unload_all
from distill.runinfo import REPO_ROOT, RESULTS, load_config, run_metadata, sha256_file, write_json


class LossRecorder(TrainingCallback):
    def __init__(self) -> None:
        self.train: list[dict] = []
        self.val: list[dict] = []

    def on_train_loss_report(self, info: dict) -> None:
        self.train.append({k: float(v) for k, v in info.items()})

    def on_val_loss_report(self, info: dict) -> None:
        self.val.append({k: float(v) for k, v in info.items()})


def plot(losses: dict, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"  # categorical slots 1-2, validated together
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=surface)
    ax.set_facecolor(surface)
    series = (("train", "train_loss", blue, "train loss"), ("val", "val_loss", orange, "validation loss"))
    for key, field, color, label in series:
        if pts := losses[key]:
            x, y = [p["iteration"] for p in pts], [p[field] for p in pts]
            ax.plot(x, y, color=color, lw=2, marker="o" if key == "val" else None, ms=4, label=label)
    ax.set_xlabel("training step", color=muted)
    ax.set_ylabel("loss on the teacher's tokens", color=muted)
    ax.grid(color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, labelcolor=ink)
    ax.set_title(
        "LoRA fine-tuning of Qwen2.5-Coder-1.5B on teacher traces", color=ink, loc="left", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=surface)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora.yaml")
    ap.add_argument("--timing", action="store_true", help="20 steps into a scratch adapter folder")
    a = ap.parse_args()

    cfg = load_config(REPO_ROOT / a.config)
    run = "timing" if a.timing else "lora"
    if a.timing:
        cfg = {**cfg, "iters": 20, "steps_per_eval": 1000, "adapter_path": "models/adapters/timing"}
    args = {**CONFIG_DEFAULTS, **cfg}
    for key in ("model", "data", "adapter_path"):  # relative to the repo root
        args[key] = str(REPO_ROOT / args[key])

    unload_all()  # a loaded Ollama model leaves too little GPU memory for training
    t0 = time.perf_counter()
    np.random.seed(args["seed"])
    model, tokenizer = load(args["model"])
    train_set, valid_set, _ = load_dataset(SimpleNamespace(**args), tokenizer)
    recorder = LossRecorder()
    train_model(SimpleNamespace(**args), model, train_set, valid_set, recorder)
    wall = time.perf_counter() - t0

    losses = {"train": recorder.train, "val": recorder.val}
    summary = {
        **run_metadata(config=cfg),
        "mlx_lm_args": args,
        "wall_clock_s": round(wall, 1),  # includes model loading and validation passes
        "validation_s": round(sum(p["val_time"] for p in recorder.val), 1),
        "peak_mem_gb": max((p["peak_memory"] for p in recorder.train), default=None),
        "train_examples_sha256": sha256_file(REPO_ROOT / "data/sft/train.jsonl"),
        "losses": losses,
    }
    if not a.timing:
        out = RESULTS / "train/adapter"
        out.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapters.safetensors"):
            shutil.copy(REPO_ROOT / cfg["adapter_path"] / name, out / name)
        summary["adapter_sha256"] = sha256_file(out / "adapters.safetensors")
        plot(losses, RESULTS / "train/loss.png")
    write_json(RESULTS / f"train/{run}.json", summary)
    print(
        f"wall clock {wall:.0f} s, validation {summary['validation_s']} s, peak mem {summary['peak_mem_gb']} GB"
    )


if __name__ == "__main__":
    main()
