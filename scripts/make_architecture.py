"""Draw docs/architecture.png: teacher -> traces -> filter -> LoRA -> student -> same agent -> evaluation.

python scripts/make_architecture.py     # ~1 s

Every number on the figure is read from results/ (nothing typed by hand). The figure has its own light background,
so it stays readable on GitHub in light and dark mode.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from distill.runinfo import REPO_ROOT, RESULTS, load_yaml  # noqa: E402

SURFACE, INK, MUTED, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#0b0b0b"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # teacher side / student side (categorical slots 1-2)


def box(ax, x, y, title, sub, color, w=0.2, h=0.25):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                fc="white", ec=color, lw=2))  # fmt: skip
    ax.text(x, y + 0.045, title, ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
    ax.text(x, y - 0.05, sub, ha="center", va="center", fontsize=9.5, color=MUTED, linespacing=1.35)


def arrow(ax, a, b, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.5, color=INK,
                                 connectionstyle=f"arc3,rad={rad}"))  # fmt: skip


def main() -> None:
    split = load_yaml(RESULTS / "split/overlap.json")
    filt = load_yaml(RESULTS / "filter/report.json")
    ev = {m: load_yaml(RESULTS / f"eval/{m}.json")["summary"]["all"] for m in ("student_base", "student_distilled", "teacher")}  # fmt: skip
    n_q = split["n_train"] + split["n_valid"]

    fig = plt.figure(figsize=(12, 5.4), facecolor=SURFACE)
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top, bottom = 0.7, 0.24
    xs = [0.12, 0.37, 0.63, 0.88]
    box(ax, xs[0], top, "Teacher", "Qwen2.5-Coder 7B\nruns the agent", BLUE)
    box(ax, xs[1], top, "Traces", f"{n_q} training questions\nevery LLM call saved", BLUE)
    box(
        ax,
        xs[2],
        top,
        "Filter",
        f"keep correct answers only\n{filt['n_kept']} of {filt['n_traces']} kept",
        BLUE,
    )
    box(
        ax,
        xs[3],
        top,
        "LoRA fine-tuning",
        f"{filt['n_examples']['train']} examples\n(one per LLM call)",
        ORANGE,
    )
    for a, b in zip(xs, xs[1:], strict=False):
        arrow(ax, (a + 0.1, top), (b - 0.1, top))

    box(ax, xs[3], bottom, "Student", "Qwen2.5-Coder 1.5B\n+ LoRA adapter", ORANGE)
    box(ax, xs[2], bottom, "Same agent", "same graph, prompts,\nsandbox; only the model changes", INK, w=0.22)
    box(ax, xs[1] - 0.04, bottom, "Evaluation", "50 frozen questions\nnever used in training", INK)
    arrow(ax, (xs[3], top - 0.125), (xs[3], bottom + 0.125))
    arrow(ax, (xs[3] - 0.1, bottom), (xs[2] + 0.11, bottom))
    arrow(ax, (xs[2] - 0.11, bottom), (xs[1] + 0.06, bottom))

    # result box, fed by the evaluation box
    res = (f"student before: {ev['student_base']['n_correct']}/50\n"
           f"student distilled: {ev['student_distilled']['n_correct']}/50\n"
           f"teacher: {ev['teacher']['n_correct']}/50")  # fmt: skip
    box(ax, 0.1, bottom, "Correct answers", res, INK, w=0.17, h=0.29)
    arrow(ax, (xs[1] - 0.14, bottom), (0.185, bottom))
    ax.text(0.0, 0.98, "Distilling the agent: the student learns only from the teacher's correct traces",
            fontsize=14, weight="bold", color=INK, va="top")  # fmt: skip

    out = REPO_ROOT / "docs/architecture.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
