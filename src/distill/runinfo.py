"""Metadata written next to every result file: git SHA, agent commit, hardware, library versions."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
DATA = REPO_ROOT / "data"

_LIBS = ["analyst", "langgraph", "langchain-core", "langchain-ollama", "pandas", "numpy", "mlx", "mlx-lm"]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text())


def git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        # "-dirty" = uncommitted code, config or input data. results/ is excluded: it holds the outputs, and
        # counting it would mark every script dirty as soon as an earlier script had written its result file.
        dirty = (
            subprocess.call(["git", "diff", "--quiet", "HEAD", "--", ".", ":(exclude)results"], cwd=REPO_ROOT)
            != 0
        )
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"  # no commit yet


def agent_commit() -> str | None:
    """The Data-Analyst-Agent commit actually installed (read from pip's direct_url.json)."""
    with contextlib.suppress(md.PackageNotFoundError, FileNotFoundError, TypeError):
        return json.loads(md.distribution("analyst").read_text("direct_url.json"))["vcs_info"]["commit_id"]
    return None


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def hardware() -> dict:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
    }
    if mem := _sysctl("hw.memsize"):
        info["ram_gb"] = round(int(mem) / 2**30, 1)
    return info


def versions() -> dict:
    out = {}
    for lib in _LIBS:
        with contextlib.suppress(md.PackageNotFoundError):
            out[lib] = md.version(lib)
    return out


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_metadata(**extra: Any) -> dict:
    return {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "agent_commit": agent_commit(),
        "hardware": hardware(),
        "versions": versions(),
        **extra,
    }


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
