"""The few Ollama HTTP calls used here: list loaded models, unload them, read model info."""

from __future__ import annotations

import json
import urllib.request

OLLAMA = "http://localhost:11434"


def ollama(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(OLLAMA + path, data=data), timeout=60) as r:
        return json.loads(r.read())


def unload_all() -> None:
    """Free the GPU memory held by Ollama (models stay on disk). Needed before training: a loaded 7B model
    (5.3 GB) plus LoRA training of the 1.5B ran out of memory on the 16 GB M4."""
    for m in ollama("/api/ps")["models"]:
        ollama("/api/generate", {"model": m["name"], "keep_alive": 0})


def model_info(name: str) -> dict:
    loaded = {m["name"].removesuffix(":latest"): m for m in ollama("/api/ps")["models"]}
    tags = {m["name"].removesuffix(":latest"): m for m in ollama("/api/tags")["models"]}
    m, t = loaded.get(name, {}), tags.get(name, {})
    return {
        "model": name,
        "server_version": ollama("/api/version")["version"],
        "digest": t.get("digest"),
        "file_size_bytes": t.get("size"),
        "quantization": t.get("details", {}).get("quantization_level"),
        "parameter_size": t.get("details", {}).get("parameter_size"),
        "loaded_size_bytes": m.get("size"),
        "loaded_size_vram_bytes": m.get("size_vram"),
        "context_length": m.get("context_length"),
    }
