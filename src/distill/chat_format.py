"""The exact text the student sees, at training time and at serving time. Both must be identical.

- Training (MLX-LM) turns a list of messages into text with the Hugging Face chat template that ships with
  Qwen2.5-Coder-1.5B-Instruct (vendored in configs/chat_templates/, pinned to one model revision).
- Serving (Ollama) uses the Go template in configs/ollama/chatml.gotmpl, written into the student's Modelfile.

`render_ollama` mirrors that Go template line by line in Python. Go templates can't run offline in Python,
so a slow test also checks the real Ollama server (same prompt token count) once the student is imported.
"""

from __future__ import annotations

from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

from distill.runinfo import REPO_ROOT

HF_TEMPLATE = REPO_ROOT / "configs/chat_templates/qwen2.5-coder-1.5b-instruct.jinja"
HF_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"  # Qwen/Qwen2.5-Coder-1.5B-Instruct on Hugging Face
OLLAMA_TEMPLATE = REPO_ROOT / "configs/ollama/chatml.gotmpl"


def render_hf(messages: list[dict], add_generation_prompt: bool, template: Path = HF_TEMPLATE) -> str:
    """Same Jinja settings as transformers' apply_chat_template (trim_blocks, lstrip_blocks, sandboxed)."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template.read_text()).render(
        messages=messages, tools=None, add_generation_prompt=add_generation_prompt
    )


def render_ollama(messages: list[dict]) -> str:
    """Python copy of configs/ollama/chatml.gotmpl. Ollama always ends with the generation prompt."""
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return turns + "<|im_start|>assistant\n"
