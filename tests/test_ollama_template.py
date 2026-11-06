"""Slow: the real Ollama server builds the same prompt as the Hugging Face template used in training.

Ollama doesn't return the prompt text, but it returns how many prompt tokens it processed. If the two templates
differed by even one marker or newline, the counts would differ. Needs `student-base` imported and the model in
models/. Each request unloads the model afterwards (keep_alive 0), so no cached prefix lowers the count.
"""

import json
import urllib.request

import pytest

from distill.chat_format import collate, render_hf
from distill.runinfo import REPO_ROOT

from .test_chat_template import CALLS

pytestmark = pytest.mark.slow


def prompt_tokens_ollama(messages):
    body = {
        "model": "student-base",
        "messages": messages,
        "stream": False,
        "keep_alive": 0,
        "options": {"num_predict": 1, "temperature": 0},
    }
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["prompt_eval_count"]


@pytest.mark.parametrize("call", CALLS)
def test_ollama_sees_the_training_prompt(call):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO_ROOT / "models/qwen2.5-coder-1.5b-instruct")
    expected = len(tok.encode(render_hf(collate(CALLS[call]), add_generation_prompt=True)))
    assert prompt_tokens_ollama(CALLS[call]) == expected
