"""The student must see the same text at training time (Hugging Face template, MLX) and at serving time (Ollama)."""

import pytest
from analyst.graph import ANSWER_SYSTEM, RETRY_HINTS_PROMPT, UNDERSTAND_SYSTEM, WRITE_SYSTEM

from distill.chat_format import OLLAMA_TEMPLATE, render_hf, render_ollama

SCHEMA = "8 rows. Columns (name: dtype):\n- 'Invoice': str\n- 'Price': float64"
CODE = '{"plan": "sum revenue", "code": "result = (df[\'Quantity\'] * df[\'Price\']).sum()"}'


def msg(role, content):
    return {"role": role, "content": content}


# The four kinds of LLM calls the agent makes, with the agent's real system prompts.
CALLS = {
    "understand": [
        msg("system", UNDERSTAND_SYSTEM),
        msg("user", f"Data:\n{SCHEMA}\n\nQuestion: How many rows?"),
    ],
    "write_code": [msg("system", WRITE_SYSTEM), msg("user", f"Data:\n{SCHEMA}\n\nQuestion: Total revenue?")],
    "retry": [
        msg("system", WRITE_SYSTEM),
        msg("user", f"Data:\n{SCHEMA}\n\nQuestion: Total revenue?"),
        msg(
            "user",
            RETRY_HINTS_PROMPT.format(
                numbered="  1|result = df['Revenue'].sum()", error="KeyError", repeat=""
            ),
        ),
    ],
    "answer": [msg("system", ANSWER_SYSTEM), msg("user", "Question: How many rows?\n\nExecution result:\n8")],
}


@pytest.mark.parametrize("call", CALLS)
def test_training_prompt_equals_serving_prompt(call):
    assert render_hf(CALLS[call], add_generation_prompt=True) == render_ollama(CALLS[call])


@pytest.mark.parametrize("call", CALLS)
def test_prompt_is_a_prefix_of_the_training_example(call):
    """--mask-prompt hides the prompt tokens from the loss: that only works if the prompt is an exact prefix."""
    prompt = render_hf(CALLS[call], add_generation_prompt=True)
    full = render_hf([*CALLS[call], msg("assistant", CODE)], add_generation_prompt=False)
    assert full == prompt + CODE + "<|im_end|>\n"  # the student learns to stop with <|im_end|>


def test_go_template_is_the_one_mirrored_in_python():
    expected = "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n"
    assert OLLAMA_TEMPLATE.read_text() == expected


def test_without_a_system_message_the_two_templates_differ():
    """Qwen's HF template adds a default system prompt; ours doesn't. The agent always sends one, so it's fine."""
    user_only = [msg("user", "hi")]
    assert "You are Qwen" in render_hf(user_only, True) and "You are Qwen" not in render_ollama(user_only)
