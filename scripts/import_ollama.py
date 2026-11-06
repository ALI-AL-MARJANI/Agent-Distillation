"""Import a Hugging Face-format model folder into Ollama, quantized like the teacher (Q4_K_M).

python scripts/import_ollama.py models/qwen2.5-coder-1.5b-instruct student-base         # ~1 min (estimate)
python scripts/import_ollama.py models/fused-distilled student-distilled

The base and the distilled student go through exactly this path, so the only difference between them is the
LoRA update. The chat template is configs/ollama/chatml.gotmpl (tests/test_chat_template.py).
"""

from __future__ import annotations

import argparse
import subprocess

from distill.chat_format import OLLAMA_TEMPLATE
from distill.runinfo import REPO_ROOT

QUANT = "q4_K_M"  # same quantization as the teacher's qwen2.5-coder:7b


def modelfile(weights: str) -> str:
    return (
        f"FROM {(REPO_ROOT / weights).resolve()}\n"
        f'TEMPLATE """{OLLAMA_TEMPLATE.read_text()}"""\n'
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|endoftext|>"\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="folder with config.json + *.safetensors")
    ap.add_argument("name", help="Ollama model name")
    args = ap.parse_args()
    path = REPO_ROOT / f"models/{args.name}.Modelfile"
    path.write_text(modelfile(args.weights))
    subprocess.run(["ollama", "create", args.name, "-q", QUANT, "-f", str(path)], check=True)


if __name__ == "__main__":
    main()
