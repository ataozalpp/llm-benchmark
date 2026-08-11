from __future__ import annotations

import hashlib

from .models import DatasetExample

PROMPT_VERSION = "mmlu_pro_cot_v1"
PROMPT_TEMPLATE = """Answer the following multiple-choice question.
Solve it carefully and put your final answer on the last line exactly as:
FINAL ANSWER: <OPTION_LETTER>

Question: {question}

{options}
"""


def build_prompt(example: DatasetExample) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in zip(example.allowed_labels, example.options, strict=True))
    return PROMPT_TEMPLATE.format(question=example.question, options=options)


def prompt_hash() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
