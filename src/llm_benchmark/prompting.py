from __future__ import annotations

import hashlib

from .models import DatasetExample

PROMPT_VERSION = "mmlu_pro_cot_v1"
PROMPT_TEMPLATE = """Answer the following multiple-choice question.
Respond with exactly one uppercase option letter from these available choices: {allowed_labels}
Do not include words, punctuation, explanation, or any other text.
Example valid response:
{example_label}

Question: {question}

{options}
"""


def build_prompt(example: DatasetExample) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in zip(example.allowed_labels, example.options, strict=True))
    return PROMPT_TEMPLATE.format(
        question=example.question,
        options=options,
        allowed_labels=", ".join(example.allowed_labels),
        example_label="B" if "B" in example.allowed_labels else example.allowed_labels[0],
    )


def prompt_hash() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
