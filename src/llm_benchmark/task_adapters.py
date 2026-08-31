from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import DatasetExample, ProviderResponse
from .parser import parse_multiple_choice
from .prompting import (
    build_prompt,
)
from .prompting import (
    prompt_hash as multiple_choice_prompt_hash,
)


@dataclass(frozen=True)
class TaskEvaluationOutcome:
    parsed_answer: str | None
    parse_status: str
    candidate_answers: tuple[str, ...]
    evaluation_status: str
    is_correct: bool


class TaskAdapter(Protocol):
    task_type: str

    def build_prompt(self, example: DatasetExample) -> str:
        """Build the provider input for one evaluation example"""
        ...

    def prompt_template_hash(self) -> str:
        """Return the stable hash of the task prompt template."""
        ...

    def evaluate_response(
            self,
            example: DatasetExample,
            response: ProviderResponse,
    ) -> TaskEvaluationOutcome:
        """Interpret a normalized provider response for this task"""


class MultipleChoiceTaskAdapter:
    task_type = "multiple_choice"

    def build_prompt(self, example: DatasetExample) -> str:
        return build_prompt(example)

    def prompt_template_hash(self) -> str:
        return multiple_choice_prompt_hash()

    def evaluate_response(
        self,
        example: DatasetExample,
        response: ProviderResponse,
    ) -> TaskEvaluationOutcome:
        parsed = parse_multiple_choice(
            response.raw_response,
            example.allowed_labels,
        )

        if response.request_status != "succeeded":
            evaluation_status = "request_failed"
        elif parsed.parsed_answer is None:
            evaluation_status = "unparseable"
        elif parsed.parsed_answer == example.correct_answer:
            evaluation_status = "correct"
        else:
            evaluation_status = "incorrect"

        return TaskEvaluationOutcome(
            parsed_answer=parsed.parsed_answer,
            parse_status=parsed.parse_status,
            candidate_answers=tuple(parsed.candidate_answers),
            evaluation_status=evaluation_status,
            is_correct=evaluation_status == "correct",
        )


DEFAULT_TASK_ADAPTER: TaskAdapter = MultipleChoiceTaskAdapter()