from __future__ import annotations

import re

from .models import ParseResult

PARSER_VERSION = "mcq_parser_v1"


def parse_multiple_choice(raw: str | None, allowed_labels: list[str]) -> ParseResult:
    if raw is None:
        return ParseResult(None, "request_failed", [])
    text = raw.strip()
    if not text:
        return ParseResult(None, "empty_response", [])
    allowed = "".join(re.escape(x) for x in allowed_labels)
    exact = re.fullmatch(rf"[\s\(\[\"']*([{allowed}])[\s\)\]\"'\.,;:]*", text, re.IGNORECASE)
    if exact:
        answer = exact.group(1).upper()
        return ParseResult(answer, "normalized_label", [answer])
    marker_pattern = rf"(?:final\s+answer|answer|final|choice|option|cevap|nihai\s+cevap)\s*[:\-]?\s*[\(\[]?([{allowed}])\b"
    marker_candidates = [x.upper() for x in re.findall(marker_pattern, text, re.IGNORECASE)]
    boxed = [x.upper() for x in re.findall(rf"\\boxed\{{\s*([{allowed}])\s*\}}", text, re.IGNORECASE)]
    leading_match = re.match(rf"^\s*[\(\[]?([{allowed}])[\)\]\.,:]\s+", text, re.IGNORECASE)
    explicit = marker_candidates + boxed
    all_standalone = [x.upper() for x in re.findall(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])", text, re.IGNORECASE)]
    distinct_all = sorted(set(all_standalone))
    if len(distinct_all) > 1:
        return ParseResult(None, "ambiguous_multiple_answers", distinct_all)
    if explicit:
        distinct = sorted(set(explicit))
        if len(distinct) == 1:
            return ParseResult(distinct[0], "explicit_final_marker" if marker_candidates else "boxed_label", distinct)
        return ParseResult(None, "ambiguous_multiple_answers", distinct)
    if leading_match:
        answer = leading_match.group(1).upper()
        return ParseResult(answer, "leading_label", [answer])
    return ParseResult(None, "no_answer_found", distinct_all)
