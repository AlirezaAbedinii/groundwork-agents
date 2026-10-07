"""How a quote is matched against text, shared by the validator and the harnesses.

Labels are checked and relevance is scored with the same two functions, so a quote
the validator accepts is a quote the scorer can find. Matching ignores whitespace
(line wraps, indentation) and keeps case.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # schemas imports this module
    from schemas import EvidenceItem

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Collapse every whitespace run to one space and strip the ends; case is kept."""
    return _WS.sub(" ", text).strip()


def covered_items(hit_text: str, evidence: Sequence[EvidenceItem]) -> frozenset[int]:
    """Indexes of the evidence items with at least one quote inside ``hit_text``."""
    hit = normalize_ws(hit_text)
    return frozenset(
        i
        for i, item in enumerate(evidence)
        if any(normalize_ws(quote.text) in hit for quote in item.quotes)
    )
