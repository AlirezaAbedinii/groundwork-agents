"""The types every part of the evaluation passes around.

Golden questions and agent tasks (the ground truth), the judge verdicts (also sent
to the judge as its structured-output schema), the tool calls read from the agent's
invocation log, and the results the metrics return. The metrics and judge prompts
are written against these; a change to the contract starts here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from textnorm import normalize_ws

Category = Literal["lookup", "multi_hop", "no_answer", "ambiguous"]
Origin = Literal["targeted", "synthetic"]
Split = Literal["dev", "test"]
ToolStatus = Literal["success", "failure", "rejected", "rate_limited"]

# Each origin's id prefix: q001… written to the coverage spec, s001… drafted from one
# chunk. Ids are never reused.
ID_PREFIX: dict[str, str] = {"targeted": "q", "synthetic": "s"}


class _Strict(BaseModel):
    # Authored data and judge verdicts: a misspelled key is an error, not a silent default.
    model_config = ConfigDict(extra="forbid")


# --- the golden set -----------------------------------------------------------------


class Quote(_Strict):
    """A verbatim excerpt of one section of one corpus file."""

    source: str  # path relative to the corpus root, e.g. "uv/concepts/cache.md"
    text: str = Field(min_length=12, max_length=120)  # verbatim, one section, no heading line

    @field_validator("source")
    @classmethod
    def _relative_posix(cls, v: str) -> str:
        path = PurePosixPath(v)
        if not v or "\\" in v or path.is_absolute() or ".." in path.parts:
            raise ValueError("must be a path relative to the corpus root, with '/' separators")
        return v

    @field_validator("text")
    @classmethod
    def _not_padding(cls, v: str) -> str:
        if len(normalize_ws(v)) < 12:
            raise ValueError("needs at least 12 characters besides whitespace")
        return v


class EvidenceItem(_Strict):
    """One fact the answer needs. Any one of its quotes covers it (the same fact on two pages)."""

    quotes: list[Quote] = Field(min_length=1)


class GoldenQuestion(_Strict):
    id: str = Field(pattern=r"^[qs]\d{3}$")
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    category: Category
    evidence: list[EvidenceItem]  # [] iff no_answer; >= 2 items for multi_hop and ambiguous
    origin: Origin
    split: Split
    verified: bool
    notes: str = ""

    @model_validator(mode="after")
    def _evidence_fits_category(self) -> Self:
        n = len(self.evidence)
        if self.category == "no_answer" and n:
            raise ValueError("a no_answer question has no evidence (evidence: [])")
        if self.category == "lookup" and not n:
            raise ValueError("a lookup question needs at least 1 evidence item")
        if self.category in ("multi_hop", "ambiguous") and n < 2:
            raise ValueError(f"a {self.category} question needs at least 2 evidence items")
        if not self.id.startswith(ID_PREFIX[self.origin]):
            raise ValueError(f"{self.origin} ids start with {ID_PREFIX[self.origin]!r}")
        return self


# --- answer judge verdicts ----------------------------------------------------------


class CorrectnessVerdict(_Strict):
    reasoning: str
    rating: int = Field(ge=1, le=5)


class ClaimVerdict(_Strict):
    claim: str
    reasoning: str
    supported: bool


class FaithfulnessVerdict(_Strict):
    claims: list[ClaimVerdict]


# --- agent tasks --------------------------------------------------------------------


class RubricCriterion(_Strict):
    id: str
    description: str
    critical: bool = True


class ExpectedCall(_Strict):
    """A call the task needs: a successful call to `tool` whose arguments contain `args`."""

    tool: str
    args: dict[str, str | int | float | bool] = {}


class AgentTask(_Strict):
    id: str
    request: str
    category: str
    must_call: list[ExpectedCall]
    may_call: list[str] = []
    forbidden: list[str] = []
    rubric: list[RubricCriterion]
    notes: str = ""


class CriterionVerdict(_Strict):
    id: str
    reasoning: str
    passed: bool


class TaskVerdict(_Strict):
    criteria: list[CriterionVerdict]


class ToolCall(BaseModel):
    """A row of the agent's invocation log (``GET /traces/{id}/tools``), minus the other fields."""

    tool_name: str
    specialist: str
    arguments: dict
    status: ToolStatus


# --- metric results -----------------------------------------------------------------


@dataclass(frozen=True)
class PRF:
    """Precision, recall and F1 with their counts; a metric is None when its denominator is 0."""

    precision: float | None
    recall: float | None
    f1: float | None
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    prf: PRF


@dataclass(frozen=True)
class ToolCallScores:
    must_call_recall: float | None
    precision: float | None
    error_rate: float | None
    forbidden_calls: int
    n_calls: int
