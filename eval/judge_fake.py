"""A judge that answers from a script, for tests and the keyless smoke runs.

It has ``judge.Judge``'s methods and identity attributes but no provider, cache or
cost. Each method returns a passing verdict unless a function was given for its kind;
that function receives the method's arguments and returns the verdict. Every request
is recorded in ``requests``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from budget import Budget
from schemas import (
    ClaimVerdict,
    Context,
    CorrectnessVerdict,
    CriterionVerdict,
    FaithfulnessVerdict,
    RubricCriterion,
    TaskVerdict,
)

SCRIPTED = "scripted"


def _correct(question: str, reference: str, answer: str) -> CorrectnessVerdict:
    return CorrectnessVerdict(reasoning=SCRIPTED, rating=5)


def _faithful(contexts: Sequence[Context], answer: str) -> FaithfulnessVerdict:
    return FaithfulnessVerdict(
        claims=[ClaimVerdict(claim=answer, reasoning=SCRIPTED, supported=True)]
    )


def _all_passed(
    request: str, final_output: str, subtask_outputs: str, criteria: Sequence[RubricCriterion]
) -> TaskVerdict:
    return TaskVerdict(
        criteria=[CriterionVerdict(id=c.id, reasoning=SCRIPTED, passed=True) for c in criteria]
    )


class FakeJudge:
    provider = "fake"

    def __init__(
        self,
        *,
        correctness: Callable[[str, str, str], CorrectnessVerdict] = _correct,
        faithfulness: Callable[[Sequence[Context], str], FaithfulnessVerdict] = _faithful,
        task: Callable[[str, str, str, Sequence[RubricCriterion]], TaskVerdict] = _all_passed,
        model: str = SCRIPTED,
        prompt_version: str = "fake",
    ) -> None:
        self.model = model
        self.prompt_version = prompt_version
        self.cache_hits = 0
        self.budget = Budget(0.0)
        self.requests: list[tuple[str, tuple]] = []
        self._correctness = correctness
        self._faithfulness = faithfulness
        self._task = task

    def correctness(self, question: str, reference: str, answer: str) -> CorrectnessVerdict:
        self.requests.append(("correctness", (question, reference, answer)))
        return self._correctness(question, reference, answer)

    def faithfulness(self, contexts: Sequence[Context], answer: str) -> FaithfulnessVerdict:
        if not contexts:  # the same input checks as the real judge
            raise ValueError("faithfulness needs the contexts the answer was written from")
        self.requests.append(("faithfulness", (contexts, answer)))
        return self._faithfulness(contexts, answer)

    def task(
        self,
        request: str,
        final_output: str,
        subtask_outputs: str,
        criteria: Sequence[RubricCriterion],
    ) -> TaskVerdict:
        if not criteria:
            raise ValueError("a task verdict needs at least one rubric criterion")
        self.requests.append(("task", (request, final_output, subtask_outputs, criteria)))
        return self._task(request, final_output, subtask_outputs, criteria)
