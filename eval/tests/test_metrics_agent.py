"""Acceptance tests for the agent metrics (metrics.py part C).

Tool-call scores come from the agent's invocation log; task success comes from the
rubric judge's verdict. Expected values are worked out in the comments.
"""

import pytest

import metrics
from schemas import (
    AgentTask,
    CriterionVerdict,
    ExpectedCall,
    RubricCriterion,
    TaskVerdict,
    ToolCall,
)


def task(**overrides) -> AgentTask:
    base = dict(
        id="t01",
        request="Look up the cache docs and write notes.md",
        category="docs",
        must_call=[
            ExpectedCall(tool="rag_search"),
            ExpectedCall(tool="file_write", args={"path": "notes.md"}),
        ],
        may_call=["code_exec"],
        forbidden=["api_call"],
        rubric=[
            RubricCriterion(id="c1", description="Names UV_CACHE_DIR"),
            RubricCriterion(id="c2", description="Under 200 words", critical=False),
            RubricCriterion(id="c3", description="Invents no flags"),
        ],
    )
    return AgentTask(**(base | overrides))


def call(tool: str, status: str = "success", **arguments) -> ToolCall:
    return ToolCall(tool_name=tool, specialist="researcher", arguments=arguments, status=status)


# --- must-call recall ---------------------------------------------------------------


def test_must_call_recall_with_and_without_args():
    calls = [call("rag_search", query="cache"), call("file_write", path="notes.md", content="x")]
    # both entries satisfied; extra arguments in the call don't matter (args is a subset)
    assert metrics.tool_call_scores(calls, task()).must_call_recall == 1.0
    calls = [call("rag_search", query="cache"), call("file_write", path="other.md")]
    # file_write was called, but not with path="notes.md": 1 of 2 entries
    assert metrics.tool_call_scores(calls, task()).must_call_recall == 0.5


def test_a_failed_call_does_not_satisfy_a_must_call():
    for status in ("failure", "rejected", "rate_limited"):
        calls = [call("rag_search", status=status), call("file_write", path="notes.md")]
        assert metrics.tool_call_scores(calls, task()).must_call_recall == 0.5


def test_args_must_match_in_type_too():
    t = task(must_call=[ExpectedCall(tool="db_query", args={"limit": 3, "strict": True})])
    assert (
        metrics.tool_call_scores([call("db_query", limit=3, strict=True)], t).must_call_recall
        == 1.0
    )
    assert (
        metrics.tool_call_scores([call("db_query", limit="3", strict=True)], t).must_call_recall
        == 0.0
    )
    assert (
        metrics.tool_call_scores([call("db_query", limit=3, strict=1)], t).must_call_recall == 0.0
    )
    assert (
        metrics.tool_call_scores([call("db_query", limit=3.0, strict=True)], t).must_call_recall
        == 0.0
    )


def test_an_entry_with_alternatives_is_met_by_any_of_them():
    t = task(must_call=[ExpectedCall(tool=["rag_search", "rag_ask"])])
    for tool in ("rag_search", "rag_ask"):
        assert metrics.tool_call_scores([call(tool)], t).must_call_recall == 1.0
    # a failed call to one of them, or a successful call to another tool, doesn't meet it
    for calls in ([call("rag_ask", status="failure")], [call("rag_list_sources")]):
        assert metrics.tool_call_scores(calls, t).must_call_recall == 0.0


def test_alternatives_share_the_entry_args():
    t = task(must_call=[ExpectedCall(tool=["rag_search", "rag_ask"], args={"mode": "dense"})])
    assert metrics.tool_call_scores([call("rag_ask", mode="dense")], t).must_call_recall == 1.0
    assert metrics.tool_call_scores([call("rag_search", mode="hybrid")], t).must_call_recall == 0.0


def test_no_must_calls_leaves_recall_undefined():
    assert (
        metrics.tool_call_scores([call("code_exec")], task(must_call=[])).must_call_recall is None
    )


# --- precision, errors, forbidden calls ---------------------------------------------


def test_precision_counts_calls_to_must_call_and_may_call_tools():
    calls = [
        call("rag_search"),
        call("file_write", path="notes.md"),
        call("code_exec"),
        call("web_search"),
    ]
    # 3 of 4 calls went to tools the task names (web_search isn't one)
    assert metrics.tool_call_scores(calls, task()).precision == 0.75


def test_precision_counts_the_tool_chosen_whatever_the_call_status():
    calls = [call("rag_search", status="failure"), call("web_search")]
    # the failed rag_search still went to a tool the task names (an error, not a detour)
    assert metrics.tool_call_scores(calls, task()).precision == 0.5


def test_precision_counts_every_alternative_as_named():
    t = task(must_call=[ExpectedCall(tool=["rag_search", "rag_ask"])], may_call=[])
    calls = [call("rag_search"), call("rag_ask"), call("web_search"), call("rag_ask")]
    # rag_search and rag_ask are both named by the must_call entry; web_search isn't: 3 of 4
    assert metrics.tool_call_scores(calls, t).precision == 0.75


def test_error_rate_counts_failure_rejected_and_rate_limited():
    calls = [
        call("rag_search"),
        call("rag_search", status="failure"),
        call("file_write", status="rejected"),
        call("rag_search", status="rate_limited"),
    ]
    assert metrics.tool_call_scores(calls, task()).error_rate == 0.75


def test_forbidden_calls_count_at_any_status():
    calls = [call("api_call", status="rejected"), call("api_call"), call("rag_search")]
    scores = metrics.tool_call_scores(calls, task())
    assert scores.forbidden_calls == 2
    assert scores.n_calls == 3


def test_no_calls():
    scores = metrics.tool_call_scores([], task())
    assert scores.must_call_recall == 0.0  # 0 of 2 required calls made
    assert scores.precision is None  # 0/0
    assert scores.error_rate is None  # 0/0
    assert (scores.forbidden_calls, scores.n_calls) == (0, 0)


# --- task success -------------------------------------------------------------------


def verdict(**passed: bool) -> TaskVerdict:
    return TaskVerdict(
        criteria=[CriterionVerdict(id=i, reasoning="r", passed=p) for i, p in passed.items()]
    )


def test_success_when_every_critical_criterion_passes():
    assert metrics.task_success(verdict(c1=True, c2=True, c3=True), task()) is True


def test_a_failed_non_critical_criterion_does_not_fail_the_task():
    assert metrics.task_success(verdict(c1=True, c2=False, c3=True), task()) is True


def test_a_failed_critical_criterion_fails_the_task():
    assert metrics.task_success(verdict(c1=True, c2=True, c3=False), task()) is False


def test_a_critical_criterion_missing_from_the_verdict_fails_closed():
    assert metrics.task_success(verdict(c1=True, c2=True), task()) is False


def test_a_task_without_a_critical_criterion_cannot_be_scored():
    t = task(rubric=[RubricCriterion(id="c2", description="Under 200 words", critical=False)])
    with pytest.raises(ValueError):
        metrics.task_success(verdict(c2=True), t)


def test_criteria_passed_fails_closed():
    t = task()  # rubric c1, c2 (non-critical), c3
    # c1 passed, c2 is missing, c3 is listed twice with one failure, c9 isn't in the rubric
    v = TaskVerdict(
        criteria=[
            CriterionVerdict(id="c1", reasoning="r", passed=True),
            CriterionVerdict(id="c3", reasoning="r", passed=True),
            CriterionVerdict(id="c3", reasoning="r", passed=False),
            CriterionVerdict(id="c9", reasoning="r", passed=True),
        ]
    )
    assert metrics.criteria_passed(v, t) == {"c1": True, "c2": False, "c3": False}
    assert metrics.task_success(v, t) is False  # c3 is critical
