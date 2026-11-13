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
    PlannedSubtask,
    RubricCriterion,
    Share,
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


# --- diagnostics --------------------------------------------------------------------


def located(tool: str, sid: str | None, status: str = "success", chars=None, **arguments):
    return ToolCall(
        tool_name=tool,
        specialist="research",
        arguments=arguments,
        status=status,
        sid=sid,
        result_chars=chars,
    )


def plan(*specialists: str) -> list[PlannedSubtask]:
    return [
        PlannedSubtask(sid=f"s{i}", specialist=s, depends_on=[f"s{i - 1}"] if i > 1 else [])
        for i, s in enumerate(specialists, start=1)
    ]


def test_a_share_is_undefined_without_a_denominator():
    assert Share(k=1, n=4).rate == 0.25
    assert Share(k=0, n=0).rate is None


def test_plan_size_and_whether_it_starts_with_research():
    assert metrics.plan_size(plan("research", "analysis", "writing")) == 3
    assert metrics.starts_with_research(plan("research", "writing")) is True
    # research later in the plan doesn't count: the first subtask is another specialist's
    assert metrics.starts_with_research(plan("analysis", "research")) is False
    # a task that never got a plan has an empty one, which starts with nothing
    assert (metrics.plan_size([]), metrics.starts_with_research([])) == (0, False)
    # an older record holds no plan at all: not recorded, not zero
    assert (metrics.plan_size(None), metrics.starts_with_research(None)) == (None, None)


def test_toolless_completed_counts_completed_subtasks_without_any_call():
    subtasks = [
        {"sid": "s1", "status": "completed"},
        {"sid": "s2", "status": "completed"},
        {"sid": "s3", "status": "completed"},
        {"sid": "s4", "status": "failed"},
    ]
    calls = [
        located("rag_search", "s1"),
        located("code_exec", "s2", status="rate_limited"),  # tried: still a call
        located("db_query", "s4"),
    ]
    # s3 completed with no call; s4 has no call either way but didn't complete
    assert metrics.toolless_completed(subtasks, calls) == Share(k=1, n=3)
    # with no calls at all, every completed subtask is tool-less
    assert metrics.toolless_completed(subtasks, []) == Share(k=3, n=3)


def test_toolless_completed_is_undefined_when_calls_have_no_subtask():
    subtasks = [{"sid": "s1", "status": "completed"}]
    assert metrics.toolless_completed(subtasks, [call("rag_search")]) is None


def test_repeat_call_share_counts_exact_repeats_at_any_status():
    calls = [
        call("rag_search", query="cache", top_k=5),
        call("rag_search", status="rate_limited", top_k=5, query="cache"),  # repeat: key order
        call("rag_search", query="cache", top_k=10),  # other args: new
        call("web_search", query="cache", top_k=5),  # other tool: new
        call("rag_search", query="cache", top_k=5),  # repeat again
        call("rag_search", query="cache", top_k="5"),  # "5" isn't 5: new
    ]
    assert metrics.repeat_call_share(calls) == Share(k=2, n=6)
    assert metrics.repeat_call_share([]) == Share(k=0, n=0)


def test_rate_limited_share():
    calls = [call("code_exec", status="rate_limited"), call("code_exec"), call("db_query")]
    assert metrics.rate_limited_share(calls) == Share(k=1, n=3)
    assert metrics.rate_limited_share([]).rate is None


def test_cut_result_share_counts_results_longer_than_the_budget():
    calls = [
        located("rag_search", "s1", chars=2001),  # over
        located("rag_search", "s1", chars=2000),  # exactly the budget: shown whole
        located("db_query", "s1", chars=12),
        located("code_exec", "s1", status="rate_limited"),  # no result: left out
    ]
    assert metrics.cut_result_share(calls) == Share(k=1, n=3)
    assert metrics.cut_result_share(calls, budget=10) == Share(k=3, n=3)
    # an older record holds no sizes: nothing to count, so the share is undefined
    assert metrics.cut_result_share([call("rag_search")]).rate is None
    with pytest.raises(ValueError):
        metrics.cut_result_share(calls, budget=0)


def test_critical_pass_share_fails_closed_and_ignores_non_critical_criteria():
    # rubric c1, c2 (non-critical), c3: c1 passed, c3 is missing from the verdict
    assert metrics.critical_pass_share(verdict(c1=True, c2=True), task()) == Share(k=1, n=2)
    assert metrics.critical_pass_share(verdict(c1=True, c3=True), task()) == Share(k=2, n=2)


def test_artifact_at_path_needs_a_successful_write_at_the_exact_path():
    t = task()  # requires notes.md
    assert metrics.artifact_at_path([call("file_write", path="notes.md")], t) == {"notes.md": True}
    for calls in (
        [call("file_write", path="docs/notes.md")],  # another directory
        [call("file_write", path="notes.txt")],  # another name
        [call("file_write", status="rate_limited", path="notes.md")],  # never written
        [call("file_read", path="notes.md")],  # read, not written
    ):
        assert metrics.artifact_at_path(calls, t) == {"notes.md": False}


def test_artifact_at_path_lists_every_required_file_and_only_those():
    t = task(
        must_call=[
            ExpectedCall(tool="file_write", args={"path": "a.md"}),
            ExpectedCall(tool=["file_write", "code_exec"], args={"path": "b.csv"}),
            ExpectedCall(tool="file_write"),  # any file will do: no path required
            ExpectedCall(tool="db_query", args={"path": "c.md"}),  # not a write
        ]
    )
    calls = [call("file_write", path="b.csv"), call("file_write", path="other.md")]
    assert metrics.artifact_at_path(calls, t) == {"a.md": False, "b.csv": True}
    assert metrics.artifact_at_path(calls, task(must_call=[])) == {}


# --- scoring rules version 2 --------------------------------------------------------


def code_run(exit_code, status: str = "success") -> ToolCall:
    return ToolCall(
        tool_name="code_exec",
        specialist="code",
        arguments={"code": "print(1)"},
        status=status,
        exit_code=exit_code,
    )


def test_a_non_zero_exit_is_an_error_only_under_version_2():
    calls = [code_run(0), code_run(1), code_run(None), code_run(None, status="rate_limited")]
    # version 1: only the rate-limited call; version 2: the exit-1 run too. A run with no
    # recorded exit code keeps the status rule.
    assert metrics.tool_call_scores(calls, task()).error_rate == 0.25
    assert metrics.tool_call_scores(calls, task(), exit_codes=True).error_rate == 0.5
    # the rule is about code runs: another tool's exit_code field means nothing
    odd = ToolCall(
        tool_name="db_query", specialist="a", arguments={}, status="success", exit_code=2
    )
    assert metrics.tool_call_scores([odd], task(), exit_codes=True).error_rate == 0.0


def test_a_non_zero_exit_changes_nothing_but_the_error_rate():
    calls = [code_run(1), call("rag_search"), call("file_write", path="notes.md")]
    v1 = metrics.tool_call_scores(calls, task())
    v2 = metrics.tool_call_scores(calls, task(), exit_codes=True)
    assert (v1.must_call_recall, v1.precision) == (v2.must_call_recall, v2.precision)
    assert (v1.error_rate, v2.error_rate) == (0.0, pytest.approx(1 / 3))


def test_criteria_about_an_unwritten_file_fail():
    subjects = {"c1": "notes.md", "c3": "plan.md"}
    calls = [
        call("file_write", path="notes.md"),
        call("file_write", path="docs/plan.md"),  # another path: plan.md was not written
        call("file_write", status="rate_limited", path="plan.md"),  # refused: not written
    ]
    decided = metrics.unwritten_file_criteria(subjects, calls)
    assert [(d.id, d.passed, d.reasoning) for d in decided] == [
        ("c3", False, metrics.UNWRITTEN_FILE)
    ]
    assert metrics.unwritten_file_criteria(subjects, []) == [
        CriterionVerdict(id="c1", reasoning=metrics.UNWRITTEN_FILE, passed=False),
        CriterionVerdict(id="c3", reasoning=metrics.UNWRITTEN_FILE, passed=False),
    ]
    assert metrics.unwritten_file_criteria({}, []) == []  # no file criteria


def test_overrule_replaces_every_entry_for_a_decided_criterion():
    v = TaskVerdict(
        criteria=[
            CriterionVerdict(id="c1", reasoning="r", passed=True),
            CriterionVerdict(id="c3", reasoning="r", passed=True),
            CriterionVerdict(id="c3", reasoning="again", passed=True),
        ]
    )
    decided = [CriterionVerdict(id="c3", reasoning=metrics.UNWRITTEN_FILE, passed=False)]
    overruled = metrics.overrule(v, decided)
    assert metrics.criteria_passed(overruled, task()) == {"c1": True, "c2": False, "c3": False}
    assert metrics.overrule(v, []) == v
