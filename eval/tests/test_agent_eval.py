"""The agent evaluation end to end, against a fake agent API and a scripted judge.

The fake answers the endpoints the harness uses over ``httpx.MockTransport``: tasks step
through scripted statuses on each poll (an ``awaiting_approval`` step waits until its
approval is resolved), and each has its own invocation log, subtasks and costs. A fake
clock advances only when the harness sleeps, so polling and timeouts take no time.
Every number below is worked out in a comment.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

import agent_eval
from judge_fake import FakeJudge
from llm import LLMError
from schemas import CriterionVerdict, TaskVerdict

CONFIG = {
    "models": {
        "supervisor": "openai:gpt-4o",
        "specialist": "openai:gpt-4o-mini",
        "reviewer": "anthropic:claude-sonnet-5",
        "memory": "openai:gpt-4o-mini",
    },
    "reviewer_provider": "cross",
    "mcp_rag": True,
    "mock_llm": False,
    "run_mode": "inline",
}
DOCS = {"tool": ["rag_search", "rag_ask"]}
TASKS = [
    {
        "id": "t01",
        "request": "Using the team's documentation, write notes.md about the cache",
        "category": "docs",
        "must_call": [DOCS, {"tool": "file_write", "args": {"path": "notes.md"}}],
        "may_call": ["rag_list_sources", "file_read"],
        "forbidden": ["api_call"],
        "rubric": [
            {"id": "c1", "description": "`notes.md` names UV_CACHE_DIR.", "critical": True},
            {"id": "c2", "description": "`notes.md` says how to prune.", "critical": False},
        ],
    },
    {
        "id": "t02",
        "request": "From the demo database, rank the databases by stars",
        "category": "data",
        "must_call": [{"tool": "db_query"}],
        "may_call": ["code_exec"],
        "forbidden": ["api_call"],
        "rubric": [{"id": "c1", "description": "The answer ranks qdrant first.", "critical": True}],
    },
    {
        "id": "t03",
        "request": "From the demo database, sum the stars",
        "category": "data",
        "must_call": [{"tool": "db_query"}],
        "may_call": [],
        "forbidden": ["api_call"],
        "rubric": [{"id": "c1", "description": "The answer gives 58,000.", "critical": True}],
    },
]


def call(tool, specialist, status="success", output=None, **arguments) -> dict:
    return {
        "created_at": "2026-10-13T10:00:00+00:00",
        "subtask_sid": "s1",
        "specialist": specialist,
        "tool_name": tool,
        "status": status,
        "arguments": arguments,
        "output": output,
        "error": None if status == "success" else "boom",
        "latency_ms": 5.0,
        "sensitive": False,
    }


def subtask(sid, specialist, attempts=1, review_score=None, output="done") -> dict:
    return {
        "sid": sid,
        "description": f"subtask {sid}",
        "specialist": specialist,
        "depends_on": [],
        "status": "completed",
        "attempts": attempts,
        "output": output,
        "review_score": review_score,
        "review_feedback": None,
        "error": None,
    }


@dataclass
class Script:
    statuses: list[str]  # one per poll; "awaiting_approval" holds until its approval is resolved
    final_output: str | None = "done"
    calls: list[dict] = field(default_factory=list)
    subtasks: list[dict] = field(default_factory=list)
    llm_usd: float = 0.0
    wall_clock_s: float = 10.0
    escalations: int = 0


def scripts() -> dict[str, Script]:
    t01, t02, t03 = (t["request"] for t in TASKS)
    return {
        t01: Script(
            ["planning", "executing", "completed"],
            final_output="Saved notes.md.",
            calls=[
                call(
                    "rag_ask", "research", output={"content": {"answer": "a", "cost_usd": 0.0004}}
                ),
                call("file_write", "writing", path="notes.md", content="first draft"),
                call("web_search", "research", query="uv cache"),
                call("file_write", "writing", path="notes.md", content="UV_CACHE_DIR moves it."),
            ],
            subtasks=[
                subtask("s1", "research", review_score=4, output="research note"),
                subtask("s2", "writing", attempts=2, review_score=3, output="Saved notes.md."),
            ],
            llm_usd=0.02,
            wall_clock_s=30.0,
        ),
        t02: Script(
            ["planning", "awaiting_approval", "executing", "completed"],
            final_output="qdrant > chroma > weaviate",
            calls=[
                call("db_query", "analysis", sql="SELECT 1"),
                call("code_exec", "analysis", status="failure", code="print(1)"),
            ],
            subtasks=[subtask("s1", "analysis", review_score=5)],
            llm_usd=0.01,
            wall_clock_s=20.0,
            escalations=1,
        ),
        t03: Script(["executing"], final_output=None, llm_usd=0.005, wall_clock_s=601.0),
    }


class FakeAgent:
    """The agent API endpoints the harness calls, driven by one script per request."""

    def __init__(self, scripts: dict[str, Script], config: dict = CONFIG) -> None:
        self.scripts = scripts
        self.config = dict(config)
        self.tasks: dict[str, dict] = {}
        self.approvals: dict[str, dict] = {}
        self.posts: list[dict] = []
        self.resolved: list[str] = []

    def status(self, tid: str) -> str:
        state = self.tasks[tid]
        statuses = state["script"].statuses
        return statuses[min(state["step"], len(statuses) - 1)]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/config":
            return httpx.Response(200, json=self.config)
        if method == "POST" and path == "/tasks":
            body = json.loads(request.content)
            self.posts.append(body)
            tid = f"task-{len(self.tasks) + 1}"
            self.tasks[tid] = {"script": self.scripts[body["request"]], "step": 0, "body": body}
            return httpx.Response(202, json={"task_id": tid, "status": "pending"})
        if path == "/approvals":
            tid = request.url.params["task_id"]
            pending = [
                a
                for a in self.approvals.values()
                if a["task_id"] == tid and a["status"] == "pending"
            ]
            return httpx.Response(200, json={"approvals": pending})
        if method == "POST" and path.startswith("/approvals/"):
            approval = self.approvals[path.split("/")[2]]
            approval["status"] = "resolved"
            self.resolved.append(approval["id"])
            self.tasks[approval["task_id"]]["step"] += 1
            return httpx.Response(200, json={"approval": approval, "resumed": True})
        parts = path.strip("/").split("/")
        if parts[0] == "tasks":
            return httpx.Response(200, json=self.poll(parts[1]))
        tid, what = parts[1], parts[2]
        script = self.tasks[tid]["script"]
        if what == "costs":
            return httpx.Response(
                200,
                json={
                    "task_id": tid,
                    "llm": {"total_calls": 3},
                    "wall_clock_s": script.wall_clock_s,
                    "escalations": script.escalations,
                    "total_usd": script.llm_usd,
                },
            )
        return httpx.Response(200, json={"task_id": tid, "calls": script.calls})

    def poll(self, tid: str) -> dict:
        state, status = self.tasks[tid], self.status(tid)
        if status == "awaiting_approval":
            aid = f"ap-{tid}-{state['step']}"
            self.approvals.setdefault(aid, {"id": aid, "task_id": tid, "status": "pending"})
        elif state["step"] < len(state["script"].statuses) - 1:
            state["step"] += 1
        script = state["script"]
        return {
            "task_id": tid,
            "request": state["body"]["request"],
            "status": status,
            "subtasks": script.subtasks,
            "final_output": script.final_output if status == "completed" else None,
            "error": None,
        }


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def failing_t02(request, final_output, subtask_outputs, criteria) -> TaskVerdict:
    passed = "rank" not in request  # t02's request asks to rank: its criterion fails
    return TaskVerdict(
        criteria=[CriterionVerdict(id=c.id, reasoning="scripted", passed=passed) for c in criteria]
    )


@pytest.fixture
def tasks_file(tmp_path) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in TASKS), encoding="utf-8")
    return path


def run(tmp_path, tasks_file, fake, *extra, judge=None, label="cross", cap="1.00", run_id="run1"):
    clock = Clock()
    args = ["run", "--tasks", str(tasks_file), "--label", label, "--run-id", run_id]
    args += ["--runs", str(tmp_path / "runs"), "--max-cost-usd", cap, *extra]
    client = httpx.Client(base_url="http://agent.test", transport=httpx.MockTransport(fake))
    return agent_eval.main(
        args,
        agent_client=client,
        judge=judge or FakeJudge(task=failing_t02),
        sleep=clock.sleep,
        clock=clock,
    )


def raw(tmp_path, label="cross", run_id="run1") -> dict[str, dict]:
    path = tmp_path / "runs" / run_id / f"agent-{label}.jsonl"
    return {r["id"]: r for r in agent_eval.read_jsonl(path)}


def score(tmp_path, run_id="run1") -> dict:
    out = tmp_path / "report"
    agent_eval.main(
        ["score", "--run-id", run_id, "--runs", str(tmp_path / "runs"), "--out", str(out)]
    )
    return json.loads((out / "agent.json").read_text())


# --- run ----------------------------------------------------------------------------


def test_each_task_runs_under_its_own_user_and_is_recorded(tmp_path, tasks_file):
    fake = FakeAgent(scripts())
    assert run(tmp_path, tasks_file, fake) == 0
    assert [p["user_id"] for p in fake.posts] == [
        "eval-run1-cross-t01",
        "eval-run1-cross-t02",
        "eval-run1-cross-t03",
    ]
    records = raw(tmp_path)
    assert {i: r["outcome"] for i, r in records.items()} == {
        "t01": "completed",
        "t02": "completed",
        "t03": "timeout",
    }
    run_json = json.loads((tmp_path / "runs" / "run1" / "agent-run.json").read_text())
    assert run_json["labels"]["cross"]["config"] == CONFIG
    assert run_json["labels"]["cross"]["tasks"]["file"] == "tasks.jsonl"


def test_pending_approvals_are_approved_and_counted(tmp_path, tasks_file):
    fake = FakeAgent(scripts())
    run(tmp_path, tasks_file, fake)
    assert fake.resolved == ["ap-task-2-1"]  # t02 paused once, at its second status
    assert raw(tmp_path)["t02"]["approvals"] == 1


def test_a_task_that_never_ends_times_out_without_a_judge_call(tmp_path, tasks_file):
    judge = FakeJudge(task=failing_t02)
    run(tmp_path, tasks_file, FakeAgent(scripts()), judge=judge)
    t03 = raw(tmp_path)["t03"]
    assert (t03["outcome"], t03["verdict"]) == ("timeout", None)
    assert [r[1][0] for r in judge.requests] == [TASKS[0]["request"], TASKS[1]["request"]]


def test_approvals_stop_at_the_limit(tmp_path, tasks_file):
    looping = scripts()
    looping[TASKS[1]["request"]].statuses = ["awaiting_approval"] * 7 + ["completed"]
    fake = FakeAgent(looping)
    run(tmp_path, tasks_file, fake)
    t02 = raw(tmp_path)["t02"]
    assert (t02["outcome"], t02["approvals"]) == ("approval_limit", agent_eval.MAX_APPROVALS)
    assert len(fake.resolved) == agent_eval.MAX_APPROVALS


def test_the_judge_sees_the_last_version_of_each_file_and_no_subtask_reports(tmp_path, tasks_file):
    judge = FakeJudge(task=failing_t02)
    run(tmp_path, tasks_file, FakeAgent(scripts()), judge=judge)
    request, final_output, shown, criteria = judge.requests[0][1]
    assert final_output == "Saved notes.md."
    # t01's criteria all name notes.md, so the subtasks' reports stay out
    assert shown == '<file path="notes.md">\nUV_CACHE_DIR moves it.\n</file>'
    # t02's criterion names the answer: no files, no reports
    assert judge.requests[1][1][2] == "(none)"


def test_cost_is_the_llm_total_plus_every_rag_ask(tmp_path, tasks_file):
    calls = [
        call("rag_ask", "research", output={"content": {"cost_usd": 0.0004}}),
        call("rag_ask", "research", output={"cost_usd": 0.0001}),  # content sent as text
        call("rag_ask", "research", status="failure"),
        call("rag_search", "research", output={"content": {"cost_usd": 9.0}}),  # not an ask
    ]
    # 0.0004 + 0.0001; a failed ask has no output, and search doesn't spend
    assert agent_eval.rag_ask_cost(calls) == pytest.approx(0.0005)
    assert agent_eval.task_cost({"costs": {"total_usd": 0.02}, "calls": calls}) == pytest.approx(
        0.0205
    )


def test_the_cap_holds_each_task_to_a_reserve(tmp_path, tasks_file, capsys):
    # t01 starts (0.25 <= 0.26) and spends 0.0204; t02 would need 0.0204 + 0.25 > 0.26
    fake = FakeAgent(scripts())
    assert run(tmp_path, tasks_file, fake, cap="0.26") == 1
    assert len(fake.posts) == 1
    assert "stopped" in capsys.readouterr().out
    # on MOCK_LLM nothing can spend, so a $0 cap runs everything
    smoke = FakeAgent(scripts(), config=CONFIG | {"mock_llm": True})
    assert run(tmp_path, tasks_file, smoke, label="smoke", cap="0", run_id="run2") == 0
    assert len(smoke.posts) == 3


def test_resume_skips_recorded_tasks_and_judges_without_running_again(tmp_path, tasks_file):
    def unavailable(*args):
        raise LLMError("judge down")

    fake = FakeAgent(scripts())
    assert run(tmp_path, tasks_file, fake, judge=FakeJudge(task=unavailable)) == 1
    assert raw(tmp_path)["t01"]["verdict"] is None
    posts = len(fake.posts)
    assert run(tmp_path, tasks_file, fake) == 0
    assert len(fake.posts) == posts  # nothing ran again
    assert raw(tmp_path)["t01"]["verdict"]["criteria"][0]["passed"] is True


@pytest.mark.parametrize(
    ("label", "config", "message"),
    [
        ("same", CONFIG, "needs REVIEWER_PROVIDER=same"),
        ("smoke", CONFIG, "MOCK_LLM=1"),
        ("cross", CONFIG | {"mock_llm": True}, "runs on MOCK_LLM"),
    ],
)
def test_a_label_must_match_the_agent(tmp_path, tasks_file, label, config, message):
    with pytest.raises(SystemExit, match=message):
        run(tmp_path, tasks_file, FakeAgent(scripts(), config=config), label=label)


def test_a_judge_from_the_producers_provider_needs_a_flag(tmp_path, tasks_file):
    judge = FakeJudge(task=failing_t02)
    judge.provider = "openai"
    with pytest.raises(SystemExit, match="allow-same-provider-judge"):
        run(tmp_path, tasks_file, FakeAgent(scripts()), judge=judge)
    assert (
        run(tmp_path, tasks_file, FakeAgent(scripts()), "--allow-same-provider-judge", judge=judge)
        == 0
    )


def test_tasks_that_need_rag_tools_need_an_agent_that_has_them(tmp_path, tasks_file):
    with pytest.raises(SystemExit, match="MCP_RAG_URL"):
        run(tmp_path, tasks_file, FakeAgent(scripts(), config=CONFIG | {"mcp_rag": False}))


def test_a_run_needs_a_cap(tmp_path, tasks_file):
    with pytest.raises(SystemExit) as exc:
        agent_eval.main(["run", "--tasks", str(tasks_file), "--label", "cross", "--run-id", "r"])
    assert exc.value.code == 2


# --- score --------------------------------------------------------------------------


def test_scores_per_task_and_label(tmp_path, tasks_file):
    run(tmp_path, tasks_file, FakeAgent(scripts()))
    scores = score(tmp_path)["scores"]
    t01, t02, t03 = scores["tasks"]["cross"]
    # t01: rag_ask meets the docs entry and the last file_write meets notes.md (2/2); 3 of 4
    # calls went to named tools (web_search isn't); no errors; $0.02 + $0.0004 of rag_ask
    assert (t01["success"], t01["must_call_recall"], t01["precision"], t01["error_rate"]) == (
        True,
        1.0,
        0.75,
        0.0,
    )
    assert t01["cost_usd"] == pytest.approx(0.0204)
    # t02: completed, but its critical criterion failed; 1 of 2 calls failed
    assert (t02["success"], t02["must_call_recall"], t02["precision"], t02["error_rate"]) == (
        False,
        1.0,
        1.0,
        0.5,
    )
    # t03 timed out: it fails without a verdict; no calls, so recall 0/1 and no ratios
    assert (t03["success"], t03["must_call_recall"], t03["precision"]) == (False, 0.0, None)

    cross = scores["labels"]["cross"]
    assert (cross["success"]["k"], cross["success"]["n"]) == (1, 3)
    assert (cross["completed"]["k"], cross["completed"]["n"]) == (2, 3)
    assert cross["must_call_recall"] == pytest.approx(2 / 3)  # (1 + 1 + 0) / 3
    assert cross["precision"] == pytest.approx(0.875)  # (0.75 + 1.0) / 2: t03 has none
    assert cross["error_rate"] == pytest.approx(0.25)  # (0 + 0.5) / 2
    assert cross["calls_per_task"] == pytest.approx(2.0)  # (4 + 2 + 0) / 3
    assert cross["cost_total"] == pytest.approx(0.0354)  # 0.0204 + 0.01 + 0.005
    assert cross["cost_per_task"] == pytest.approx(0.0118)  # 0.0354 / 3
    assert cross["cost_per_completed_task"] == pytest.approx(0.0152)  # (0.0204 + 0.01) / 2
    assert cross["wall_clock_per_task"] == pytest.approx(217.0)  # (30 + 20 + 601) / 3
    assert cross["wall_clock_per_completed_task"] == pytest.approx(25.0)  # (30 + 20) / 2
    assert (cross["escalations"], cross["tasks_escalated"], cross["approvals"]) == (1, 1, 1)
    assert cross["review_score_mean"] == pytest.approx(4.0)  # (4 + 3 + 5) / 3
    assert cross["rework_rate"] == pytest.approx(1 / 3)  # t01's s2 of 3 subtasks
    assert (cross["non_critical"]["k"], cross["non_critical"]["n"]) == (1, 1)  # t01's c2
    assert cross["by_category"]["docs"]["k"] == 1 and cross["by_category"]["data"]["k"] == 0
    assert cross["outcomes"] == {"completed": 2, "timeout": 1}


def test_cross_vs_same_pairs_the_tasks(tmp_path, tasks_file):
    run(tmp_path, tasks_file, FakeAgent(scripts()))

    def all_pass(request, final_output, subtask_outputs, criteria) -> TaskVerdict:
        return TaskVerdict(
            criteria=[CriterionVerdict(id=c.id, reasoning="", passed=True) for c in criteria]
        )

    same = FakeAgent(scripts(), config=CONFIG | {"reviewer_provider": "same"})
    run(tmp_path, tasks_file, same, label="same", judge=FakeJudge(task=all_pass))
    report = score(tmp_path)
    # t01 passes under both, t02 only under same, t03 times out under both
    assert report["scores"]["cross_vs_same"] == {
        "n": 3,
        "both": 1,
        "neither": 1,
        "only_first": [],
        "only_second": ["t02"],
    }
    md = (tmp_path / "report" / "agent.md").read_text()
    assert "|  | cross | same |" in md and "## Cross vs same reviewer" in md
    for row in (
        "task success",
        "tool-call must-call recall",
        "cost per task",
        "wall-clock per task",
    ):
        assert f"| {row} |" in md


def test_records_cut_long_arguments_and_keep_the_task(tmp_path, tasks_file):
    long = scripts()
    long[TASKS[0]["request"]].calls[3]["arguments"]["content"] = "x" * 500
    run(tmp_path, tasks_file, FakeAgent(long))
    records = agent_eval.build_records(tmp_path / "runs" / "run1")
    header, t01 = records[0], records[1]
    assert header["labels"]["cross"]["unjudged"] == 0
    assert t01["task"]["id"] == "t01" and t01["rag_usd"] == pytest.approx(0.0004)
    assert t01["calls"][3]["arguments"]["content"] == "x" * 200 + "… (500 chars)"
    assert t01["calls"][3]["arguments"]["path"] == "notes.md"


def test_score_from_records_reproduces_score_from_the_run(tmp_path, tasks_file):
    run(tmp_path, tasks_file, FakeAgent(scripts()))
    score(tmp_path)
    first = tmp_path / "report"
    again = tmp_path / "again"
    agent_eval.main(["score", "--records", str(first / "agent-records.jsonl"), "--out", str(again)])
    for name in ("agent.md", "agent.json"):
        assert (again / name).read_bytes() == (first / name).read_bytes()
