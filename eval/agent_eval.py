"""Evaluate the agent service over HTTP: run tasks once, score them for free.

    python -m agent_eval run --tasks tasks/agent_tasks.jsonl --label cross --run-id my-run \\
        --max-cost-usd 1.50
    python -m agent_eval score --run-id my-run                        # -> runs/my-run/report/
    python -m agent_eval score --run-id my-run --out reports          # publish
    python -m agent_eval score --records reports/agent-records.jsonl  # re-score in place

``run`` is the only step that talks to the agent or a judge. It records ``GET /config``
per label in ``runs/<id>/agent-run.json`` and refuses a label the agent doesn't match:
``cross`` and ``same`` name the reviewer policy and need real models, ``smoke`` needs
MOCK_LLM. Tasks run one at a time, each under its own user id, so no task's memories
reach another's plan: ``POST /tasks``, then ``GET /tasks/{id}`` every 2 s, approving each
pending approval (counted, at most 5 per task) until the task completes, fails or runs
out of time (600 s). Then the task's costs, invocation log and bundle are fetched, and a
completed task goes to the rubric judge, which sees the final output, the files the run
wrote and, only when a criterion needs them, the subtasks' outputs. ``--max-cost-usd``
caps the agent's spend and the judge's together: a task starts only if its reserve
(``--task-reserve-usd``, 0 on MOCK_LLM) fits, and its real cost, the LLM total plus every
``rag_ask``'s ``cost_usd``, is recorded when it ends. Runs resume: a recorded task is
skipped, and a completed task without a verdict is judged again without running again.

``score`` never calls anything. It turns the run into compact records (each holding the
task it ran, so the records file alone is enough to re-score), computes every number
from them with ``metrics``, and writes the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

import metrics
import report
from budget import Budget, BudgetExceeded
from judge import SubtaskOutput, render_task_outputs
from llm import LLMError, load_env
from schemas import AgentTask, CriterionVerdict, TaskVerdict, ToolCall

LABELS = ("cross", "same", "smoke")
POLL_S = 2.0
TIMEOUT_S = 600.0
# An escalated subtask that is approved runs again, and may escalate again; past this
# many approvals the harness stops approving, and the task ends as "approval_limit".
MAX_APPROVALS = 5
APPROVAL_NOTE = "approved by the evaluation harness"
TASK_RESERVE_USD = 0.25  # what a task may spend before the cap is checked again
TERMINAL = ("completed", "failed", "rejected", "escalated")
# Tool-call arguments go into the records cut to this many characters (file contents
# and code are long); must_call args are short, so scoring never needs more.
ARG_CHARS = 200
MAX_USER_ID = 64  # the agent's user_id columns


def load_tasks(path: Path) -> list[AgentTask]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [AgentTask.model_validate_json(line) for line in lines if line.strip()]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append(f, record: dict) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def user_id(run_id: str, label: str, task_id: str) -> str:
    """A fresh user per run, label and task: long-term memory is per user, so one task's
    memories never shape another's plan, nor one reviewer arm's the other's."""
    return f"eval-{run_id}-{label}-{task_id}"


# --- run: the label's snapshot ------------------------------------------------------


def check_label(label: str, config: dict) -> None:
    if label == "smoke":
        if not config["mock_llm"]:
            raise SystemExit(
                "label smoke replays the agent's fixtures: run the agent with MOCK_LLM=1"
            )
    elif config["mock_llm"]:
        raise SystemExit(f"label {label} measures real models, but the agent runs on MOCK_LLM")
    elif config["reviewer_provider"] != label:
        raise SystemExit(
            f"label {label} needs REVIEWER_PROVIDER={label}; "
            f"the agent runs {config['reviewer_provider']}"
        )


def open_label(agent: httpx.Client, run_dir: Path, label: str, tasks_path: Path) -> dict:
    """The agent's config for ``label``, checked against the label. The first run of a
    label records it with the tasks file's hash; resuming the label needs the same config
    and the same tasks file."""
    config = agent.get("/config").raise_for_status().json()
    check_label(label, config)
    tasks = {
        "file": tasks_path.name,
        "sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "agent-run.json"
    run = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    run = run or {"run_id": run_dir.name, "labels": {}}
    if label in run["labels"]:
        held = run["labels"][label]
        if held["config"] != config:
            raise SystemExit(
                f"{run_dir.name}: the agent's /config changed since label {label} began"
            )
        if held["tasks"] != tasks:
            raise SystemExit(f"{run_dir.name}: {tasks_path} changed since label {label} began")
        return config
    started = datetime.now(UTC).isoformat(timespec="seconds")
    run["labels"][label] = {"started_at": started, "config": config, "tasks": tasks}
    path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return config


def provider(spec: str) -> str:
    """``openai`` from ``openai:gpt-4o``."""
    return spec.split(":", 1)[0]


def check_judge(judge: Any, config: dict, allow_same_provider: bool) -> None:
    producers = {provider(config["models"][a]) for a in ("supervisor", "specialist")}
    if judge.provider in producers and not allow_same_provider:
        raise SystemExit(
            f"the judge ({judge.provider}:{judge.model}) comes from a provider that writes the "
            f"agent's outputs ({', '.join(sorted(producers))}); pass "
            "--allow-same-provider-judge to run it anyway"
        )


# --- run: one task ------------------------------------------------------------------


def run_task(
    agent: httpx.Client,
    task: AgentTask,
    user: str,
    *,
    poll_s: float,
    timeout_s: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> dict:
    """Start the task, approve what it asks for, wait for it to end, fetch what it did."""
    tid = agent.post("/tasks", json={"request": task.request, "user_id": user})
    tid = tid.raise_for_status().json()["task_id"]
    start, approvals, outcome = clock(), 0, None
    while outcome is None:
        status = agent.get(f"/tasks/{tid}").raise_for_status().json()["status"]
        if status in TERMINAL:
            outcome = status
            break
        if status == "awaiting_approval":
            pending = agent.get("/approvals", params={"status": "pending", "task_id": tid})
            for approval in pending.raise_for_status().json()["approvals"]:
                if approvals >= MAX_APPROVALS:
                    outcome = "approval_limit"
                    break
                resolved = agent.post(
                    f"/approvals/{approval['id']}/resolve",
                    json={"action": "approve", "notes": APPROVAL_NOTE},
                )
                if resolved.status_code != 409:  # 409: someone else resolved it meanwhile
                    resolved.raise_for_status()
                    approvals += 1
        if outcome is None and clock() - start >= timeout_s:
            outcome = "timeout"
        if outcome is None:
            sleep(poll_s)
    return {
        "agent_task_id": tid,
        "outcome": outcome,
        "approvals": approvals,
        "bundle": agent.get(f"/tasks/{tid}").raise_for_status().json(),
        "costs": agent.get(f"/traces/{tid}/costs").raise_for_status().json(),
        "calls": agent.get(f"/traces/{tid}/tools").raise_for_status().json()["calls"],
    }


def rag_ask_cost(calls: Sequence[dict]) -> float:
    """What the RAG service spent answering the run's ``rag_ask`` calls: each successful
    call's output carries it (the agent's LLM total doesn't include it)."""
    total = 0.0
    for call in calls:
        if call["tool_name"] != "rag_ask" or call["status"] != "success":
            continue
        output = call.get("output") or {}
        content = output.get("content", output) if isinstance(output, dict) else {}
        if isinstance(content, dict) and isinstance(content.get("cost_usd"), int | float):
            total += content["cost_usd"]
    return total


def task_cost(raw: dict) -> float:
    return raw["costs"]["total_usd"] + rag_ask_cost(raw["calls"])


def files_written(calls: Sequence[dict]) -> dict[str, str]:
    """Each file's content as the run last wrote it: a reworked file is written again."""
    files: dict[str, str] = {}
    for call in calls:
        args = call.get("arguments") or {}
        if call["tool_name"] == "file_write" and call["status"] == "success" and "path" in args:
            files[args["path"]] = str(args.get("content", ""))
    return files


def subtask_outputs(bundle: dict) -> list[SubtaskOutput]:
    return [
        SubtaskOutput(s["sid"], s["specialist"], s["description"], s["output"])
        for s in bundle.get("subtasks", [])
        if s.get("output")
    ]


def needs_verdict(record: dict) -> bool:
    return record["outcome"] == "completed" and not record.get("verdict")


def _judge(judge: Any, task: AgentTask, record: dict) -> tuple[dict, BudgetExceeded | None]:
    """The rubric verdict of a completed task; a budget stop leaves it for a resume."""
    record = dict(record)
    if not needs_verdict(record):
        return record, None
    shown = render_task_outputs(
        task, files_written(record["calls"]), subtask_outputs(record["bundle"])
    )
    before = judge.budget.spent_usd
    try:
        verdict = judge.task(
            task.request, record["bundle"]["final_output"] or "", shown, task.rubric
        )
    except BudgetExceeded as exc:
        return record, exc
    except LLMError as exc:
        record["judge_error"] = _error(exc)
        return record, None
    record.pop("judge_error", None)
    record["verdict"] = {
        "criteria": [c.model_dump() for c in verdict.criteria],
        "provider": judge.provider,
        "model": judge.model,
        "prompt_version": judge.prompt_version,
        "cost_usd": judge.budget.spent_usd - before,  # 0 for a cached verdict
    }
    return record, None


def run_label(
    agent: httpx.Client,
    judge: Any,
    budget: Budget,
    tasks: Sequence[AgentTask],
    label: str,
    run_dir: Path,
    *,
    reserve: float,
    notional: bool = False,
    poll_s: float = POLL_S,
    timeout_s: float = TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run and judge every task under ``label``; returns 1 on a budget stop or any error.

    ``notional``: the agent runs on MOCK_LLM, which prices its scripted replies like the
    models they stand in for. Nothing is spent, so those costs aren't held against the cap.
    """
    out = run_dir / f"agent-{label}.jsonl"
    previous = {r["id"]: r for r in read_jsonl(out)}
    counts: Counter[str] = Counter()
    with out.open("a", encoding="utf-8") as f:
        try:
            for task in tasks:
                record = previous.get(task.id)
                if record and "error" not in record:
                    if not needs_verdict(record):
                        counts["recorded"] += 1
                        continue
                    counts["re-judged"] += 1
                else:
                    budget.check(reserve)
                    user = user_id(run_dir.name, label, task.id)
                    try:
                        raw = run_task(
                            agent,
                            task,
                            user,
                            poll_s=poll_s,
                            timeout_s=timeout_s,
                            sleep=sleep,
                            clock=clock,
                        )
                    except (httpx.HTTPError, KeyError, ValueError) as exc:
                        _append(f, {"id": task.id, "label": label, "error": _error(exc)})
                        counts["errors"] += 1
                        continue
                    if not notional:
                        budget.record(task_cost(raw))
                    record = {"id": task.id, "label": label, "user_id": user}
                    record |= {"task": task.model_dump(), **raw, "verdict": None}
                    counts[raw["outcome"]] += 1
                record, stop = _judge(judge, task, record)
                if "judge_error" in record:
                    counts["judge errors"] += 1
                _append(f, record)
                if stop:
                    raise stop
        except BudgetExceeded as exc:
            print(f"stopped: {exc}")
            counts["stopped"] += 1
    summary = ", ".join(f"{n} {what}" for what, n in sorted(counts.items()))
    print(f"{label}: {summary or 'nothing to do'} -> {out} · {budget.summary()}")
    return 1 if counts["stopped"] or counts["errors"] or counts["judge errors"] else 0


# --- records ------------------------------------------------------------------------


def _cut(value: Any) -> Any:
    if isinstance(value, str) and len(value) > ARG_CHARS:
        return value[:ARG_CHARS] + f"… ({len(value)} chars)"
    if isinstance(value, dict):
        return {k: _cut(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cut(v) for v in value]
    return value


def _judge_id(verdict: dict) -> str:
    return f"{verdict['provider']}:{verdict['model']} prompts {verdict['prompt_version']}"


def _record(label: str, raw: dict) -> dict:
    task, bundle, costs, verdict = raw["task"], raw["bundle"], raw["costs"], raw["verdict"]
    return {
        "record": "agent",
        "label": label,
        "id": raw["id"],
        "category": task["category"],
        "task": task,
        "agent_task_id": raw["agent_task_id"],
        "outcome": raw["outcome"],
        "error": (bundle.get("error") or "")[:300] or None,
        "calls": [
            {k: _cut(c[k]) for k in ("tool_name", "specialist", "status", "arguments")}
            for c in raw["calls"]
        ],
        "subtasks": [
            {k: s[k] for k in ("sid", "specialist", "status", "attempts", "review_score")}
            for s in bundle.get("subtasks", [])
        ],
        "llm_usd": costs["total_usd"],
        "rag_usd": rag_ask_cost(raw["calls"]),
        "llm_calls": costs["llm"]["total_calls"],
        "wall_clock_s": costs["wall_clock_s"],
        "escalations": costs["escalations"],
        "approvals": raw["approvals"],
        "criteria": (
            [{"id": c["id"], "passed": c["passed"]} for c in verdict["criteria"]]
            if verdict
            else None
        ),
        "judge": _judge_id(verdict) if verdict else None,
        "judge_cost_usd": verdict["cost_usd"] if verdict else 0.0,
    }


def build_records(run_dir: Path) -> list[dict]:
    """The run as a header and one compact record per label and task."""
    run = json.loads((run_dir / "agent-run.json").read_text(encoding="utf-8"))
    header: dict = {"record": "run", "run_id": run["run_id"], "labels": {}}
    records: list[dict] = []
    for label, snapshot in run["labels"].items():
        raw = {r["id"]: r for r in read_jsonl(run_dir / f"agent-{label}.jsonl")}
        rows = [_record(label, r) for r in raw.values() if "error" not in r]
        rows.sort(key=lambda r: r["id"])
        header["labels"][label] = {
            "collected": snapshot["started_at"][:10],
            "config": snapshot["config"],
            "tasks": snapshot["tasks"],
            "missing": sum("error" in r for r in raw.values()),
            "unjudged": sum(r["outcome"] == "completed" and not r["criteria"] for r in rows),
            "judges": sorted({r["judge"] for r in rows if r["judge"]}),
        }
        records += rows
    return [header, *records]


# --- score --------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _rate(flags: Sequence[bool]) -> dict:
    k, n = sum(flags), len(flags)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci": metrics.wilson_interval(k, n)}


def task_scores(record: dict) -> dict:
    """One task run's numbers. A task succeeds only if it completed and its critical
    criteria passed; a completed task the judge never saw has success None."""
    task = AgentTask.model_validate(record["task"])
    calls = [ToolCall.model_validate(c) for c in record["calls"]]
    tool = metrics.tool_call_scores(calls, task)
    verdict = None
    if record["criteria"] is not None:
        verdict = TaskVerdict(
            criteria=[
                CriterionVerdict(id=c["id"], reasoning="", passed=c["passed"])
                for c in record["criteria"]
            ]
        )
    completed = record["outcome"] == "completed"
    if not completed:
        success = False
    else:
        success = None if verdict is None else metrics.task_success(verdict, task)
    passed = metrics.criteria_passed(verdict, task) if verdict is not None else {}
    non_critical = [passed[c.id] for c in task.rubric if not c.critical] if passed else []
    return {
        "id": record["id"],
        "agent_task_id": record["agent_task_id"],
        "category": record["category"],
        "outcome": record["outcome"],
        "completed": completed,
        "success": success,
        "must_call_recall": tool.must_call_recall,
        "precision": tool.precision,
        "error_rate": tool.error_rate,
        "forbidden_calls": tool.forbidden_calls,
        "n_calls": tool.n_calls,
        "cost_usd": record["llm_usd"] + record["rag_usd"],
        "wall_clock_s": record["wall_clock_s"],
        "escalations": record["escalations"],
        "approvals": record["approvals"],
        "review_scores": [
            s["review_score"] for s in record["subtasks"] if s["review_score"] is not None
        ],
        "subtasks": len(record["subtasks"]),
        "reworked": sum(s["attempts"] > 1 for s in record["subtasks"]),
        "non_critical": non_critical,
        "judge_cost_usd": record["judge_cost_usd"],
    }


def _defined(per: list[dict], key: str) -> list[float]:
    return [p[key] for p in per if p[key] is not None]


def label_scores(per: list[dict]) -> dict:
    """A label's aggregates. Tool-call ratios are means over tasks (each task counts once);
    "per completed task" means are over the tasks that completed."""
    done = [p for p in per if p["completed"]]
    reviewed = [s for p in per for s in p["review_scores"]]
    subtasks = sum(p["subtasks"] for p in per)
    categories = sorted({p["category"] for p in per})
    return {
        "n": len(per),
        "outcomes": dict(Counter(p["outcome"] for p in per)),
        "completed": _rate([p["completed"] for p in per]),
        "success": _rate(_defined(per, "success")),
        "unjudged": sum(p["success"] is None for p in per),
        "by_category": {
            c: _rate([p["success"] for p in per if p["category"] == c and p["success"] is not None])
            for c in categories
        },
        "must_call_recall": _mean(_defined(per, "must_call_recall")),
        "precision": _mean(_defined(per, "precision")),
        "error_rate": _mean(_defined(per, "error_rate")),
        "calls_per_task": _mean([p["n_calls"] for p in per]),
        "forbidden_calls": sum(p["forbidden_calls"] for p in per),
        "tasks_with_forbidden_calls": sum(p["forbidden_calls"] > 0 for p in per),
        "cost_total": sum(p["cost_usd"] for p in per),
        "cost_per_task": _mean([p["cost_usd"] for p in per]),
        "cost_per_completed_task": _mean([p["cost_usd"] for p in done]),
        "wall_clock_per_task": _mean([p["wall_clock_s"] for p in per]),
        "wall_clock_per_completed_task": _mean([p["wall_clock_s"] for p in done]),
        "escalations": sum(p["escalations"] for p in per),
        "tasks_escalated": sum(p["escalations"] > 0 for p in per),
        "approvals": sum(p["approvals"] for p in per),
        "review_score_mean": _mean(reviewed),
        "reviewed_subtasks": len(reviewed),
        "rework_rate": sum(p["reworked"] for p in per) / subtasks if subtasks else None,
        "subtasks": subtasks,
        "non_critical": _rate([x for p in per for x in p["non_critical"]]),
        "judge_cost": sum(p["judge_cost_usd"] for p in per),
    }


def paired(a: list[dict], b: list[dict]) -> dict:
    """Success on the tasks both labels judged: where they agree and where they don't."""
    left = {p["id"]: p["success"] for p in a if p["success"] is not None}
    right = {p["id"]: p["success"] for p in b if p["success"] is not None}
    both = sorted(left.keys() & right.keys())
    return {
        "n": len(both),
        "both": sum(left[i] and right[i] for i in both),
        "only_first": sorted(i for i in both if left[i] and not right[i]),
        "only_second": sorted(i for i in both if right[i] and not left[i]),
        "neither": sum(not left[i] and not right[i] for i in both),
    }


def score_records(records: list[dict]) -> dict:
    header, rows = records[0], records[1:]
    per = {
        label: [task_scores(r) for r in rows if r["label"] == label] for label in header["labels"]
    }
    scores: dict = {
        "labels": {label: label_scores(p) for label, p in per.items()},
        "tasks": per,
        "cross_vs_same": None,
    }
    if "cross" in per and "same" in per:
        scores["cross_vs_same"] = paired(per["cross"], per["same"])
    return scores


# --- CLI ----------------------------------------------------------------------------


def _make_judge(kind: str, budget: Budget) -> Any:
    if kind == "fake":
        from judge_fake import FakeJudge

        return FakeJudge()
    from judge import Judge

    return Judge.from_env(budget=budget)


def main(
    argv: Sequence[str] | None = None,
    *,
    agent_client: httpx.Client | None = None,
    judge: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent_eval", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run tasks through the agent into runs/<id>/")
    run.add_argument("--tasks", type=Path, default=Path("tasks/agent_tasks.jsonl"))
    run.add_argument("--label", choices=LABELS, required=True)
    run.add_argument("--agent", metavar="URL", help="the agent API (default: $AGENT_URL)")
    run.add_argument("--run-id", required=True)
    run.add_argument("--runs", type=Path, default=Path("runs"))
    run.add_argument("--ids", help="only these task ids, comma-separated (a pilot)")
    run.add_argument(
        "--max-cost-usd", type=float, required=True, help="the agent's and judge's spend"
    )
    run.add_argument(
        "--task-reserve-usd",
        type=float,
        default=TASK_RESERVE_USD,
        help=f"held against the cap before each task (default {TASK_RESERVE_USD}; 0 on MOCK_LLM)",
    )
    run.add_argument(
        "--judge",
        choices=["env", "fake"],
        default="env",
        help="env: JUDGE_PROVIDER/JUDGE_MODEL (default claude-haiku-4-5); fake: scripted, $0",
    )
    run.add_argument("--allow-same-provider-judge", action="store_true")
    score = sub.add_parser("score", help="score a run or a records file; never calls anything")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id")
    source.add_argument("--records", type=Path, help="an agent-records.jsonl written by score")
    score.add_argument("--runs", type=Path, default=Path("runs"))
    score.add_argument(
        "--out", type=Path, help="report directory (default: runs/<id>/report, or the records' own)"
    )
    args = parser.parse_args(argv)

    if args.command == "run":
        load_env(Path(".env"))
        url = args.agent or os.environ.get("AGENT_URL")
        if not url and agent_client is None:
            parser.error("--agent or AGENT_URL is required")
        tasks = load_tasks(args.tasks)
        if args.ids:
            wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
            if unknown := wanted - {t.id for t in tasks}:
                parser.error(f"not in {args.tasks}: {sorted(unknown)}")
            tasks = [t for t in tasks if t.id in wanted]
        users = {t.id: user_id(args.run_id, args.label, t.id) for t in tasks}
        if too_long := sorted(i for i, u in users.items() if len(u) > MAX_USER_ID):
            parser.error(f"--run-id is too long: user ids for {too_long} pass {MAX_USER_ID} chars")
        run_dir = args.runs / args.run_id
        budget = Budget(args.max_cost_usd)
        with agent_client or httpx.Client(base_url=url, timeout=30) as agent:
            config = open_label(agent, run_dir, args.label, args.tasks)
            needs_rag = any(
                tool.startswith("rag_") for t in tasks for e in t.must_call for tool in e.tools
            )
            if needs_rag and not config["mcp_rag"]:
                raise SystemExit("the tasks need the rag_* tools, but the agent has no MCP_RAG_URL")
            judge = judge or _make_judge(args.judge, budget)
            check_judge(judge, config, args.allow_same_provider_judge)
            mock = config["mock_llm"]
            reserve = 0.0 if mock else args.task_reserve_usd
            return run_label(
                agent,
                judge,
                budget,
                tasks,
                args.label,
                run_dir,
                reserve=reserve,
                notional=mock,
                sleep=sleep,
                clock=clock,
            )

    if args.records:
        records = read_jsonl(args.records)
        if not records or records[0].get("record") != "run" or "labels" not in records[0]:
            parser.error(f"{args.records} isn't an agent records file written by score")
        out = args.out or args.records.parent
    else:
        records = build_records(args.runs / args.run_id)
        out = args.out or args.runs / args.run_id / "report"
    written = report.write_agent(
        records, score_records(records), out, with_records=not args.records
    )
    print("\n".join(f"wrote {path}" for path in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
