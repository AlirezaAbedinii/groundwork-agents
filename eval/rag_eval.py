"""Evaluate the RAG service over HTTP: collect raw runs once, score them for free.

    python -m rag_eval collect --stage retrieval --rag http://localhost:8000 --run-id my-run
    python -m rag_eval collect --stage answers --rag http://localhost:8000 --run-id my-run \\
        --max-cost-usd 1.00
    python -m rag_eval score --run-id my-run                    # -> runs/my-run/report/
    python -m rag_eval score --run-id my-run --out reports      # publish
    python -m rag_eval score --records reports/rag-records.jsonl  # re-score in place

``collect`` is the only step that talks to the service or a judge. It records
``GET /v1/config`` in ``runs/<id>/run.json`` (not committed) and refuses to continue a
run against a service whose config changed. The retrieval stage sends every golden
question in each mode to ``POST /v1/search`` with ``top_k`` 10. The answer stage sends
it to ``POST /v1/ask`` with ``top_k`` 5 in both modes, so the generator sees the same
number of contexts either way, then asks the judge for what isn't decided already:
correctness for an answerable question that was answered, faithfulness for every
answer. A refused answerable question fails correctness and a no_answer question is
correct iff refused, without a judge call. ``--max-cost-usd`` caps the RAG's spend and
the judge's together and is checked before every call; the judge must come from
another provider than the generator unless ``--allow-same-provider-judge``. Both stages
resume: what was recorded without an error is skipped, and an answer that lacks a
verdict is judged again without being asked again.

``score`` never calls anything. It joins the run with the golden set (and the human
grades, if any) into compact records, computes every number from them with
``metrics``, and writes the report. The records file is enough to re-score, so the
published numbers can be recomputed without the run, the golden set or any key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

import metrics
import report
from budget import Budget, BudgetExceeded
from llm import LLMError, cost_usd, load_env
from schemas import (
    PRF,
    ClaimVerdict,
    Context,
    CorrectnessVerdict,
    FaithfulnessVerdict,
    GoldenQuestion,
    SweepPoint,
)
from textnorm import covered_items

TOP_K = 10
TOP_K_ASK = 5
MODES = ("dense", "hybrid")
# MCP `search` cuts each hit to its first SNIPPET_CHARS characters; the snippet-size
# question is 3 hits of 300 chars against 5 hits of 150 (mcp/tool_models.py).
SNIPPETS = ((3, 300), (5, 150))
RECALL_KS = (1, 3, 5, 10)
# What one /v1/ask can cost at most, reserved against the cap before the call: a
# generation with five contexts, and one citation check per context when the pipeline
# verifies citations. The real cost (the response's cost_usd) is what gets recorded.
ASK_INPUT_TOKENS, ASK_OUTPUT_TOKENS = 3000, 1000
JUDGED = ("correctness", "faithfulness")


# --- shared -------------------------------------------------------------------------


def load_golden(path: Path) -> list[GoldenQuestion]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldenQuestion.model_validate_json(line) for line in lines if line.strip()]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest(path: Path) -> dict[tuple[str, str], dict]:
    """The last record per question and mode: later lines supersede earlier ones."""
    return {(r["id"], r["mode"]): r for r in read_jsonl(path)}


def answer_sha(answer: str) -> str:
    """Identifies the answer a human grade was given for."""
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16]


def _open_run(rag: httpx.Client, run_dir: Path) -> dict:
    """The service's config, recorded on the run's first collect and checked on every later one."""
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / "run.json"
    config = rag.get("/v1/config").raise_for_status().json()
    if run_path.is_file():
        if json.loads(run_path.read_text(encoding="utf-8"))["config"] != config:
            raise SystemExit(
                f"{run_dir.name}: the service's /v1/config changed since this run started"
            )
    else:
        started = datetime.now(UTC).isoformat(timespec="seconds")
        run = {"run_id": run_dir.name, "started_at": started, "config": config}
        run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return config


def _append(f, record: dict) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


# --- collect: retrieval -------------------------------------------------------------


def _search(rag: httpx.Client, question: str, mode: str) -> dict:
    response = rag.post("/v1/search", json={"query": question, "mode": mode, "top_k": TOP_K})
    response.raise_for_status()
    body = response.json()
    hits = [
        {k: hit.get(k) for k in ("chunk_id", "source_file", "section_heading", "score", "text")}
        for hit in body["hits"]
    ]
    return {"hits": hits, "timings_ms": body.get("timings_ms", {})}


def collect_retrieval(
    rag: httpx.Client, questions: Sequence[GoldenQuestion], modes: Sequence[str], run_dir: Path
) -> int:
    """Search every question in every mode into ``run_dir``; returns the number of errors."""
    _open_run(rag, run_dir)
    out = run_dir / "retrieval.jsonl"
    done = {key for key, r in latest(out).items() if "error" not in r}
    searched = errors = 0
    with out.open("a", encoding="utf-8") as f:
        for q in questions:
            for mode in modes:
                if (q.id, mode) in done:
                    continue
                try:
                    record = {"id": q.id, "mode": mode, **_search(rag, q.question, mode)}
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    record = {"id": q.id, "mode": mode, "error": _error(exc)}
                    errors += 1
                _append(f, record)
                searched += 1
    skipped = len(questions) * len(modes) - searched
    print(f"{searched} searches ({errors} errors), {skipped} already recorded -> {out}")
    return errors


# --- collect: answers ---------------------------------------------------------------


def ask_worst_case_usd(config: dict) -> float:
    calls = 1 + (TOP_K_ASK if config.get("citation_verification") else 0)
    return calls * cost_usd(config["generation_model"], ASK_INPUT_TOKENS, ASK_OUTPUT_TOKENS)


def _ask(rag: httpx.Client, question: str, mode: str) -> dict:
    response = rag.post("/v1/ask", json={"question": question, "mode": mode, "top_k": TOP_K_ASK})
    response.raise_for_status()
    body = response.json()
    return {
        "answer": body["answer"],
        "refused": body["refused"],
        "refused_by": body.get("refused_by"),
        "retrieval_confidence": body.get("retrieval_confidence"),
        "contexts": body.get("contexts", []),
        "citations": body.get("citations", []),
        "cost_usd": body.get("cost_usd", 0.0),
        "total_ms": body.get("timings_ms", {}).get("total_ms"),
    }


def needs_verdicts(q: GoldenQuestion, record: dict) -> list[str]:
    """The verdicts an answer still lacks; a refusal needs none."""
    if record["refused"]:
        return []
    needed = []
    if q.category != "no_answer" and not record.get("correctness"):
        needed.append("correctness")
    if record["contexts"] and not record.get("faithfulness"):
        needed.append("faithfulness")
    return needed


def _judge(judge: Any, q: GoldenQuestion, record: dict) -> tuple[dict, BudgetExceeded | None]:
    """Fill in the missing verdicts; a budget stop returns what was judged so far."""
    record, errors = dict(record), {}
    for kind in needs_verdicts(q, record):
        before = judge.budget.spent_usd
        try:
            if kind == "correctness":
                verdict = judge.correctness(q.question, q.reference_answer, record["answer"])
            else:
                contexts = [Context(**c) for c in record["contexts"]]
                verdict = judge.faithfulness(contexts, record["answer"])
        except BudgetExceeded as exc:
            return record, exc
        except LLMError as exc:
            errors[kind] = _error(exc)
            continue
        record[kind] = {
            "verdict": verdict.model_dump(),
            "provider": judge.provider,
            "model": judge.model,
            "prompt_version": judge.prompt_version,
            "cost_usd": judge.budget.spent_usd - before,  # 0 for a cached verdict
        }
    if errors:
        record["judge_errors"] = errors
    else:
        record.pop("judge_errors", None)
    return record, None


def collect_answers(
    rag: httpx.Client,
    judge: Any,
    budget: Budget,
    questions: Sequence[GoldenQuestion],
    modes: Sequence[str],
    run_dir: Path,
    *,
    allow_same_provider_judge: bool = False,
) -> int:
    """Ask and judge every question in every mode into ``run_dir``; returns 1 on a budget
    stop or any error, else 0."""
    config = _open_run(rag, run_dir)
    if judge.provider == config["llm_provider"] and not allow_same_provider_judge:
        raise SystemExit(
            f"the judge ({judge.provider}:{judge.model}) comes from the generator's provider "
            f"({config['llm_provider']}); pass --allow-same-provider-judge to run it anyway"
        )
    reserve = ask_worst_case_usd(config)
    out = run_dir / "answers.jsonl"
    previous = latest(out)
    counts: Counter[str] = Counter()
    with out.open("a", encoding="utf-8") as f:
        try:
            for q in questions:
                for mode in modes:
                    record = previous.get((q.id, mode))
                    if record and "error" not in record:
                        if not needs_verdicts(q, record):
                            counts["recorded"] += 1
                            continue
                        counts["re-judged"] += 1
                    else:
                        budget.check(reserve)
                        try:
                            asked = _ask(rag, q.question, mode)
                        except (httpx.HTTPError, KeyError, ValueError) as exc:
                            _append(f, {"id": q.id, "mode": mode, "error": _error(exc)})
                            counts["errors"] += 1
                            continue
                        budget.record(asked["cost_usd"])
                        record = {"id": q.id, "mode": mode, **asked}
                        record |= {"correctness": None, "faithfulness": None}
                        counts["asked"] += 1
                    record, stop = _judge(judge, q, record)
                    if failed := len(record.get("judge_errors", {})):
                        counts["judge errors"] += failed
                    _append(f, record)
                    if stop:
                        raise stop
        except BudgetExceeded as exc:
            print(f"stopped: {exc}")
            counts["stopped"] += 1
    summary = ", ".join(f"{n} {what}" for what, n in sorted(counts.items()))
    print(f"{summary or 'nothing to do'} -> {out} · {budget.summary()}")
    return 1 if counts["stopped"] or counts["errors"] or counts["judge errors"] else 0


# --- records ------------------------------------------------------------------------


def _retrieval_record(q: GoldenQuestion, mode: str, raw: dict) -> dict:
    hits = [
        {
            "source": h["source_file"],
            "score": h["score"],
            "covered": sorted(covered_items(h["text"], q.evidence)),
            **{
                f"covered_{chars}": sorted(covered_items(h["text"][:chars], q.evidence))
                for _, chars in SNIPPETS
            },
        }
        for h in raw["hits"]
    ]
    return {
        "record": "retrieval",
        "id": q.id,
        "mode": mode,
        "category": q.category,
        "origin": q.origin,
        "split": q.split,
        "n_items": len(q.evidence),
        "top_score": hits[0]["score"] if hits else 0.0,
        "total_ms": raw["timings_ms"].get("total_ms"),
        "hits": hits,
    }


def _judge_id(entry: dict) -> str:
    return f"{entry['provider']}:{entry['model']} prompts {entry['prompt_version']}"


def _answer_record(q: GoldenQuestion, mode: str, raw: dict, grade: dict | None) -> dict:
    correctness, faithfulness = raw.get("correctness"), raw.get("faithfulness")
    verified = [c["supported"] for c in raw["citations"] if c.get("supported") is not None]
    return {
        "record": "answer",
        "id": q.id,
        "mode": mode,
        "category": q.category,
        "origin": q.origin,
        "split": q.split,
        "refused": raw["refused"],
        "refused_by": raw["refused_by"],
        "rating": correctness["verdict"]["rating"] if correctness else None,
        "supported": (
            [c["supported"] for c in faithfulness["verdict"]["claims"]] if faithfulness else None
        ),
        "unjudged": needs_verdicts(q, raw),
        "citations_verified": len(verified),
        "citations_supported": sum(verified),
        "cost_usd": raw["cost_usd"],
        "total_ms": raw["total_ms"],
        "judge_cost_usd": sum(e["cost_usd"] for e in (correctness, faithfulness) if e),
        "judges": sorted({_judge_id(e) for e in (correctness, faithfulness) if e}),
        "human_pass": grade["pass"] if grade else None,
    }


def build_records(run_dir: Path, golden_path: Path, grades_path: Path | None = None) -> list[dict]:
    """The run joined with the golden set: a header, then one record per question, mode
    and stage. A human grade joins only the exact answer it was given for."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    questions = load_golden(golden_path)
    grades = {
        (g["id"], g["mode"], g["answer_sha"]): g
        for g in (read_jsonl(grades_path) if grades_path else [])
    }
    header: dict = {
        "record": "run",
        "run_id": run["run_id"],
        "collected": run["started_at"][:10],
        "config": run["config"],
        "golden": {
            "rows": len(questions),
            "sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
            "categories": dict(Counter(q.category for q in questions)),
            "origins": dict(Counter(q.origin for q in questions)),
        },
        "retrieval": None,
        "answers": None,
    }
    records: list[dict] = []
    for stage, top_k in (("retrieval", TOP_K), ("answers", TOP_K_ASK)):
        raw = latest(run_dir / f"{stage}.jsonl")
        if not raw:
            continue
        modes, missing, rows = sorted({mode for _, mode in raw}), 0, []
        for mode in modes:
            for q in questions:
                r = raw.get((q.id, mode))
                if r is None or "error" in r:
                    missing += 1
                elif stage == "retrieval":
                    rows.append(_retrieval_record(q, mode, r))
                else:
                    grade = grades.get((q.id, mode, answer_sha(r["answer"])))
                    rows.append(_answer_record(q, mode, r, grade))
        header[stage] = {"modes": modes, "top_k": top_k, "missing": missing}
        if stage == "answers":
            header[stage] |= {
                "unjudged": sum(bool(r["unjudged"]) for r in rows),
                "judges": sorted({j for r in rows for j in r["judges"]}),
            }
        records += rows
    return [header, *records]


# --- score: retrieval ---------------------------------------------------------------


def _ranked(record: dict, field: str = "covered") -> list[frozenset[int]]:
    return [frozenset(hit[field]) for hit in record["hits"]]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def retrieval_scores(records: list[dict]) -> dict:
    per_metric: dict[str, Callable[[dict], float]] = {
        **{
            f"R@{k}": lambda r, k=k: metrics.recall_at_k(_ranked(r), r["n_items"], k)
            for k in RECALL_KS
        },
        "MRR@10": lambda r: metrics.mrr_at_k(_ranked(r), 10),
        "nDCG@10": lambda r: metrics.ndcg_at_k(_ranked(r), r["n_items"], 10),
    }
    return {"n": len(records), **{m: _mean([f(r) for r in records]) for m, f in per_metric.items()}}


def snippet_scores(records: list[dict]) -> dict:
    out: dict = {"n": len(records)}
    for k, chars in SNIPPETS:
        full = [metrics.recall_at_k(_ranked(r), r["n_items"], k) for r in records]
        cut = [
            metrics.recall_at_k(_ranked(r, f"covered_{chars}"), r["n_items"], k) for r in records
        ]
        out[f"R@{k} full"] = _mean(full)
        out[f"R@{k} {chars} chars"] = _mean(cut)
    return out


def _prf_dict(prf: PRF) -> dict:
    return {
        "precision": prf.precision,
        "recall": prf.recall,
        "f1": prf.f1,
        "tp": prf.tp,
        "fp": prf.fp,
        "fn": prf.fn,
        "tn": prf.tn,
        "precision_ci": metrics.wilson_interval(prf.tp, prf.tp + prf.fp),
        "recall_ci": metrics.wilson_interval(prf.tp, prf.tp + prf.fn),
    }


def _sweep_dict(points: list[SweepPoint]) -> list[dict]:
    # JSON has no infinity: null stands for the +inf end of the sweep (refuse everything).
    return [
        {
            "threshold": None if p.threshold == float("inf") else p.threshold,
            "precision": p.prf.precision,
            "recall": p.prf.recall,
            "f1": p.prf.f1,
        }
        for p in points
    ]


def refusal_scores(records: list[dict], configured: float | None) -> dict:
    """Gate-only refusals: the threshold is chosen on dev and measured on test."""

    def split(name: str) -> tuple[list[float], list[bool]]:
        rows = [r for r in records if r["split"] == name]
        return [r["top_score"] for r in rows], [r["category"] == "no_answer" for r in rows]

    def at(threshold: float, scores: list[float], should: list[bool]) -> dict:
        return _prf_dict(metrics.refusal_prf([s < threshold for s in scores], should))

    (dev_scores, dev_should), (test_scores, test_should) = split("dev"), split("test")
    dev_sweep = metrics.refusal_sweep(dev_scores, dev_should)
    out: dict = {
        "dev": {"n": len(dev_scores), "should_refuse": sum(dev_should)},
        "test": {"n": len(test_scores), "should_refuse": sum(test_should)},
        "sweeps": {
            "dev": _sweep_dict(dev_sweep),
            "test": _sweep_dict(metrics.refusal_sweep(test_scores, test_should)),
        },
    }
    try:
        chosen = metrics.choose_threshold(dev_sweep)
    except ValueError:
        chosen = None
    out["chosen_threshold"] = chosen
    out["chosen"] = None
    if chosen is not None:
        out["chosen"] = {
            "dev": at(chosen, dev_scores, dev_should),
            "test": at(chosen, test_scores, test_should),
        }
    out["configured_threshold"] = configured
    out["configured"] = None if configured is None else at(configured, test_scores, test_should)
    return out


def score_retrieval(header: dict, rows: list[dict]) -> dict:
    thresholds = header["config"].get("thresholds", {})
    scores: dict = {}
    for mode in header["retrieval"]["modes"]:
        mine = [r for r in rows if r["mode"] == mode]
        answerable = [r for r in mine if r["category"] != "no_answer"]
        scores[mode] = {
            "retrieval": {
                "all": retrieval_scores(answerable),
                **{
                    f"by_{key}": {
                        value: retrieval_scores([r for r in answerable if r[key] == value])
                        for value in sorted({r[key] for r in answerable})
                    }
                    for key in ("category", "origin", "split")
                },
            },
            "snippets": snippet_scores([r for r in answerable if r["split"] == "test"]),
            "refusal": refusal_scores(mine, thresholds.get(mode)),
        }
    return scores


# --- score: answers -----------------------------------------------------------------


def correct(record: dict) -> bool | None:
    """Correct end to end; None while an answer still lacks its correctness verdict."""
    if record["category"] == "no_answer":
        return record["refused"]
    if record["refused"]:
        return False
    if record["rating"] is None:
        return None
    return metrics.correctness_pass(CorrectnessVerdict(reasoning="", rating=record["rating"]))


def faithfulness(record: dict) -> float | None:
    if record["refused"] or record["supported"] is None:
        return None
    claims = [ClaimVerdict(claim="", reasoning="", supported=s) for s in record["supported"]]
    return metrics.faithfulness_score(FaithfulnessVerdict(claims=claims))


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least p % of them at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)] if ordered else None


def _rate(flags: Sequence[bool]) -> dict:
    k, n = sum(flags), len(flags)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci": metrics.wilson_interval(k, n)}


def answer_scores(rows: list[dict]) -> dict:
    flags = [c for r in rows if (c := correct(r)) is not None]
    ratings = [r["rating"] for r in rows if r["rating"] is not None and not r["refused"]]
    faithful = [f for r in rows if (f := faithfulness(r)) is not None]
    verified = sum(r["citations_verified"] for r in rows)
    latency = [r["total_ms"] for r in rows if r["total_ms"] is not None]
    return {
        "n": len(rows),
        "correct": _rate(flags),
        "mean_rating": _mean(ratings),
        "rated": len(ratings),
        "faithfulness": _mean(faithful),
        "fully_supported": _rate([f == 1.0 for f in faithful]),
        "citation_accuracy_self": (
            sum(r["citations_supported"] for r in rows) / verified if verified else None
        ),
        "cost_per_query": _mean([r["cost_usd"] for r in rows]),
        "cost_total": sum(r["cost_usd"] for r in rows),
        "p50_ms": percentile(latency, 50),
        "p95_ms": percentile(latency, 95),
        "judge_cost": sum(r["judge_cost_usd"] for r in rows),
    }


def score_answers(header: dict, rows: list[dict]) -> dict:
    scores: dict = {"modes": {}}
    for mode in header["answers"]["modes"]:
        mine = [r for r in rows if r["mode"] == mode]
        test = [r for r in mine if r["split"] == "test"]
        prf = metrics.refusal_prf(
            [r["refused"] for r in test], [r["category"] == "no_answer" for r in test]
        )
        by = Counter(r["refused_by"] for r in test if r["refused"])
        scores["modes"][mode] = {
            **answer_scores(mine),
            "by_category": {
                c: answer_scores([r for r in mine if r["category"] == c])
                for c in sorted({r["category"] for r in mine})
            },
            "refusal": {
                **_prf_dict(prf),
                "n": len(test),
                "should_refuse": sum(r["category"] == "no_answer" for r in test),
                "by_gate": by["gate"],
                "by_model": by["model"],
            },
        }
    graded = [r for r in rows if r["human_pass"] is not None and r["rating"] is not None]
    judge = [correct(r) for r in graded]
    human = [r["human_pass"] for r in graded]
    scores["agreement"] = {
        "n": len(graded),
        "raw": _mean([j == h for j, h in zip(judge, human, strict=True)]),
        "kappa": metrics.cohens_kappa(judge, human) if graded else None,
    }
    return scores


def score_records(records: list[dict]) -> dict:
    header, rows = records[0], records[1:]
    return {
        "retrieval": (
            score_retrieval(header, [r for r in rows if r["record"] == "retrieval"])
            if header["retrieval"]
            else None
        ),
        "answers": (
            score_answers(header, [r for r in rows if r["record"] == "answer"])
            if header["answers"]
            else None
        ),
    }


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
    rag_client: httpx.Client | None = None,
    judge: Any = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag_eval", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="query the RAG service into runs/<id>/")
    collect.add_argument("--stage", choices=["retrieval", "answers"], required=True)
    collect.add_argument("--modes", default=",".join(MODES), help="comma-separated")
    collect.add_argument("--rag", metavar="URL", help="the RAG API (default: $RAG_URL)")
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    collect.add_argument("--runs", type=Path, default=Path("runs"))
    collect.add_argument("--limit", type=int, help="only the first N questions (a smoke run)")
    collect.add_argument("--max-cost-usd", type=float, help="spending cap (answers: required)")
    collect.add_argument(
        "--judge",
        choices=["env", "fake"],
        default="env",
        help="env: JUDGE_PROVIDER/JUDGE_MODEL (default claude-haiku-4-5); fake: scripted, $0",
    )
    collect.add_argument("--allow-same-provider-judge", action="store_true")
    score = sub.add_parser("score", help="score a run or a records file; never calls anything")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id")
    source.add_argument("--records", type=Path, help="a rag-records.jsonl written by score")
    score.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    score.add_argument("--grades", type=Path, default=Path("golden/human_grades.jsonl"))
    score.add_argument("--runs", type=Path, default=Path("runs"))
    score.add_argument(
        "--out", type=Path, help="report directory (default: runs/<id>/report, or the records' own)"
    )
    args = parser.parse_args(argv)

    if args.command == "collect":
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        if unknown := set(modes) - set(MODES):
            parser.error(f"unknown modes: {sorted(unknown)}")
        if args.stage == "answers" and args.max_cost_usd is None:
            parser.error("--max-cost-usd is required for the answer stage")
        load_env(Path(".env"))
        url = args.rag or os.environ.get("RAG_URL")
        if not url and rag_client is None:
            parser.error("--rag or RAG_URL is required")
        questions = load_golden(args.golden)[: args.limit]
        run_dir = args.runs / args.run_id
        with rag_client or httpx.Client(base_url=url, timeout=120) as rag:
            if args.stage == "retrieval":
                return 1 if collect_retrieval(rag, questions, modes, run_dir) else 0
            budget = Budget(args.max_cost_usd)
            return collect_answers(
                rag,
                judge or _make_judge(args.judge, budget),
                budget,
                questions,
                modes,
                run_dir,
                allow_same_provider_judge=args.allow_same_provider_judge,
            )

    if args.records:
        records = read_jsonl(args.records)
        if not records or records[0].get("record") != "run":
            parser.error(f"{args.records} isn't a records file written by score")
        out = args.out or args.records.parent
    else:
        records = build_records(args.runs / args.run_id, args.golden, args.grades)
        out = args.out or args.runs / args.run_id / "report"
    written = report.write_rag(records, score_records(records), out, with_records=not args.records)
    print("\n".join(f"wrote {path}" for path in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
