"""Evaluate the RAG service over HTTP: collect raw runs once, score them for free.

    python -m rag_eval collect --stage retrieval --rag http://localhost:8000 --run-id my-run
    python -m rag_eval collect --stage answers --rag http://localhost:8000 --run-id my-run \\
        --max-cost-usd 1.00
    python -m rag_eval score --run-id my-run                    # -> runs/my-run/report/
    python -m rag_eval score --run-id my-run --out reports      # publish
    python -m rag_eval score --records reports/rag-records.jsonl  # re-score in place
    python -m rag_eval stats --records reports/rag-records.jsonl  # paired statistics
    python -m rag_eval ablate --arms ablation/arms.json --run-id my-ablation  # dev only
    python -m rag_eval ablate --records runs/my-ablation/ablation-records.jsonl

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

``ablate`` compares retrieval configurations. Each arm in the arms file is a mode on a
service started with the arm's settings; its ``/v1/config`` must serve what the arm
expects, or nothing is searched. It searches the answerable questions of the given
splits (dev unless told otherwise, so test is searched only when asked) with ``top_k``
10, skips what an arm already recorded, and writes one table: per split and arm the
retrieval metrics, the search latency, and the paired bootstrap difference to the
reference arm, overall and per stratum.

``stats`` never calls anything either. It prints, from a records file, the answerable
questions each mode refused (by the gate and by the model), the paired comparison of
the two modes, and how many readings of each ambiguous question each mode retrieved.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
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
    AblationArm,
    AblationPlan,
    ClaimVerdict,
    Context,
    CorrectnessVerdict,
    FaithfulnessVerdict,
    GoldenQuestion,
    Share,
    SweepPoint,
)
from textnorm import covered_items

TOP_K = 10
TOP_K_ASK = 5
MODES = ("dense", "hybrid")
# MCP `search` returns whole passages while its result fits a budget, then excerpts.
# Each retrieval record says what every hit covers as `search` returns it for these
# top_k (its largest is the most `search` takes), sized by the server's own code.
SEARCH_TOP_KS = (3, 5, 8)
# The snippets `search` returned before: each hit's first 300 characters, compared
# with 150 characters for 5 hits.
SNIPPETS = ((3, 300), (5, 150))
RECALL_KS = (1, 3, 5, 10)
# What one /v1/ask can cost at most, reserved against the cap before the call: a
# generation with five contexts, and one citation check per context when the pipeline
# verifies citations. The real cost (the response's cost_usd) is what gets recorded.
ASK_INPUT_TOKENS, ASK_OUTPUT_TOKENS = 3000, 1000
JUDGED = ("correctness", "faithfulness")
# The kinds of no_answer question, named by the first word of the row's notes
# ("near-miss: rate limiting. ..."): the docs cover a neighbouring feature; a fact a
# model may know from elsewhere that the pinned docs don't state; or something outside
# any docs (pricing, roadmaps, benchmarks). Read once into the records, so re-scoring
# never parses notes.
NO_ANSWER_KINDS = ("near-miss", "knows-elsewhere", "out-of-scope")


# --- shared -------------------------------------------------------------------------


def _load_search_sizing() -> Any:
    """mcp/tool_models.py, where `search` sizes its result; loaded by path, since the
    mcp/ project is not a package. Measuring with the server's code keeps the table to
    exactly what an agent is sent."""
    path = Path(__file__).resolve().parents[1] / "mcp" / "tool_models.py"
    spec = importlib.util.spec_from_file_location("mcp_tool_models", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # pydantic resolves the models' types through it
    spec.loader.exec_module(module)
    return module


search_sizing = _load_search_sizing()


def load_golden(path: Path) -> list[GoldenQuestion]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldenQuestion.model_validate_json(line) for line in lines if line.strip()]


def no_answer_kind(q: GoldenQuestion) -> str | None:
    """A no_answer question's kind from its notes ("unlabelled" without one); None otherwise."""
    if q.category != "no_answer":
        return None
    kind = q.notes.split(":", 1)[0].strip()
    return kind if kind in NO_ANSWER_KINDS else "unlabelled"


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


STAGES = ("retrieval", "answers")
# What every stage of one run must share: they measure the same collection. Thresholds,
# the generator and citation checks may differ between stages (that's the point of
# calibrating on the retrieval stage and answering with the result).
SHARED_CONFIG = ("collection", "embedding_provider", "embedding_model", "index")


def read_run(run_dir: Path) -> dict:
    """``run.json`` with one config snapshot per stage. A run recorded before snapshots
    were per stage has a single ``config``: it stands for every stage the run holds."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if "stages" not in run:
        legacy = {"started_at": run["started_at"], "config": run["config"]}
        held = [s for s in STAGES if (run_dir / f"{s}.jsonl").is_file()]
        run = {"run_id": run["run_id"], "started_at": run["started_at"]}
        run["stages"] = {s: dict(legacy) for s in held}
    return run


def open_stage(rag: httpx.Client, run_dir: Path, stage: str) -> dict:
    """The service's config for ``stage``. The first collect of a stage records it as that
    stage's snapshot, which is never rewritten; later collects of the stage must find the
    same config, and every stage must serve the same collection as the others."""
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / "run.json"
    config = rag.get("/v1/config").raise_for_status().json()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if run_path.is_file():
        run = read_run(run_dir)
    else:
        run = {"run_id": run_dir.name, "started_at": now, "stages": {}}
    if stage in run["stages"]:
        if run["stages"][stage]["config"] != config:
            raise SystemExit(
                f"{run_dir.name}: the service's /v1/config changed since this run's "
                f"{stage} stage started"
            )
        return config
    for other, snapshot in run["stages"].items():
        if differ := [k for k in SHARED_CONFIG if snapshot["config"].get(k) != config.get(k)]:
            raise SystemExit(
                f"{run_dir.name}: the service doesn't serve what this run's {other} stage "
                f"measured ({', '.join(differ)} differ); start a new run"
            )
    run["stages"][stage] = {"started_at": now, "config": config}
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
    open_stage(rag, run_dir, "retrieval")
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
    config = open_stage(rag, run_dir, "answers")
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
    # The hits as `search` returns them with each top_k; the search was this question.
    returned = {
        k: search_sizing.fit_hits(q.question, mode, raw["hits"][:k])["hits"]
        for k in SEARCH_TOP_KS
    }
    hits = [
        {
            "source": h["source_file"],
            "score": h["score"],
            "covered": sorted(covered_items(h["text"], q.evidence)),
            **{
                f"covered_{chars}": sorted(covered_items(h["text"][:chars], q.evidence))
                for _, chars in SNIPPETS
            },
            **{
                f"covered_search{k}": sorted(covered_items(shown[rank]["text"], q.evidence))
                for k, shown in returned.items()
                if rank < len(shown)
            },
        }
        for rank, h in enumerate(raw["hits"])
    ]
    return {
        "record": "retrieval",
        "id": q.id,
        "mode": mode,
        "category": q.category,
        "no_answer_kind": no_answer_kind(q),
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
        "no_answer_kind": no_answer_kind(q),
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
        "human_note": grade.get("note", "") if grade else None,
    }


def build_records(run_dir: Path, golden_path: Path, grades_path: Path | None = None) -> list[dict]:
    """The run joined with the golden set: a header, then one record per question, mode
    and stage. A human grade joins only the exact answer it was given for."""
    run = read_run(run_dir)
    questions = load_golden(golden_path)
    grades = {
        (g["id"], g["mode"], g["answer_sha"]): g
        for g in (read_jsonl(grades_path) if grades_path else [])
    }
    header: dict = {
        "record": "run",
        "run_id": run["run_id"],
        "collected": run["started_at"][:10],
        # The shared settings (collection, embedding model, chunking), from the first stage.
        "config": next(iter(run["stages"].values()))["config"],
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
        snapshot = run["stages"][stage]
        header[stage] = {
            "modes": modes,
            "top_k": top_k,
            "missing": missing,
            "collected": snapshot["started_at"][:10],
            "config": snapshot["config"],
        }
        if stage == "answers":
            header[stage] |= {
                "unjudged": sum(bool(r["unjudged"]) for r in rows),
                "judges": sorted({j for r in rows for j in r["judges"]}),
            }
        records += rows
    return [header, *records]


def stage_config(header: dict, stage: str) -> dict:
    """A stage's config snapshot (records written before per-stage snapshots have one)."""
    return header[stage].get("config", header["config"])


# --- score: retrieval ---------------------------------------------------------------


def _ranked(record: dict, field: str = "covered") -> list[frozenset[int]]:
    return [frozenset(hit[field]) for hit in record["hits"]]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


# Each retrieval metric of one question's record, as the report names it.
RETRIEVAL_METRICS: dict[str, Callable[[dict], float]] = {
    **{
        f"R@{k}": lambda r, k=k: metrics.recall_at_k(_ranked(r), r["n_items"], k)
        for k in RECALL_KS
    },
    "MRR@10": lambda r: metrics.mrr_at_k(_ranked(r), 10),
    "nDCG@10": lambda r: metrics.ndcg_at_k(_ranked(r), r["n_items"], 10),
}


def retrieval_scores(records: list[dict]) -> dict:
    return {
        "n": len(records),
        **{m: _mean([f(r) for r in records]) for m, f in RETRIEVAL_METRICS.items()},
    }


def snippet_scores(records: list[dict]) -> dict:
    out: dict = {"n": len(records)}
    for k, chars in SNIPPETS:
        full = [metrics.recall_at_k(_ranked(r), r["n_items"], k) for r in records]
        cut = [
            metrics.recall_at_k(_ranked(r, f"covered_{chars}"), r["n_items"], k) for r in records
        ]
        out[f"R@{k} full"] = _mean(full)
        out[f"R@{k} {chars} chars"] = _mean(cut)
    for k in SEARCH_TOP_KS:
        field = f"covered_search{k}"
        full = [metrics.recall_at_k(_ranked(r), r["n_items"], k) for r in records]
        out.setdefault(f"R@{k} full", _mean(full))
        # Records written before the field existed have it on no hit: no value. A hit
        # without it in newer records is one `search` left out, so it shows nothing.
        returned = [
            metrics.recall_at_k(
                [frozenset(hit.get(field, ())) for hit in r["hits"]], r["n_items"], k
            )
            for r in records
        ]
        known = any(field in hit for r in records for hit in r["hits"])
        out[f"R@{k} as returned"] = _mean(returned) if known else None
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


def config_value(threshold: float, scores: Sequence[float]) -> float:
    """The threshold to put in the service's settings: ``threshold`` truncated to six
    decimals, or to more if that would move a recorded score across it, so it refuses
    exactly what ``threshold`` refuses. (Rounding up could refuse the question the
    threshold was chosen at, which sits exactly on it.)"""
    for digits in (6, 9, 12):
        value = math.floor(threshold * 10**digits) / 10**digits
        if not any(value <= s < threshold for s in scores):
            return value
    return threshold


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
    out["chosen_threshold_config"] = (
        None if chosen is None else config_value(chosen, [r["top_score"] for r in records])
    )
    out["chosen"] = None
    if chosen is not None:
        out["chosen"] = {
            "dev": at(chosen, dev_scores, dev_should),
            "test": at(chosen, test_scores, test_should),
        }
    out["configured_threshold"] = configured
    out["configured"] = None if configured is None else at(configured, test_scores, test_should)
    out["by_kind"] = _refusals_by_kind(records, chosen, configured)
    return out


def _kinds(records: list[dict]) -> list[str]:
    found = {r["no_answer_kind"] for r in records if r["category"] == "no_answer"}
    return [k for k in (*NO_ANSWER_KINDS, "unlabelled") if k in found]


def _refusals_by_kind(records: list[dict], chosen: float | None, configured: float | None) -> dict:
    """How many no_answer questions of each kind the gate refuses, per split (counts only:
    a kind has a handful of questions)."""

    def refused(rows: list[dict], threshold: float | None) -> int | None:
        return None if threshold is None else sum(r["top_score"] < threshold for r in rows)

    out = {}
    for kind in _kinds(records):
        rows = [r for r in records if r["category"] == "no_answer" and r["no_answer_kind"] == kind]
        dev = [r for r in rows if r["split"] == "dev"]
        test = [r for r in rows if r["split"] == "test"]
        out[kind] = {
            "median_top_score": statistics.median(r["top_score"] for r in rows),
            "dev": {"n": len(dev), "refused_chosen": refused(dev, chosen)},
            "test": {
                "n": len(test),
                "refused_chosen": refused(test, chosen),
                "refused_configured": refused(test, configured),
            },
        }
    return out


def score_retrieval(header: dict, rows: list[dict]) -> dict:
    thresholds = stage_config(header, "retrieval").get("thresholds", {})
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
        no_answer = [r for r in mine if r["category"] == "no_answer"]
        scores["modes"][mode] = {
            **answer_scores(mine),
            "no_answer_by_kind": {
                kind: {
                    split: {
                        "n": len(rows),
                        "gate": sum(r["refused_by"] == "gate" for r in rows),
                        "model": sum(r["refused_by"] == "model" for r in rows),
                        "answered": sum(not r["refused"] for r in rows),
                    }
                    for split in ("dev", "test")
                    for rows in [
                        [
                            r
                            for r in no_answer
                            if r["no_answer_kind"] == kind and r["split"] == split
                        ]
                    ]
                }
                for kind in _kinds(no_answer)
            },
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
        "disagreements": [
            {
                "id": r["id"],
                "mode": r["mode"],
                "category": r["category"],
                "rating": r["rating"],
                "judge_pass": j,
                "human_pass": h,
                "note": r.get("human_note") or "",
            }
            for r, j, h in sorted(
                zip(graded, judge, human, strict=True), key=lambda t: (t[0]["mode"], t[0]["id"])
            )
            if j != h
        ],
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


# --- stats: answerable refusals and the modes compared question by question --------
# Printed by ``stats`` from a records file; the published report doesn't include them yet.

# Each statistic is given over all the rows it applies to, then within each stratum.
STRATA = ("category", "split", "origin")


def _stratified(
    rows: list[dict], fn: Callable[..., Any], *columns: list, strata: Sequence[str] = STRATA
) -> dict:
    """``fn`` of the columns over all ``rows``, then per stratum (column i is ``rows[i]``)."""
    return {
        "all": fn(*columns),
        **{
            f"by_{key}": metrics.per_stratum([r[key] for r in rows], fn, *columns) for key in strata
        },
    }


def _share(share: Share) -> dict:
    ci = metrics.wilson_interval(share.k, share.n)
    return {"k": share.k, "n": share.n, "rate": share.rate, "ci": ci}


def _false_refusals(refused_by: list[str | None]) -> dict:
    found = metrics.false_refusals(refused_by)
    return {cause: _share(getattr(found, cause)) for cause in ("refused", *metrics.REFUSERS)}


def false_refusal_stats(rows: list[dict], modes: list[str]) -> dict:
    """Per mode, the answerable questions refused, by the gate and by the model."""
    out = {}
    for mode in modes:
        answerable = [r for r in rows if r["mode"] == mode and r["category"] != "no_answer"]
        if any(r["refused"] != (r["refused_by"] is not None) for r in answerable):
            raise ValueError(f"{mode}: a record's refused and refused_by disagree")
        out[mode] = _stratified(answerable, _false_refusals, [r["refused_by"] for r in answerable])
    return out


def _pairs(rows: list[dict], a: str, b: str) -> list[tuple[dict, dict]]:
    """Each question's rows in modes a and b, for the questions both have, by id."""
    by_key = {(r["id"], r["mode"]): r for r in rows}
    ids = {r["id"] for r in rows if r["mode"] == a} & {r["id"] for r in rows if r["mode"] == b}
    return [(by_key[i, a], by_key[i, b]) for i in sorted(ids)]


def mcnemar_stats(rows: list[dict], a: str, b: str) -> dict | None:
    """Exact McNemar on end-to-end correctness, every question, a against b.

    Correctness as the report scores it (a no_answer question is correct iff refused); a
    question still lacking a verdict in either mode is left out of the pairs.
    """
    pairs = [(x, y) for x, y in _pairs(rows, a, b) if None not in (correct(x), correct(y))]
    if not pairs:
        return None
    return _stratified(
        [x for x, _ in pairs],
        lambda p, q: dataclasses.asdict(metrics.mcnemar_exact(p, q)),
        [correct(x) for x, _ in pairs],
        [correct(y) for _, y in pairs],
    )


def bootstrap_stats(rows: list[dict], a: str, b: str, *, resamples: int, seed: int) -> dict:
    """Per retrieval metric, the paired bootstrap of b - a over the answerable questions."""
    pairs = [(x, y) for x, y in _pairs(rows, a, b) if x["category"] != "no_answer"]
    if not pairs:
        return {}

    def boot(p: list[float], q: list[float]) -> dict:
        found = metrics.paired_bootstrap(p, q, resamples=resamples, seed=seed)
        return dataclasses.asdict(found)

    return {
        name: _stratified(
            [x for x, _ in pairs], boot, [f(x) for x, _ in pairs], [f(y) for _, y in pairs]
        )
        for name, f in RETRIEVAL_METRICS.items()
    }


def _outcome(record: dict | None) -> str:
    if record is None:
        return "not asked"
    if record["refused"]:
        return f"refused by {record['refused_by']}"
    return "not judged" if record["rating"] is None else f"rated {record['rating']}"


def ambiguous_stats(retrieval: list[dict], answers: list[dict]) -> list[dict]:
    """Per ambiguous question and mode, its readings (evidence items) in the top 5 of the
    recorded search, as many contexts as ``/ask`` was run with, and the answer's outcome."""
    answer = {(r["id"], r["mode"]): r for r in answers}
    out: dict[str, dict] = {}
    for r in sorted(retrieval, key=lambda r: r["id"]):
        if r["category"] != "ambiguous":
            continue
        found = metrics.items_found_at_k(_ranked(r), r["n_items"], TOP_K_ASK)
        entry = out.setdefault(r["id"], {"id": r["id"], "readings": r["n_items"], "modes": {}})
        entry["modes"][r["mode"]] = {
            "found": found.k,
            "outcome": _outcome(answer.get((r["id"], r["mode"]))),
        }
    return list(out.values())


def paired_stats(
    records: list[dict],
    *,
    resamples: int = metrics.BOOTSTRAP_RESAMPLES,
    seed: int = metrics.BOOTSTRAP_SEED,
) -> dict:
    """Every statistic ``stats`` prints, from the records alone.

    The paired comparison is hybrid against dense (differences are hybrid - dense) and
    is None unless the stage holds both modes.
    """
    header, rows = records[0], records[1:]
    answers = [r for r in rows if r["record"] == "answer"]
    retrieval = [r for r in rows if r["record"] == "retrieval"]
    a, b = MODES

    def both(stage: str) -> bool:
        return bool(header[stage]) and {a, b} <= set(header[stage]["modes"])

    return {
        "modes": [a, b],
        "false_refusals": (
            false_refusal_stats(answers, header["answers"]["modes"]) if header["answers"] else None
        ),
        "mcnemar": mcnemar_stats(answers, a, b) if both("answers") else None,
        "bootstrap": (
            bootstrap_stats(retrieval, a, b, resamples=resamples, seed=seed)
            if both("retrieval")
            else None
        ),
        "ambiguous": ambiguous_stats(retrieval, answers),
    }


# --- ablate: retrieval configurations compared --------------------------------------
# Each arm is one retrieval configuration (AblationArm): a mode on a service started with
# the arm's settings. Only the answerable questions are searched, since only they have
# evidence to find, with the same top_k as the retrieval stage. Raw searches go to
# runs/<id>/arms/<arm>.jsonl and each arm's /v1/config to runs/<id>/arms.json; the table
# is computed from compact records, like the report, so it can be recomputed without
# the services.

# Strata of the ablation table: the question's category, and how it words its fact.
ABLATION_STRATA = ("category", "phrasing")
SPLITS = ("dev", "test")
# Sent once to each arm before its questions and not recorded: the service loads its
# models and builds its indexes on first use, which would otherwise be timed.
WARM_UP_QUERY = "How do I configure the cache directory?"
_MISSING = object()


def load_arms(path: Path) -> AblationPlan:
    return AblationPlan.model_validate_json(path.read_text(encoding="utf-8"))


def _served(config: dict, key: str) -> Any:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def arm_mismatches(arm: AblationArm, config: dict) -> list[str]:
    """The values the arm expects that ``config`` (a /v1/config body) doesn't serve."""
    out = []
    for key, expected in arm.expect.items():
        served = _served(config, key)
        if served is _MISSING:
            out.append(f"{key}: not served, expected {expected!r}")
        elif served != expected:
            out.append(f"{key}: served {served!r}, expected {expected!r}")
    return out


def open_arm(rag: httpx.Client, arm: AblationArm, run_dir: Path) -> dict:
    """The arm's service config, checked against the arm. The first collect of an arm
    records it in arms.json; later ones must find the same config and mode."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "arms.json"
    arms = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config = rag.get("/v1/config").raise_for_status().json()
    if wrong := arm_mismatches(arm, config):
        raise SystemExit(f"{arm.name}: {rag.base_url} doesn't serve this arm: " + "; ".join(wrong))
    if arm.name in arms:
        held = arms[arm.name]
        if held["mode"] != arm.mode or held["config"] != config:
            raise SystemExit(
                f"{run_dir.name}: arm {arm.name} was recorded with another mode or "
                "/v1/config; start a new run or rename the arm"
            )
        return config
    now = datetime.now(UTC).isoformat(timespec="seconds")
    arms[arm.name] = {"mode": arm.mode, "started_at": now, "config": config}
    path.write_text(json.dumps(arms, indent=2) + "\n", encoding="utf-8")
    return config


def _arm_path(run_dir: Path, arm: AblationArm) -> Path:
    return run_dir / "arms" / f"{arm.name}.jsonl"


def unsearched(
    arm: AblationArm, questions: Sequence[GoldenQuestion], run_dir: Path
) -> list[GoldenQuestion]:
    """The questions the arm hasn't recorded without an error."""
    done = {qid for (qid, _), r in latest(_arm_path(run_dir, arm)).items() if "error" not in r}
    return [q for q in questions if q.id not in done]


def collect_arm(
    rag: httpx.Client, arm: AblationArm, todo: Sequence[GoldenQuestion], run_dir: Path
) -> int:
    """Search ``todo`` with the arm after checking its service; returns the number of
    errors."""
    out = _arm_path(run_dir, arm)
    open_arm(rag, arm, run_dir)
    _search(rag, WARM_UP_QUERY, arm.mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with out.open("a", encoding="utf-8") as f:
        for q in todo:
            try:
                record = {"id": q.id, "mode": arm.mode, **_search(rag, q.question, arm.mode)}
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                record = {"id": q.id, "mode": arm.mode, "error": _error(exc)}
                errors += 1
            _append(f, record)
    print(f"{arm.name}: {len(todo)} searches ({errors} errors) -> {out}")
    return errors


def ablation_records(run_dir: Path, plan: AblationPlan, golden_path: Path) -> list[dict]:
    """The arms' searches joined with the golden set: a header naming every recorded arm
    with its config, then one record per arm and question."""
    questions = {q.id: q for q in load_golden(golden_path)}
    path = run_dir / "arms.json"
    held = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    arms = [arm for arm in plan.arms if arm.name in held]
    if plan.reference not in held:
        raise SystemExit(f"{run_dir.name}: the reference arm {plan.reference} has no searches")
    header = {
        "record": "ablation",
        "run_id": run_dir.name,
        "reference": plan.reference,
        "top_k": TOP_K,
        "golden": {
            "rows": len(questions),
            "sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        },
        "arms": [
            {"name": a.name, "change": a.change, "mode": a.mode, "config": held[a.name]["config"]}
            for a in arms
        ],
    }
    records = []
    for arm in arms:
        for (qid, _), raw in sorted(latest(_arm_path(run_dir, arm)).items()):
            if "error" in raw:
                continue
            q = questions[qid]
            records.append(
                {
                    "record": "arm",
                    "arm": arm.name,
                    "id": q.id,
                    "split": q.split,
                    "category": q.category,
                    "phrasing": q.phrasing or "unlabelled",
                    "origin": q.origin,
                    "n_items": len(q.evidence),
                    "hits": [
                        {
                            "covered": sorted(covered_items(h["text"], q.evidence)),
                            "score": h["score"],
                        }
                        for h in raw["hits"]
                    ],
                    "timings_ms": raw["timings_ms"],
                }
            )
    return [header, *records]


def _latency(rows: list[dict], stage: str) -> dict | None:
    values = [r["timings_ms"][stage] for r in rows if stage in r["timings_ms"]]
    if not values:
        return None
    return {"p50": percentile(values, 50), "p95": percentile(values, 95), "n": len(values)}


def ablation_scores(
    records: list[dict],
    *,
    resamples: int = metrics.BOOTSTRAP_RESAMPLES,
    seed: int = metrics.BOOTSTRAP_SEED,
) -> dict:
    """Per split and arm: the retrieval metrics, the search's latency (the service's own
    timing, and its rerank stage alone), and the paired bootstrap of arm - reference on
    every metric, over all the questions both searched and per stratum."""
    header, rows = records[0], records[1:]
    reference = header["reference"]

    def boot(p: list[float], q: list[float]) -> dict:
        found = metrics.paired_bootstrap(p, q, resamples=resamples, seed=seed)
        return dataclasses.asdict(found)

    out: dict = {"reference": reference, "splits": {}}
    for split in SPLITS:
        in_split = [r for r in rows if r["split"] == split]
        if not in_split:
            continue
        ref = {r["id"]: r for r in in_split if r["arm"] == reference}
        per_arm: dict = {}
        for arm in header["arms"]:
            mine = [r for r in in_split if r["arm"] == arm["name"]]
            if not mine:
                continue
            entry = {
                **retrieval_scores(mine),
                "latency_ms": _latency(mine, "total_ms"),
                "rerank_ms": _latency(mine, "rerank"),
                "vs_reference": None,
            }
            pairs = [(ref[r["id"]], r) for r in mine if r["id"] in ref]
            if arm["name"] != reference and pairs:
                entry["vs_reference"] = {
                    name: _stratified(
                        [x for _, x in pairs],
                        boot,
                        [f(x) for x, _ in pairs],
                        [f(y) for _, y in pairs],
                        strata=ABLATION_STRATA,
                    )
                    for name, f in RETRIEVAL_METRICS.items()
                }
            per_arm[arm["name"]] = entry
        out["splits"][split] = per_arm
    return out


# --- CLI ----------------------------------------------------------------------------


def _make_judge(kind: str, budget: Budget) -> Any:
    if kind == "fake":
        from judge_fake import FakeJudge

        return FakeJudge()
    from judge import Judge

    return Judge.from_env(budget=budget)


def _connect(url: str) -> httpx.Client:
    return httpx.Client(base_url=url, timeout=120)


def ablate(args: argparse.Namespace, parser: argparse.ArgumentParser, connect) -> int:
    """Collect what the chosen arms haven't searched yet, then write the table."""
    if args.resamples < 1:
        parser.error("--resamples must be at least 1")
    if args.records:
        records = read_jsonl(args.records)
        if not records or records[0].get("record") != "ablation":
            parser.error(f"{args.records} isn't an ablation records file written by ablate")
        out = args.out or args.records.parent
    else:
        if not args.run_id:
            parser.error("--run-id is required with --arms")
        plan = load_arms(args.arms)
        only = [a.strip() for a in (args.only or "").split(",") if a.strip()]
        if unknown := set(only) - {arm.name for arm in plan.arms}:
            parser.error(f"not in {args.arms}: {sorted(unknown)}")
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        if not splits or (unknown := set(splits) - set(SPLITS)):
            parser.error(f"--splits takes {', '.join(SPLITS)}")
        load_env(Path(".env"))
        default_url = args.rag or os.environ.get("RAG_URL")
        questions = [
            q
            for q in load_golden(args.golden)
            if q.split in splits and q.category != "no_answer"
        ]
        run_dir = args.runs / args.run_id
        errors = 0
        for arm in plan.arms:
            if only and arm.name not in only:
                continue
            todo = unsearched(arm, questions, run_dir)
            if not todo:  # an arm with nothing left to search never contacts its service
                print(f"{arm.name}: all {len(questions)} questions already recorded")
                continue
            url = arm.rag or default_url
            if not url:
                parser.error(f"{arm.name}: no URL (set the arm's rag, --rag or RAG_URL)")
            with connect(url) as rag:
                errors += collect_arm(rag, arm, todo, run_dir)
        records = ablation_records(run_dir, plan, args.golden)
        out = args.out or run_dir
    found = ablation_scores(records, resamples=args.resamples, seed=args.seed)
    written = report.write_ablation(records, found, out, with_records=not args.records)
    print("\n".join(f"wrote {path}" for path in written))
    return 1 if not args.records and errors else 0


def main(
    argv: Sequence[str] | None = None,
    *,
    rag_client: httpx.Client | None = None,
    judge: Any = None,
    connect: Callable[[str], httpx.Client] = _connect,
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
    collect.add_argument("--ids", help="only these golden ids, comma-separated (a pilot)")
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
    stats = sub.add_parser(
        "stats",
        help="print refusals of answerable questions and the paired mode comparison; "
        "never calls anything",
    )
    stats.add_argument("--records", type=Path, required=True, help="a rag-records.jsonl")
    stats.add_argument("--resamples", type=int, default=metrics.BOOTSTRAP_RESAMPLES)
    stats.add_argument("--seed", type=int, default=metrics.BOOTSTRAP_SEED)
    ablate_cmd = sub.add_parser(
        "ablate",
        help="search the answerable questions with each retrieval configuration (arm) in "
        "an arms file and compare them in one table",
    )
    arms_source = ablate_cmd.add_mutually_exclusive_group(required=True)
    arms_source.add_argument("--arms", type=Path, help="the arms file (JSON: reference, arms)")
    arms_source.add_argument(
        "--records", type=Path, help="an ablation-records.jsonl to re-score; calls nothing"
    )
    ablate_cmd.add_argument("--run-id", help="with --arms: runs/<id>/ holds the searches")
    ablate_cmd.add_argument("--only", help="collect only these arms, comma-separated")
    ablate_cmd.add_argument(
        "--splits", default="dev", help="splits to search, comma-separated (default: dev)"
    )
    ablate_cmd.add_argument("--rag", metavar="URL", help="for arms naming no URL ($RAG_URL)")
    ablate_cmd.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    ablate_cmd.add_argument("--runs", type=Path, default=Path("runs"))
    ablate_cmd.add_argument(
        "--out", type=Path, help="table directory (default: runs/<id>, or the records' own)"
    )
    ablate_cmd.add_argument("--resamples", type=int, default=metrics.BOOTSTRAP_RESAMPLES)
    ablate_cmd.add_argument("--seed", type=int, default=metrics.BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    if args.command == "ablate":
        return ablate(args, parser, connect)

    if args.command == "stats":
        if args.resamples < 1:
            parser.error("--resamples must be at least 1")
        records = read_jsonl(args.records)
        if not records or records[0].get("record") != "run":
            parser.error(f"{args.records} isn't a records file written by score")
        found = paired_stats(records, resamples=args.resamples, seed=args.seed)
        print(report.stats_markdown(records[0], found), end="")
        return 0

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
        questions = load_golden(args.golden)
        if args.ids:
            wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
            if unknown := wanted - {q.id for q in questions}:
                parser.error(f"not in the golden set: {sorted(unknown)}")
            questions = [q for q in questions if q.id in wanted]
        questions = questions[: args.limit]
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
