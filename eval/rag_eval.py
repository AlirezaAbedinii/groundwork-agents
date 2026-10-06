"""Evaluate the RAG service over HTTP: collect raw runs once, score them for free.

    python -m rag_eval collect --stage retrieval --modes dense,hybrid \\
        --rag http://localhost:8000 --run-id 2026-10-06-minilm
    python -m rag_eval score --run-id 2026-10-06-minilm               # -> runs/<id>/report/
    python -m rag_eval score --run-id 2026-10-06-minilm --out reports  # publish
    python -m rag_eval score --records reports/rag-records.jsonl       # re-score in place

``collect`` is the only step that talks to the service. It records ``GET /v1/config``
and, for every golden question and mode, ``POST /v1/search`` with ``top_k`` 10 into
``runs/<id>/`` (not committed). It resumes: a question and mode already recorded
without an error is skipped, and a run refuses to continue against a service whose
config changed.

``score`` never calls anything. It joins the run with the golden set into compact
records (one per question and mode: which evidence items each hit covers, in full and
cut to MCP's snippet lengths, and the top score), computes every number from those
records with ``metrics``, and writes the report. The records file is enough to
re-score, so the published numbers can be recomputed without the run or the golden set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

import metrics
import report
from llm import load_env
from schemas import PRF, GoldenQuestion, SweepPoint
from textnorm import covered_items

TOP_K = 10
MODES = ("dense", "hybrid")
# MCP `search` cuts each hit to its first SNIPPET_CHARS characters; the snippet-size
# question is 3 hits of 300 chars against 5 hits of 150 (mcp/tool_models.py).
SNIPPETS = ((3, 300), (5, 150))
RECALL_KS = (1, 3, 5, 10)


# --- collect ------------------------------------------------------------------------


def load_golden(path: Path) -> list[GoldenQuestion]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldenQuestion.model_validate_json(line) for line in lines if line.strip()]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path, out = run_dir / "run.json", run_dir / "retrieval.jsonl"
    config = rag.get("/v1/config").raise_for_status().json()
    if run_path.is_file():
        recorded = json.loads(run_path.read_text(encoding="utf-8"))["config"]
        if recorded != config:
            raise SystemExit(
                f"{run_dir.name}: the service's /v1/config changed since this run started"
            )
    else:
        started = datetime.now(UTC).isoformat(timespec="seconds")
        run = {"run_id": run_dir.name, "started_at": started, "config": config}
        run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

    done = {(r["id"], r["mode"]) for r in _read_jsonl(out) if "error" not in r}
    searched = errors = 0
    with out.open("a", encoding="utf-8") as f:
        for q in questions:
            for mode in modes:
                if (q.id, mode) in done:
                    continue
                try:
                    record = {"id": q.id, "mode": mode, **_search(rag, q.question, mode)}
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    record = {"id": q.id, "mode": mode, "error": f"{type(exc).__name__}: {exc}"}
                    errors += 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                searched += 1
    skipped = len(questions) * len(modes) - searched
    print(f"{searched} searches ({errors} errors), {skipped} already recorded -> {out}")
    return errors


# --- records ------------------------------------------------------------------------


def build_records(run_dir: Path, golden_path: Path) -> list[dict]:
    """The run joined with the golden set: a header, then one record per question and mode."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    questions = load_golden(golden_path)
    latest = {(r["id"], r["mode"]): r for r in _read_jsonl(run_dir / "retrieval.jsonl")}
    modes = sorted({mode for _, mode in latest})
    records, missing = [], 0
    for mode in modes:
        for q in questions:
            raw = latest.get((q.id, mode))
            if raw is None or "error" in raw:
                missing += 1
                continue
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
            records.append(
                {
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
            )
    golden_bytes = golden_path.read_bytes()
    header = {
        "record": "run",
        "run_id": run["run_id"],
        "collected": run["started_at"][:10],
        "config": run["config"],
        "top_k": TOP_K,
        "modes": modes,
        "golden": {
            "rows": len(questions),
            "sha256": hashlib.sha256(golden_bytes).hexdigest(),
            "categories": dict(Counter(q.category for q in questions)),
            "origins": dict(Counter(q.origin for q in questions)),
        },
        "missing": missing,
    }
    return [header, *records]


# --- score --------------------------------------------------------------------------


def _ranked(record: dict, field: str = "covered") -> list[frozenset[int]]:
    return [frozenset(hit[field]) for hit in record["hits"]]


def _mean(values: list[float]) -> float | None:
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


def score_records(records: list[dict]) -> dict:
    header, rows = records[0], [r for r in records[1:] if r["record"] == "retrieval"]
    thresholds = header["config"].get("thresholds", {})
    scores: dict = {}
    for mode in header["modes"]:
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


# --- CLI ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, rag_client: httpx.Client | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag_eval", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="query the RAG service into runs/<id>/")
    collect.add_argument("--stage", choices=["retrieval"], required=True)
    collect.add_argument("--modes", default=",".join(MODES), help="comma-separated")
    collect.add_argument("--rag", metavar="URL", help="the RAG API (default: $RAG_URL)")
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    collect.add_argument("--runs", type=Path, default=Path("runs"))
    collect.add_argument("--limit", type=int, help="only the first N questions (a smoke run)")
    score = sub.add_parser("score", help="score a run or a records file; never calls anything")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id")
    source.add_argument("--records", type=Path, help="a rag-records.jsonl written by score")
    score.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    score.add_argument("--runs", type=Path, default=Path("runs"))
    score.add_argument(
        "--out", type=Path, help="report directory (default: runs/<id>/report, or the records' own)"
    )
    args = parser.parse_args(argv)

    if args.command == "collect":
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        if unknown := set(modes) - set(MODES):
            parser.error(f"unknown modes: {sorted(unknown)}")
        load_env(Path(".env"))
        url = args.rag or os.environ.get("RAG_URL")
        if not url and rag_client is None:
            parser.error("--rag or RAG_URL is required")
        questions = load_golden(args.golden)[: args.limit]
        rag = rag_client or httpx.Client(base_url=url, timeout=60)
        with rag:
            errors = collect_retrieval(rag, questions, modes, args.runs / args.run_id)
        return 1 if errors else 0

    if args.records:
        records = _read_jsonl(args.records)
        if not records or records[0].get("record") != "run":
            parser.error(f"{args.records} isn't a records file written by score")
        out = args.out or args.records.parent
    else:
        records = build_records(args.runs / args.run_id, args.golden)
        out = args.out or args.runs / args.run_id / "report"
    written = report.write_rag(records, score_records(records), out, with_records=not args.records)
    print("\n".join(f"wrote {path}" for path in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
