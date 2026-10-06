"""Write evaluation reports: markdown for people, JSON for programs, SVG for the curves.

Everything here is a pure function of the records and their scores (no clock, no
environment), so re-scoring the same records reproduces every file byte for byte.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import svg

METRICS = ("R@1", "R@3", "R@5", "R@10", "MRR@10", "nDCG@10")


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(value: float | None, ci: list[float] | tuple[float, float]) -> str:
    return "n/a" if value is None else f"{value:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _rounded(value, key: str = ""):
    """Floats to 6 decimals, so the JSON stays stable and readable; thresholds keep full
    precision, since the gate refuses strictly below them and a rounded value can flip
    the question a threshold sits on."""
    if isinstance(value, float):
        return value if "threshold" in key else round(value, 6)
    if isinstance(value, dict):
        return {k: _rounded(v, k) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v, key) for v in value]
    return value


def threshold_text(t: float) -> str:
    """A threshold as it goes into the service's settings: every decimal it has."""
    return f"{t:.12f}".rstrip("0").rstrip(".")


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def rag_markdown(header: dict, scores: dict) -> str:
    cfg, golden = header["config"], header["golden"]
    index = cfg.get("index") or {}
    lines = [
        "# RAG evaluation",
        "",
        f"Run `{header['run_id']}`, collected {header['collected']}. Collection "
        f"`{cfg['collection']}`: {index.get('chunks', '?')} chunks "
        f"({index.get('chunk_strategy', '?')}, {index.get('chunk_size', '?')} characters, "
        f"{index.get('chunk_overlap', '?')} overlap), embedded with `{cfg['embedding_model']}`. "
        "Hybrid mode adds BM25 and a cross-encoder reranker to the dense search.",
        "",
        f"Golden set: {golden['rows']} questions (sha256 `{golden['sha256'][:12]}`): "
        f"{_counts(golden['categories'])}; {_counts(golden['origins'])}.",
    ]
    for stage, what in (("retrieval", "search"), ("answers", "answer")):
        if header[stage] and header[stage]["missing"]:
            lines += [
                "",
                f"**{header[stage]['missing']} question-mode pairs have no successful {what}** "
                f"and are left out of the {stage} numbers.",
            ]
    if header["answers"] and header["answers"]["unjudged"]:
        lines += [
            "",
            f"**{header['answers']['unjudged']} answers lack a judge verdict** and are left "
            "out of the numbers that need one.",
        ]
    if scores["retrieval"]:
        modes = header["retrieval"]["modes"]
        lines += _retrieval_section(header, scores["retrieval"], modes)
        lines += _snippet_section(scores["retrieval"], modes)
        lines += _refusal_section(scores["retrieval"], modes)
    if scores["answers"]:
        lines += _answers_section(header, scores["answers"])
    return "\n".join(lines) + "\n"


def _retrieval_section(header: dict, scores: dict, modes: list[str]) -> list[str]:
    first = scores[modes[0]]["retrieval"]["all"]["n"] if modes else 0
    lines = [
        "",
        "## Retrieval",
        "",
        f"The {first} answerable questions (every category but no_answer), top "
        f"{header['retrieval']['top_k']} hits from `/v1/search`. A hit is relevant to an "
        "evidence item if it contains one of the item's quotes; an item found by several hits "
        "counts once, at its first rank, so recall and nDCG measure the facts found, not the "
        "chunks.",
        "",
        *_table(
            ["mode", "n", *METRICS],
            [
                [m, str(s["n"]), *(_f(s[k]) for k in METRICS)]
                for m in modes
                for s in [scores[m]["retrieval"]["all"]]
            ],
        ),
    ]
    notes = {
        "origin": "Synthetic questions were drafted from one chunk and share its wording, "
        "which flatters retrieval: read them against the targeted ones.",
    }
    for key in ("category", "origin", "split"):
        lines += ["", f"### By {key}", ""]
        if key in notes:
            lines += [notes[key], ""]
        lines += _table(
            ["mode", key, "n", *METRICS],
            [
                [m, value, str(s["n"]), *(_f(s[k]) for k in METRICS)]
                for m in modes
                for value, s in scores[m]["retrieval"][f"by_{key}"].items()
            ],
        )
    return lines


def _snippet_section(scores: dict, modes: list[str]) -> list[str]:
    cols = ["R@3 full", "R@3 300 chars", "R@5 full", "R@5 150 chars"]
    n = scores[modes[0]]["snippets"]["n"] if modes else 0
    lines = [
        "",
        "## Snippet size for MCP search",
        "",
        "MCP `search` gives an agent each hit cut to its first characters. Evidence recall "
        f"on the test split ({n} answerable questions) with whole chunks and with the "
        "snippets an agent sees:",
        "",
        *_table(
            ["mode", "n", *cols],
            [
                [m, str(scores[m]["snippets"]["n"]), *(_f(scores[m]["snippets"][c]) for c in cols)]
                for m in modes
            ],
        ),
    ]
    hybrid = scores.get("hybrid", {}).get("snippets", {})
    if hybrid.get("R@5 150 chars") is not None and hybrid.get("R@3 300 chars") is not None:
        gain = 100 * (hybrid["R@5 150 chars"] - hybrid["R@3 300 chars"])
        verdict = "switch to 5 × 150" if gain >= 5 else "keep 3 × 300"
        lines += [
            "",
            "Rule: switch MCP from 3 hits of 300 characters to 5 of 150 only if that raises "
            f"hybrid recall by at least 5 points. Here 5 × 150 is {gain:+.1f} points: "
            f"**{verdict}**.",
        ]
    return lines


def _refusal_section(scores: dict, modes: list[str]) -> list[str]:
    rows = []
    for m in modes:
        r = scores[m]["refusal"]
        for label, threshold, result in (
            ("chosen on dev", r["chosen_threshold_config"], r["chosen"] and r["chosen"]["test"]),
            ("configured", r["configured_threshold"], r["configured"]),
        ):
            if threshold is None or result is None:
                rows.append([m, f"n/a ({label})", *["n/a"] * 7])
                continue
            rows.append(
                [
                    m,
                    f"{threshold_text(threshold)} ({label})",
                    *(str(result[k]) for k in ("tp", "fp", "fn", "tn")),
                    _ci(result["precision"], result["precision_ci"]),
                    _ci(result["recall"], result["recall_ci"]),
                    _f(result["f1"]),
                ]
            )
    test = scores[modes[0]]["refusal"]["test"] if modes else {"n": 0, "should_refuse": 0}
    return [
        "",
        "## Refusal gate",
        "",
        "The gate refuses before generating when the top hit's score is below the mode's "
        "threshold; a question should be refused if and only if it is no_answer. Each "
        "mode's threshold is chosen on the dev split (the highest F1, ties to the lower "
        f"threshold) and measured on the test split ({test['n']} questions, "
        f"{test['should_refuse']} to refuse), with 95 % Wilson intervals; the configured "
        "threshold is shown for comparison. A chosen threshold is printed as the value to "
        "configure: the dev score it was chosen at, truncated so that every decision stays "
        "the same (the gate refuses strictly below it, so rounding up would refuse that "
        "question). Gate only: refusals by the model itself come from the answer run.",
        "",
        *_table(["mode", "threshold", "TP", "FP", "FN", "TN", "precision", "recall", "F1"], rows),
        *_kinds_section(scores, modes),
        "",
        *(
            f"![Refusal gate, {m}: precision and recall against the threshold](refusal_{m}.svg)"
            for m in modes
        ),
    ]


KINDS_INTRO = (
    "No_answer questions come in kinds: near-miss (the docs cover a neighbouring feature), "
    "knows-elsewhere (a fact a model may know from other sources that the pinned docs "
    "don't state) and out-of-scope (pricing, roadmaps, benchmarks)."
)


def _count(k: int | None, n: int) -> str:
    return "n/a" if k is None else f"{k}/{n}"


def _kinds_section(scores: dict, modes: list[str]) -> list[str]:
    rows = [
        [
            m,
            kind,
            _f(k["median_top_score"]),
            _count(k["dev"]["refused_chosen"], k["dev"]["n"]),
            _count(k["test"]["refused_chosen"], k["test"]["n"]),
            _count(k["test"]["refused_configured"], k["test"]["n"]),
        ]
        for m in modes
        for kind, k in scores[m]["refusal"]["by_kind"].items()
    ]
    if not rows:
        return []
    return [
        "",
        "### Refusals by no_answer kind",
        "",
        KINDS_INTRO + " How many of each the gate refuses, as counts (each kind has only a "
        "few questions); the median is the kind's top score over both splits, and the dev "
        "rows are the ones the threshold was chosen on.",
        "",
        *_table(
            [
                "mode",
                "kind",
                "median top score",
                "dev refused (chosen)",
                "test refused (chosen)",
                "test refused (configured)",
            ],
            rows,
        ),
    ]


def _answer_kinds(per_mode: dict, modes: list[str]) -> list[str]:
    rows = [
        [m, kind, split, str(c["n"]), str(c["gate"]), str(c["model"]), str(c["answered"])]
        for m in modes
        for kind, by_split in per_mode[m]["no_answer_by_kind"].items()
        for split, c in by_split.items()
        if c["n"]
    ]
    if not rows:
        return []
    return [
        "",
        "### No_answer questions by kind",
        "",
        KINDS_INTRO + " Who declined each one: the gate (before generating), the model "
        "(after reading the contexts), or nobody (it was answered).",
        "",
        *_table(
            ["mode", "kind", "split", "n", "refused by gate", "refused by model", "answered"],
            rows,
        ),
    ]


def _money(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.5f}"


def _ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def _answers_section(header: dict, scores: dict) -> list[str]:
    cfg, info = header["config"], header["answers"]
    modes, judges = info["modes"], info["judges"]
    same = any(j.split(":")[0] == cfg["llm_provider"] for j in judges)
    intro = (
        f"`POST /v1/ask` with `top_k` {info['top_k']} in both modes, generated by "
        f"`{cfg['generation_model']}` ({cfg['llm_provider']}). Judge: "
        + (", ".join(f"`{j}`" for j in judges) or "none")
        + "."
        + (" **The judge comes from the generator's provider.**" if same else "")
        + (" More than one judge graded this run." if len(judges) > 1 else "")
    )
    per_mode = scores["modes"]
    lines = [
        "",
        "## Answers",
        "",
        intro,
        "An answerable question the system refused fails correctness without a judge call, and "
        "a no_answer question is correct if and only if it was refused; otherwise an answer "
        "passes at a rating of 4 or 5. Faithfulness is the share of an answer's claims that its "
        "retrieved contexts support, judged without the reference answer.",
        "",
        *_table(
            [
                "mode",
                "n",
                "correct [95 % CI]",
                "mean rating",
                "faithfulness",
                "fully supported",
                "citations (self-check)",
                "cost/query",
                "P50 ms",
                "P95 ms",
            ],
            [
                [
                    m,
                    str(a["n"]),
                    _ci(a["correct"]["rate"], a["correct"]["ci"]),
                    _f(a["mean_rating"]),
                    _f(a["faithfulness"]),
                    _ci(a["fully_supported"]["rate"], a["fully_supported"]["ci"]),
                    _f(a["citation_accuracy_self"]),
                    _money(a["cost_per_query"]),
                    _ms(a["p50_ms"]),
                    _ms(a["p95_ms"]),
                ]
                for m in modes
                for a in [per_mode[m]]
            ],
        ),
        "",
        "The mean rating is over the answers the judge rated (answerable and not refused). The "
        "judge rates a complete answer that adds correct detail beyond the reference 4 rather "
        "than 5, so the mean understates thorough answers; the pass rate is unaffected. "
        "Citations (self-check) is the pipeline's own verdict on its citations, n/a when it "
        "doesn't verify them.",
        "",
        "### Correct by category",
        "",
        *_table(
            ["mode", "category", "n", "correct [95 % CI]", "mean rating", "faithfulness"],
            [
                [
                    m,
                    c,
                    str(a["n"]),
                    _ci(a["correct"]["rate"], a["correct"]["ci"]),
                    _f(a["mean_rating"]),
                    _f(a["faithfulness"]),
                ]
                for m in modes
                for c, a in per_mode[m]["by_category"].items()
            ],
        ),
        "",
        "### Refusals end to end (test split)",
        "",
        "The gate and the model together: a refusal by either counts.",
        "",
        *_table(
            [
                "mode",
                "n",
                "TP",
                "FP",
                "FN",
                "TN",
                "precision",
                "recall",
                "F1",
                "by gate",
                "by model",
            ],
            [
                [
                    m,
                    str(r["n"]),
                    *(str(r[k]) for k in ("tp", "fp", "fn", "tn")),
                    _ci(r["precision"], r["precision_ci"]),
                    _ci(r["recall"], r["recall_ci"]),
                    _f(r["f1"]),
                    str(r["by_gate"]),
                    str(r["by_model"]),
                ]
                for m in modes
                for r in [per_mode[m]["refusal"]]
            ],
        ),
        *_answer_kinds(per_mode, modes),
        "",
        f"Spend in this run: RAG ${sum(per_mode[m]['cost_total'] for m in modes):.4f}, judge "
        f"${sum(per_mode[m]['judge_cost'] for m in modes):.4f} (a cached verdict costs nothing).",
    ]
    agreement = scores["agreement"]
    lines += ["", "### Judge agreement with a human", ""]
    if agreement["n"]:
        kappa = (
            "n/a (both raters constant)"
            if agreement["kappa"] is None
            else (f"{agreement['kappa']:.3f}")
        )
        lines.append(
            f"{agreement['n']} rated answers graded pass or fail by a person who didn't see the "
            f"judge's verdict: raw agreement {agreement['raw']:.3f}, Cohen's κ {kappa}. With "
            "this few items, κ is a coarse check."
        )
    else:
        lines.append("No human grades for this run yet.")
    return lines


def _points(sweep: list[dict]) -> list[svg.Point]:
    return [
        (math.inf if p["threshold"] is None else p["threshold"], p["precision"], p["recall"])
        for p in sweep
    ]


def write_rag(records: list[dict], scores: dict, out: Path, *, with_records: bool) -> list[Path]:
    """Write rag.md, rag.json, a refusal SVG per retrieval mode and (from a run) the records."""
    header = records[0]
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "rag.md": rag_markdown(header, scores),
        "rag.json": json.dumps(
            _rounded({"run": header, "scores": scores}), indent=2, sort_keys=True
        )
        + "\n",
    }
    for mode in (header["retrieval"] or {}).get("modes", []):
        r = scores["retrieval"][mode]["refusal"]
        panels = [
            (
                f"{split}: {r[split]['n']} questions, {r[split]['should_refuse']} to refuse",
                _points(r["sweeps"][split]),
            )
            for split in ("dev", "test")
        ]
        files[f"refusal_{mode}.svg"] = svg.refusal_chart(
            f"Refusal gate, {mode}: precision and recall against the threshold",
            panels,
            r["chosen_threshold_config"],
        )
    if with_records:
        files["rag-records.jsonl"] = "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records
        )
    written = []
    for name, text in files.items():
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
