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


def _rounded(value):
    """Floats to 6 decimals, so the JSON stays stable and readable."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    return value


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def rag_markdown(header: dict, scores: dict) -> str:
    cfg, golden, modes = header["config"], header["golden"], header["modes"]
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
    if header["missing"]:
        lines += [
            "",
            f"**{header['missing']} question-mode pairs have no successful search** "
            "and are left out of every number below.",
        ]

    first = scores[modes[0]]["retrieval"]["all"]["n"] if modes else 0
    lines += [
        "",
        "## Retrieval",
        "",
        f"The {first} answerable questions (every category but no_answer), top "
        f"{header['top_k']} hits from `/v1/search`. A hit is relevant to an evidence item "
        "if it contains one of the item's quotes; an item found by several hits counts once, "
        "at its first rank, so recall and nDCG measure the facts found, not the chunks.",
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

    lines += _snippet_section(scores, modes)
    lines += _refusal_section(scores, modes)
    return "\n".join(lines) + "\n"


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
            ("chosen on dev", r["chosen_threshold"], r["chosen"] and r["chosen"]["test"]),
            ("configured", r["configured_threshold"], r["configured"]),
        ):
            if threshold is None or result is None:
                rows.append([m, f"n/a ({label})", *["n/a"] * 7])
                continue
            rows.append(
                [
                    m,
                    f"{threshold:.3f} ({label})",
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
        "threshold is shown for comparison. Gate only: refusals by the model itself come "
        "from the answer run.",
        "",
        *_table(["mode", "threshold", "TP", "FP", "FN", "TN", "precision", "recall", "F1"], rows),
        "",
        *(
            f"![Refusal gate, {m}: precision and recall against the threshold](refusal_{m}.svg)"
            for m in modes
        ),
    ]


def _points(sweep: list[dict]) -> list[svg.Point]:
    return [
        (math.inf if p["threshold"] is None else p["threshold"], p["precision"], p["recall"])
        for p in sweep
    ]


def write_rag(records: list[dict], scores: dict, out: Path, *, with_records: bool) -> list[Path]:
    """Write rag.md, rag.json, one refusal SVG per mode and (from a run) rag-records.jsonl."""
    header = records[0]
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "rag.md": rag_markdown(header, scores),
        "rag.json": json.dumps(
            _rounded({"run": header, "scores": scores}), indent=2, sort_keys=True
        )
        + "\n",
    }
    for mode in header["modes"]:
        r = scores[mode]["refusal"]
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
            r["chosen_threshold"],
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
