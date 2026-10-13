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
        *_provenance(header),
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


def _stage_config(header: dict, stage: str) -> dict:
    # Records written before per-stage snapshots carry one config for the whole run.
    return header[stage].get("config", header["config"])


def _provenance(header: dict) -> list[str]:
    stages = [s for s in ("retrieval", "answers") if header[s]]
    if not stages:
        return []
    configs = {s: _stage_config(header, s) for s in stages}

    def thresholds(cfg: dict) -> str:
        t = cfg.get("thresholds", {})
        return " / ".join(threshold_text(t[m]) if m in t else "n/a" for m in ("dense", "hybrid"))

    rows = [
        ["collected", *(header[s].get("collected", header["collected"]) for s in stages)],
        ["`top_k` asked for", *(str(header[s]["top_k"]) for s in stages)],
        ["refusal thresholds (dense / hybrid)", *(thresholds(configs[s]) for s in stages)],
        [
            "generator",
            *(f"`{configs[s]['generation_model']}` ({configs[s]['llm_provider']})" for s in stages),
        ],
        [
            "citation verification",
            *("on" if configs[s].get("citation_verification") else "off" for s in stages),
        ],
    ]
    return [
        "",
        "Each stage records the service's settings when it starts, and every stage of a run "
        "serves the same collection:",
        "",
        *_table(["", *stages], rows),
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
    cfg, info = _stage_config(header, "answers"), header["answers"]
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
        "doesn't verify them. Faithfulness, fully supported and the citation self-check only "
        "cover the answers a mode gave, not its refusals, so a mode that refuses more of its "
        "harder questions can score higher on them: they compare how answers are grounded, not "
        "which mode does better end to end. Correctness is the end-to-end measure.",
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
        disagreements = agreement.get("disagreements", [])
        if disagreements:
            lines += [
                "",
                "Where they disagree (the judge passes a rating of 4 or 5):",
                "",
                *_table(
                    ["id", "mode", "category", "judge rating", "judge", "person", "person's note"],
                    [
                        [
                            d["id"],
                            d["mode"],
                            d["category"],
                            str(d["rating"]),
                            "pass" if d["judge_pass"] else "fail",
                            "pass" if d["human_pass"] else "fail",
                            d["note"].replace("|", "\\|") or "-",
                        ]
                        for d in disagreements
                    ],
                ),
            ]
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


# --- agent evaluation ---------------------------------------------------------------


def _secs(x: float | None) -> str:
    return "n/a" if x is None else f"{x:,.1f} s"


def _rate_text(r: dict) -> str:
    return "n/a" if r["rate"] is None else f"{_ci(r['rate'], r['ci'])} ({r['k']}/{r['n']})"


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _agent_provenance(header: dict, scores: dict, labels: list[str]) -> list[str]:
    info = header["labels"]

    def row(name: str, value) -> list[str]:
        return [name, *(value(info[label], scores["labels"][label]) for label in labels)]

    rows = [
        row("collected", lambda i, s: i["collected"]),
        row("supervisor", lambda i, s: f"`{i['config']['models']['supervisor']}`"),
        row("specialists", lambda i, s: f"`{i['config']['models']['specialist']}`"),
        row("reviewer, as routed", lambda i, s: f"`{i['config']['models']['reviewer']}`"),
        row("reviewer policy", lambda i, s: f"`{i['config']['reviewer_provider']}`"),
        row("memory", lambda i, s: f"`{i['config']['models']['memory']}`"),
        row("MOCK_LLM", lambda i, s: _yes(i["config"]["mock_llm"])),
        row("rag_* tools", lambda i, s: _yes(i["config"]["mcp_rag"])),
        row(
            "tasks",
            lambda i, s: (
                f"{s['n']} from `{i['tasks']['file']}` (sha256 `{i['tasks']['sha256'][:12]}`)"
            ),
        ),
        row("task judge", lambda i, s: ", ".join(f"`{j}`" for j in i["judges"]) or "none"),
        row(
            "spend: agent, judge",
            lambda i, s: (
                f"none (notional {_money(s['cost_total'])}), {_money(s['judge_cost'])}"
                if i["config"]["mock_llm"]
                else f"{_money(s['cost_total'])}, {_money(s['judge_cost'])}"
            ),
        ),
    ]
    return _table(["", *labels], rows)


def _agent_results(scores: dict, labels: list[str]) -> list[str]:
    per = scores["labels"]

    def row(name: str, value) -> list[str]:
        return [name, *(value(per[label]) for label in labels)]

    rows = [
        row("task success", lambda s: _rate_text(s["success"])),
        row("completed", lambda s: f"{s['completed']['k']}/{s['completed']['n']}"),
        row("tool-call must-call recall", lambda s: _f(s["must_call_recall"])),
        row("tool-call precision", lambda s: _f(s["precision"])),
        row("tool-call error rate", lambda s: _f(s["error_rate"])),
        row("tool calls per task", lambda s: _f(s["calls_per_task"])),
        row(
            "forbidden calls",
            lambda s: f"{s['forbidden_calls']} ({s['tasks_with_forbidden_calls']} tasks)",
        ),
        row("cost per task", lambda s: _money(s["cost_per_task"])),
        row("cost per completed task", lambda s: _money(s["cost_per_completed_task"])),
        row("wall-clock per task", lambda s: _secs(s["wall_clock_per_task"])),
        row("wall-clock per completed task", lambda s: _secs(s["wall_clock_per_completed_task"])),
        row("escalations", lambda s: f"{s['escalations']} ({s['tasks_escalated']} tasks)"),
        row("approvals given", lambda s: str(s["approvals"])),
        row(
            "reviewer mean score",
            lambda s: (
                "n/a"
                if s["review_score_mean"] is None
                else f"{s['review_score_mean']:.2f} ({s['reviewed_subtasks']} subtasks)"
            ),
        ),
        row(
            "rework rate",
            lambda s: (
                "n/a"
                if s["rework_rate"] is None
                else f"{s['rework_rate']:.3f} ({s['subtasks']} subtasks)"
            ),
        ),
        row("non-critical criteria passed", lambda s: _rate_text(s["non_critical"])),
    ]
    return _table(["", *labels], rows)


def _agent_tasks(scores: dict, label: str) -> list[str]:
    rows = []
    for p in scores["tasks"][label]:
        success = "n/a" if p["success"] is None else ("pass" if p["success"] else "fail")
        rows.append(
            [
                p["id"],
                p["category"],
                p["outcome"],
                success,
                _f(p["must_call_recall"]),
                _f(p["precision"]),
                _f(p["error_rate"]),
                str(p["forbidden_calls"]),
                _money(p["cost_usd"]),
                _secs(p["wall_clock_s"]),
                str(p["escalations"]),
                f"`{p['agent_task_id']}`",
            ]
        )
    header = ["task", "category", "outcome", "success", "must-call recall", "precision"]
    header += ["error rate", "forbidden", "cost", "wall-clock", "escalations", "agent task"]
    return _table(header, rows)


def agent_markdown(header: dict, scores: dict) -> str:
    labels = list(header["labels"])
    lines = [
        "# Agent evaluation",
        "",
        f"Run `{header['run_id']}`. Each task runs once per label through the agent API, "
        "under a user of its own so that no task's memories reach another's plan. Tool calls "
        "are scored from the agent's invocation log; the outcome comes from a rubric judge "
        "that sees only the outputs: the final answer, the files the run wrote and, when a "
        "criterion asks about the work as a whole, what each subtask reported.",
        "",
        "## Provenance",
        "",
        *_agent_provenance(header, scores, labels),
    ]
    if mocked := [label for label in labels if header["labels"][label]["config"]["mock_llm"]]:
        lines += [
            "",
            f"**{', '.join(mocked)}: the agent ran on MOCK_LLM**, replaying recorded replies. "
            "It prices them like the models they stand in for, so the costs below are "
            "notional: nothing was spent.",
        ]
    for label in labels:
        info = header["labels"][label]
        if info["missing"]:
            lines += ["", f"**{label}: {info['missing']} tasks have no record** (an API error)."]
        if info["unjudged"]:
            lines += [
                "",
                f"**{label}: {info['unjudged']} completed tasks have no verdict** and are left "
                "out of task success.",
            ]
    lines += [
        "",
        "## Results",
        "",
        *_agent_results(scores, labels),
        "",
        "Task success is over every task run, with a 95 % Wilson interval: a task that didn't "
        "complete fails without a judge call, and one that did succeeds when every critical "
        "criterion passed. Tool-call ratios are means over tasks, each task counting once: "
        "must-call recall is the share of required calls that a successful call made; "
        "precision the share of calls to tools the task names; the error rate the share that "
        "failed, were rejected or were rate limited. Cost is the agent's LLM spend plus what "
        "`rag_ask` spent in the RAG service; the judge's spend is not in it. \"Per completed "
        'task" means are over the tasks that completed. Escalations are the approvals the '
        "agent asked for; the harness approves each one (at most 5 per task). The reviewer's "
        "mean score is over the subtasks it scored, and the rework rate is the share of "
        "subtasks run more than once.",
    ]
    if pair := scores["cross_vs_same"]:
        only_cross = ", ".join(pair["only_first"]) or "none"
        only_same = ", ".join(pair["only_second"]) or "none"
        lines += [
            "",
            "## Cross vs same reviewer",
            "",
            f"On the {pair['n']} tasks judged under both labels: {pair['both']} succeeded under "
            f"both, {pair['neither']} under neither; only with the cross-provider reviewer: "
            f"{only_cross}; only with the same-provider reviewer: {only_same}.",
        ]
    categories = sorted({c for label in labels for c in scores["labels"][label]["by_category"]})
    by = [
        [
            c,
            *(
                _rate_text(scores["labels"][label]["by_category"].get(c, {"rate": None}))
                for label in labels
            ),
        ]
        for c in categories
    ]
    lines += ["", "## Task success by category", "", *_table(["category", *labels], by)]
    for label in labels:
        lines += ["", f"## Per task: {label}", "", *_agent_tasks(scores, label)]
    return "\n".join(lines) + "\n"


def write_agent(records: list[dict], scores: dict, out: Path, *, with_records: bool) -> list[Path]:
    """Write agent.md, agent.json and (from a run) the records."""
    header = records[0]
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "agent.md": agent_markdown(header, scores),
        "agent.json": json.dumps(
            _rounded({"run": header, "scores": scores}), indent=2, sort_keys=True
        )
        + "\n",
    }
    if with_records:
        files["agent-records.jsonl"] = "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records
        )
    written = []
    for name, text in files.items():
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
