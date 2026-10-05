"""Check the golden set (or, with --tasks, the agent task file) before anything is scored.

    python -m dataset.validate golden/golden_set.jsonl \\
        [--corpus ../services/rag/data/raw/toolchain_docs] [--rag http://localhost:8000] \\
        [--assign-splits]
    python -m dataset.validate tasks/agent_tasks.jsonl --tasks

Always, offline: every row against the schema (``schemas.GoldenQuestion``), unique
ids, ``verified: true`` on every row, and the counts per category, origin and split
against the target composition (warnings).

--corpus DIR     every quote occurs in its source file, inside one section: the RAG
                 loader leaves heading lines out of chunk text. A miss prints the
                 file's closest line.
--rag URL        every quote lies inside a chunk of the collection the RAG API
                 serves, which also catches a quote cut by a chunk boundary or lost
                 with a duplicate chunk. A quote in more than 2 chunks is flagged as
                 not specific enough: every chunk containing it counts as relevant.
--assign-splits  gives each row without a split one, per category toward 40 % dev
                 and 60 % test, and rewrites only those lines; assigned rows never move.
--tasks          the file holds agent tasks (``schemas.AgentTask``).

Exits 1 when there are errors; warnings don't fail the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import get_args

import httpx
from pydantic import BaseModel, ValidationError

from schemas import AgentTask, Category, GoldenQuestion, Origin
from textnorm import normalize_ws

CATEGORIES: tuple[str, ...] = get_args(Category)
ORIGINS: tuple[str, ...] = get_args(Origin)

# The golden set's target composition: 50 targeted rows (12 lookup, 14 multi_hop,
# 18 no_answer, 6 ambiguous) and 25 synthetic lookups.
TARGET_CATEGORIES = {"lookup": 37, "multi_hop": 14, "no_answer": 18, "ambiguous": 6}
TARGET_ORIGINS = {"targeted": 50, "synthetic": 25}
DEV_SHARE = 0.4
MAX_CHUNKS_PER_QUOTE = 2


@dataclass
class Row:
    line: int  # 1-based line number in the file
    data: dict | None  # None when the line isn't a JSON object

    def where(self) -> str:
        rid = self.data.get("id") if self.data else None
        return f"{rid} (line {self.line})" if isinstance(rid, str) else f"line {self.line}"


@dataclass
class Findings:
    errors: list[tuple[str, str]] = field(default_factory=list)  # (where, message)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)

    def error(self, where: str, message: str) -> None:
        self.errors.append((where, message))

    def warn(self, where: str, message: str) -> None:
        self.warnings.append((where, message))


# --- reading and the schema ---------------------------------------------------------


def read_rows(lines: list[str], findings: Findings) -> list[Row]:
    rows = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            findings.error(f"line {n}", f"invalid JSON ({exc.msg}, column {exc.colno})")
            data = None
        else:
            if not isinstance(data, dict):
                findings.error(f"line {n}", "not a JSON object")
                data = None
        rows.append(Row(line=n, data=data))
    return rows


def _loc(parts: Sequence[int | str]) -> str:
    """``("evidence", 0, "quotes", 1, "text")`` -> ``evidence[0].quotes[1].text``."""
    out = ""
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else part
    return out


def _describe(err: dict) -> str:
    loc = _loc(err["loc"])
    if loc == "split" and err["type"] == "missing":
        return "split is missing (run with --assign-splits)"
    message = err["msg"].removeprefix("Value error, ")
    return f"{loc}: {message}" if loc else message


def parse_rows[M: BaseModel](rows: list[Row], model: type[M], findings: Findings) -> list[M]:
    """Validate each row against ``model``; report duplicate ids and schema errors."""
    seen: dict[str, int] = {}
    parsed = []
    for row in rows:
        if row.data is None:
            continue
        rid = row.data.get("id")
        if isinstance(rid, str):
            if rid in seen:
                findings.error(row.where(), f"duplicate id (first on line {seen[rid]})")
            else:
                seen[rid] = row.line
        try:
            parsed.append(model.model_validate(row.data))
        except ValidationError as exc:
            for err in exc.errors():
                findings.error(row.where(), _describe(err))
    return parsed


def check_golden(rows: list[Row], findings: Findings) -> list[GoldenQuestion]:
    questions = parse_rows(rows, GoldenQuestion, findings)
    for q in questions:
        if not q.verified:
            findings.error(q.id, "verified is false: only checked rows belong in the set")
    return questions


# --- splits -------------------------------------------------------------------------


def _round_half_up(x: float) -> int:
    return int(x + 0.5)


def _with_split(data: dict, split: str) -> dict:
    """``data`` with ``split`` set, placed after ``origin`` when there is one."""
    out = {}
    for key, value in data.items():
        if key == "split":
            continue
        out[key] = value
        if key == "origin":
            out["split"] = split
    out.setdefault("split", split)
    return out


def assign_splits(rows: list[Row]) -> list[Row]:
    """Give each row without a split one, per category toward ``DEV_SHARE`` dev.

    Rows that already have a split keep it. The rows to fill are taken in an order
    fixed by a hash of their ids, so the dev rows aren't simply the first ones
    written. Returns the rows that got a split.
    """
    by_category: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.data is not None and row.data.get("category") in CATEGORIES:
            by_category[row.data["category"]].append(row)
    assigned = []
    for members in by_category.values():
        dev_target = _round_half_up(DEV_SHARE * len(members))
        dev = sum(1 for r in members if r.data.get("split") == "dev")
        todo = [
            r for r in members if r.data.get("split") is None and isinstance(r.data.get("id"), str)
        ]
        todo.sort(key=lambda r: hashlib.sha256(r.data["id"].encode()).hexdigest())
        for row in todo:
            split = "dev" if dev < dev_target else "test"
            dev += split == "dev"
            row.data = _with_split(row.data, split)
            assigned.append(row)
    return assigned


def write_splits(path: Path, lines: list[str], rows: list[Row]) -> list[Row]:
    """``assign_splits``, then rewrite only the lines of the rows that got a split."""
    assigned = assign_splits(rows)
    if assigned:
        for row in assigned:
            lines[row.line - 1] = json.dumps(row.data, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return assigned


# --- the corpus on disk -------------------------------------------------------------


@dataclass
class CorpusFile:
    lines: list[str]
    text: str  # the whole file, whitespace-normalized
    sections: list[str]  # each section's text without its heading line, normalized


def _sections(lines: list[str]) -> list[str]:
    """Section texts as the RAG markdown loader cuts them: a ``#`` line outside a code
    fence is a heading, which ends the section and belongs to no chunk's text."""
    sections: list[list[str]] = [[]]
    in_fence = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and stripped.startswith("#"):
            sections.append([])
            continue
        sections[-1].append(line)
    return [normalize_ws("\n".join(s)) for s in sections]


class Corpus:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._files: dict[str, CorpusFile | None] = {}

    def file(self, source: str) -> CorpusFile | None:
        if source not in self._files:
            path = self.root / source
            if path.is_file():
                lines = path.read_text(encoding="utf-8").splitlines()
                markdown = path.suffix in (".md", ".markdown")
                text = normalize_ws("\n".join(lines))
                sections = _sections(lines) if markdown else [text]
                self._files[source] = CorpusFile(lines, text, sections)
            else:
                self._files[source] = None
        return self._files[source]


def closest_line(lines: list[str], quote: str) -> tuple[int, str] | None:
    """The line holding the most of ``quote`` in order: where a near-miss came from."""
    best: tuple[float, int, str] | None = None
    for n, line in enumerate(lines, start=1):
        text = normalize_ws(line)
        if not text:
            continue
        blocks = SequenceMatcher(None, text, quote, autojunk=False).get_matching_blocks()
        score = sum(b.size for b in blocks) / len(quote)
        if best is None or score > best[0]:
            best = (score, n, text)
    return (best[1], best[2]) if best else None


def _short(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _quotes(questions: list[GoldenQuestion]):
    for q in questions:
        for i, item in enumerate(q.evidence):
            for j, quote in enumerate(item.quotes):
                yield f"{q.id} evidence[{i}].quotes[{j}] ({quote.source})", quote


def check_corpus(questions: list[GoldenQuestion], root: Path, findings: Findings) -> None:
    corpus = Corpus(root)
    n_quotes, n_bad, sources = 0, 0, set()
    for where, quote in _quotes(questions):
        n_quotes += 1
        sources.add(quote.source)
        needle = normalize_ws(quote.text)
        doc = corpus.file(quote.source)
        if doc is None:
            findings.error(where, f"no such file under {root}")
        elif needle not in doc.text:
            hint = closest_line(doc.lines, needle)
            near = f"; closest is line {hint[0]}: {_short(hint[1])!r}" if hint else ""
            findings.error(where, f"not in the file{near}")
        elif not any(needle in section for section in doc.sections):
            findings.error(where, "on or across a heading line, which no chunk holds")
        else:
            continue
        n_bad += 1
    if n_quotes:
        result = "all found" if not n_bad else f"{n_bad} not found"
        files = f"{len(sources)} file{'s' * (len(sources) != 1)}"
        findings.summary.append(f"corpus: {n_quotes} quotes in {files}, {result}")


# --- the served collection ----------------------------------------------------------


def _get(client: httpx.Client, path: str, **params: str) -> dict:
    response = client.get(path, params=params)
    response.raise_for_status()
    return response.json()


def check_rag(questions: list[GoldenQuestion], client: httpx.Client, findings: Findings) -> None:
    quotes = list(_quotes(questions))
    try:
        config = _get(client, "/v1/config")
        collection, index = config["collection"], config["index"]
        if index is None:
            findings.error("rag", f"collection {collection!r} isn't indexed")
            return
        documents = _get(client, "/v1/documents")["documents"]
        sources = {d["source_file"] for d in documents}
        chunks: list[tuple[str, str]] = []  # (source, normalized text) for the whole collection
        if quotes:
            for source in sorted(sources):
                body = _get(client, "/v1/chunks", source_file=source)
                chunks.extend((source, normalize_ws(c["text"])) for c in body["chunks"])
    except httpx.HTTPError as exc:
        findings.error("rag", f"{client.base_url}: {exc}")
        return

    for where, quote in quotes:
        if quote.source not in sources:
            findings.error(where, f"source isn't indexed in {collection}")
            continue
        needle = normalize_ws(quote.text)
        holders = [source for source, text in chunks if needle in text]
        if not holders:
            findings.error(where, f"no chunk of {collection} contains it")
            continue
        if quote.source not in holders:
            others = ", ".join(sorted(set(holders)))
            findings.warn(where, f"only chunks of other sources contain it ({others})")
        if len(holders) > MAX_CHUNKS_PER_QUOTE:
            spread = Counter(holders).most_common(3)
            listed = ", ".join(f"{s} ×{n}" for s, n in spread)
            more = ", …" if len(set(holders)) > 3 else ""
            findings.warn(where, f"in {len(holders)} chunks ({listed}{more}): not specific enough")

    chunking = f"{index['chunk_strategy']} {index['chunk_size']}/{index['chunk_overlap']}"
    findings.summary.append(
        f"rag: {len(quotes)} quotes checked against {collection} "
        f"({index['embedding_model']}; {chunking}; {index['chunks']:,} chunks)"
    )


# --- summaries ----------------------------------------------------------------------


def summarize_golden(
    path: Path, n_rows: int, questions: list[GoldenQuestion], findings: Findings
) -> None:
    by_category = Counter(q.category for q in questions)
    by_origin = Counter(q.origin for q in questions)
    by_split = Counter((q.category, q.split) for q in questions)

    parsed = f" ({len(questions)} parsed)" if len(questions) != n_rows else ""
    origins = " / ".join(f"{o} {by_origin[o]}" for o in ORIGINS)
    categories = ", ".join(f"{c} {by_category[c]}" for c in CATEGORIES)
    findings.summary.insert(0, f"{path}: {n_rows} rows{parsed} · {origins} · {categories}")
    table = [f"  {'category':<11}{'rows':>5}{'target':>8}{'dev':>5}{'test':>6}"]
    for c in CATEGORIES:
        dev, test = by_split[c, "dev"], by_split[c, "test"]
        table.append(f"  {c:<11}{by_category[c]:>5}{TARGET_CATEGORIES[c]:>8}{dev:>5}{test:>6}")
        expected = _round_half_up(DEV_SHARE * by_category[c])
        if abs(dev - expected) > 1:
            findings.warn("splits", f"{c} has {dev} dev / {test} test, {expected} dev expected")
    findings.summary[1:1] = table

    targets = dict(TARGET_CATEGORIES) | dict(TARGET_ORIGINS)
    actual = {c: by_category[c] for c in CATEGORIES} | {o: by_origin[o] for o in ORIGINS}
    if actual != targets:
        gaps = ", ".join(f"{k} {actual[k]}/{n}" for k, n in targets.items() if actual[k] != n)
        findings.warn("composition", f"{gaps} (rows/target; the target is 75 rows)")


def summarize_tasks(path: Path, n_rows: int, tasks: list[AgentTask], findings: Findings) -> None:
    by_category = sorted(Counter(t.category for t in tasks).items())
    parsed = f" ({len(tasks)} parsed)" if len(tasks) != n_rows else ""
    line = f"{path}: {n_rows} tasks{parsed}"
    if by_category:
        line += " · " + ", ".join(f"{c} {n}" for c, n in by_category)
    findings.summary.insert(0, line)


def _grouped(entries: list[tuple[str, str]]) -> list[str]:
    """Messages about the same place together (a quote's corpus and RAG findings),
    places in the order they first came up."""
    first: dict[str, int] = {}
    for where, _ in entries:
        first.setdefault(where, len(first))
    return [f"{w}: {m}" for w, m in sorted(entries, key=lambda e: first[e[0]])]


def report(findings: Findings) -> None:
    for line in _grouped(findings.errors):
        print(f"ERROR {line}")
    for line in _grouped(findings.warnings):
        print(f"WARN  {line}")
    for line in findings.summary:
        print(line)
    n_err, n_warn = len(findings.errors), len(findings.warnings)
    print(f"{n_err} error{'s' * (n_err != 1)}, {n_warn} warning{'s' * (n_warn != 1)}")


# --- command line -------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, rag_client: httpx.Client | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.validate",
        description="Check the golden set, or with --tasks the agent task file.",
    )
    parser.add_argument("file", type=Path, help="a JSONL file, one row per line")
    parser.add_argument("--corpus", type=Path, metavar="DIR", help="check quotes in the files")
    parser.add_argument("--rag", metavar="URL", help="check quotes in the served chunks")
    parser.add_argument(
        "--assign-splits", action="store_true", help="fill in missing splits, then check"
    )
    parser.add_argument("--tasks", action="store_true", help="the file holds agent tasks")
    args = parser.parse_args(argv)
    if args.tasks and (args.corpus or args.rag or args.assign_splits):
        parser.error("--tasks can't be combined with --corpus, --rag or --assign-splits")
    if not args.file.is_file():
        parser.error(f"no such file: {args.file}")
    if args.corpus and not args.corpus.is_dir():
        parser.error(f"no such directory: {args.corpus}")

    findings = Findings()
    lines = args.file.read_text(encoding="utf-8").splitlines()
    rows = read_rows(lines, findings)

    if args.tasks:
        tasks = parse_rows(rows, AgentTask, findings)
        summarize_tasks(args.file, len(rows), tasks, findings)
        report(findings)
        return 1 if findings.errors else 0

    if args.assign_splits:
        assigned = write_splits(args.file, lines, rows)
        if assigned:
            ids = ", ".join(sorted(f"{r.data['id']} {r.data['split']}" for r in assigned))
            print(f"assigned splits to {len(assigned)} rows: {ids}")

    questions = check_golden(rows, findings)
    if args.corpus:
        check_corpus(questions, args.corpus, findings)
    if args.rag:
        client = rag_client or httpx.Client(base_url=args.rag, timeout=30)
        with client:
            check_rag(questions, client, findings)
    summarize_golden(args.file, len(rows), questions, findings)
    report(findings)
    return 1 if findings.errors else 0


if __name__ == "__main__":
    sys.exit(main())
